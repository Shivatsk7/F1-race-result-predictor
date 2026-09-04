"""T-505..T-509 — calibration of simulated probabilities (FR-701..706).

The unit-level tests check each piece (a reliability curve, a Brier score, a
baseline, the threshold label) against hand-computed numbers. The one
integration test runs the whole pipeline — train, fit simulation parameters,
simulate every holdout race, build the report — on a small synthetic corpus,
offline, to prove the pieces actually compose.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from f1pred.corpus import CorpusWindow
from f1pred.dataset import experiment
from f1pred.evaluate.calibration import (
    CALIBRATION_ERROR_THRESHOLD,
    CalibrationError,
    brier_score,
    brier_score_by_race,
    build_calibration_report,
    calibration_status,
    fit_event_baselines,
    interval_coverage,
    max_reliability_gap,
    reliability_curve,
    simulate_holdout,
)
from f1pred.evaluate.split import ChronologicalSplit
from f1pred.models.ranker import Hyperparameters, train
from f1pred.simulate.montecarlo import fit_simulation_parameters, pace_noise_std_from_interval
from tests.fixtures.races import synthetic_corpus

SEASONS = tuple(range(2013, 2021))  # eight seasons


class TestReliabilityCurve:
    def test_perfectly_calibrated_probabilities_have_zero_gap(self):
        # All five rows predict 0.2, and exactly one in five actually occurs:
        # the bin's mean prediction and observed frequency agree exactly.
        probabilities = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2])
        outcomes = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0])
        curve = reliability_curve(probabilities, outcomes, n_bins=10)
        assert max_reliability_gap(curve, min_count=1) == pytest.approx(0.0, abs=1e-9)

    def test_refuses_mismatched_lengths(self):
        with pytest.raises(CalibrationError, match="same length"):
            reliability_curve(pd.Series([0.1]), pd.Series([0.0, 1.0]))

    def test_refuses_empty_input(self):
        with pytest.raises(CalibrationError, match="no rows"):
            reliability_curve(pd.Series([], dtype=float), pd.Series([], dtype=float))

    def test_empty_bins_do_not_pollute_the_gap(self):
        # Every probability lands in the top bin; the rest are unoccupied.
        # min_count=1 here isolates "empty bins are excluded" from the
        # separate thin-bin filter tested below.
        probabilities = pd.Series([0.95, 0.95, 0.95])
        outcomes = pd.Series([0.0, 1.0, 1.0])
        curve = reliability_curve(probabilities, outcomes, n_bins=10)
        gap = max_reliability_gap(curve, min_count=1)
        assert gap == pytest.approx(abs(0.95 - 2 / 3), abs=1e-9)

    def test_a_bin_below_min_count_is_excluded_even_though_occupied(self):
        # One row in the top bin, disagreeing wildly with its own prediction:
        # exactly the shape that produced a 0.499 "worst gap" from a single
        # win observation on the real corpus. The default min_count excludes
        # it; a lower one does not.
        probabilities = pd.Series([0.1] * 20 + [0.95])
        outcomes = pd.Series([0.0] * 20 + [0.0])
        curve = reliability_curve(probabilities, outcomes, n_bins=10)
        assert max_reliability_gap(curve) == pytest.approx(0.1, abs=1e-9)
        assert max_reliability_gap(curve, min_count=1) == pytest.approx(0.95, abs=1e-9)

    def test_all_bins_below_min_count_gives_nan(self):
        probabilities = pd.Series([0.1, 0.9])
        outcomes = pd.Series([0.0, 1.0])
        curve = reliability_curve(probabilities, outcomes, n_bins=10)
        assert np.isnan(max_reliability_gap(curve))


class TestBrierScore:
    def test_matches_hand_computation(self):
        probabilities = pd.Series([0.2, 0.8])
        outcomes = pd.Series([0.0, 1.0])
        assert brier_score(probabilities, outcomes) == pytest.approx(0.04)

    def test_a_perfect_forecaster_scores_zero(self):
        probabilities = pd.Series([0.0, 1.0, 0.0, 1.0])
        outcomes = pd.Series([0.0, 1.0, 0.0, 1.0])
        assert brier_score(probabilities, outcomes) == pytest.approx(0.0)

    def test_refuses_mismatched_lengths(self):
        with pytest.raises(CalibrationError, match="same length"):
            brier_score(pd.Series([0.1]), pd.Series([0.0, 1.0]))


class TestCalibrationStatus:
    def test_within_threshold_is_calibrated(self):
        assert calibration_status(CALIBRATION_ERROR_THRESHOLD - 0.01) == "CALIBRATED"

    def test_at_threshold_is_calibrated(self):
        assert calibration_status(CALIBRATION_ERROR_THRESHOLD) == "CALIBRATED"

    def test_beyond_threshold_is_uncalibrated(self):
        assert calibration_status(CALIBRATION_ERROR_THRESHOLD + 0.01) == "UNCALIBRATED"

    def test_an_unmeasured_nan_is_uncalibrated_not_calibrated(self):
        # FR-705 covers calibration that is "unmeasured or exceeds threshold" —
        # a NaN must not read as a pass by falling through a numeric compare.
        assert calibration_status(float("nan")) == "UNCALIBRATED"


class TestFitEventBaselines:
    def test_climatological_rate_matches_the_training_frame(self):
        frame = pd.DataFrame(
            {"target_position": [1, 2, 3, 1, 5], "grid_position": [1, 2, 3, 1, 5]}
        )
        baselines = fit_event_baselines(frame, event="win")
        assert baselines.climatological_rate == pytest.approx(2 / 5)

    def test_grid_probability_falls_back_to_climatological_for_an_unseen_slot(self):
        frame = pd.DataFrame(
            {"target_position": [1, 2, 3], "grid_position": [1, 2, 3]}
        )
        baselines = fit_event_baselines(frame, event="podium")
        assert baselines.grid_probability(99) == pytest.approx(baselines.climatological_rate)

    def test_refuses_an_unknown_event(self):
        frame = pd.DataFrame({"target_position": [1], "grid_position": [1]})
        with pytest.raises(CalibrationError, match="unknown event"):
            fit_event_baselines(frame, event="fastest_lap")

    def test_refuses_an_empty_frame(self):
        with pytest.raises(CalibrationError, match="empty"):
            fit_event_baselines(pd.DataFrame({"target_position": [], "grid_position": []}), event="win")


class TestCalibrationReportAlwaysCarriesAStatus:
    """T-509: no path through this module can produce a report with no
    calibration status attached — the render layer has one field to check."""

    def test_as_dict_always_has_status_fields(self):
        _, report = _small_calibration_report()
        payload = report.as_dict()
        assert "any_uncalibrated" in payload
        for event_payload in payload["events"].values():
            assert "status" in event_payload
        assert "status" in payload["interval_coverage"]


def _small_calibration_report():
    raw = synthetic_corpus(seasons=SEASONS, rounds=4)
    split = ChronologicalSplit(
        window=CorpusWindow(width=len(SEASONS), earliest_available=SEASONS[0]),
        holdout_seasons=3,
        season_end_dates={s: date(s, 12, 1) for s in SEASONS},
    )
    built = experiment(raw)
    model = train(
        built.frame,
        split=split,
        features=built.feature_names,
        params=Hyperparameters(n_estimators=15, max_depth=3),
        name="test_ranker",
    )
    training_raw, _ = split.split(raw)
    pace_std = pace_noise_std_from_interval(
        half_width=(model.interval.upper_offset - model.interval.lower_offset) / 2,
        coverage=model.interval.coverage,
    )
    params = fit_simulation_parameters(training_raw, pace_noise_std=pace_std)
    simulated = simulate_holdout(built, model, params, split, n_simulations=150, seed=3)
    training_frame, _ = split.split(built.frame)
    report = build_calibration_report(
        simulated,
        training_frame=training_frame,
        stated_interval_coverage=model.interval.coverage,
    )
    return simulated, report


class TestEndToEndCalibration:
    """The whole pipeline, offline: train, fit, simulate, report."""

    @pytest.fixture(scope="class")
    def simulated_and_report(self):
        return _small_calibration_report()

    @pytest.fixture(scope="class")
    def simulated(self, simulated_and_report):
        return simulated_and_report[0]

    @pytest.fixture(scope="class")
    def report(self, simulated_and_report):
        return simulated_and_report[1]

    def test_the_stated_interval_is_on_the_same_scale_as_the_race_it_describes(
        self, simulated
    ):
        # Regression test for a real bug: calling predict_with_interval on the
        # whole holdout at once ranks every driver 1..len(holdout) as if it
        # were one giant race, instead of 1..field_size per actual race.
        # `simulate_race`'s own outputs stay in range regardless (a race's
        # finishing order is local to that race by construction), but the
        # *interval* — computed as point-prediction plus a fixed offset, then
        # clipped to the frame it was ranked against — silently inherited the
        # wrong frame's scale, and every race's stated band ended up far
        # outside that race's actual field. Observed containment was exactly
        # 0.0 on the real corpus before this was fixed.
        for (_, _), race in simulated.groupby(["season", "round"]):
            field_size = len(race)
            assert race["predicted_position_lower"].between(1, field_size).all()
            assert race["predicted_position_upper"].between(1, field_size).all()

    def test_every_event_type_is_reported(self, report):
        assert set(report.events) == {"win", "podium", "points"}

    def test_reliability_curves_have_the_right_shape(self, report):
        for event in report.events.values():
            assert {"bin", "mean_predicted", "observed_frequency", "count"} <= set(
                event.reliability_curve.columns
            )

    def test_brier_scores_are_bounded(self, report):
        for event in report.events.values():
            for interval in (
                event.brier_model,
                event.brier_climatological,
                event.brier_grid,
            ):
                assert 0.0 <= interval.point <= 1.0

    def test_coverage_is_a_rate_between_zero_and_one(self, report):
        assert 0.0 <= report.interval_coverage.point <= 1.0

    def test_race_and_row_counts_are_recorded(self, report):
        assert report.n_holdout_races == 3 * 4  # 3 holdout seasons, 4 rounds each
        assert report.n_holdout_rows > 0

    def test_report_serialises_to_json_compatible_dict(self, report):
        import json

        json.dumps(report.as_dict(), default=str)  # must not raise
