"""T-512, T-513 — the irreducible noise floor (FR-754, FR-755, OQ-4).

Both estimators are exercised against hand-built frames with a known answer,
then against the shared synthetic corpus for the shape of a real run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.evaluate.noise_floor import (
    DEFAULT_MAX_GRID_GAP,
    NoiseFloorError,
    NoiseFloorEstimate,
    NoiseFloorReport,
    dnf_incident_floor,
    error_relative_to_floor,
    estimate_noise_floor,
    teammate_pair_floor,
)
from tests.fixtures.races import synthetic_corpus


def _row(**kwargs):
    base = {
        "season": 2020,
        "round": 1,
        "constructor_id": "team_a",
        "classification_outcome": "classified",
        "status_raw": "Finished",
        "grid_position": 1,
        "classified_position": 1,
    }
    base.update(kwargs)
    return base


class TestDnfIncidentFloor:
    def test_matches_a_hand_counted_share(self):
        frame = pd.DataFrame(
            [
                _row(status_raw="Finished"),
                _row(status_raw="Finished"),
                _row(status_raw="Collision"),
                _row(status_raw="Engine"),
            ]
        )
        estimate = dnf_incident_floor(frame)
        assert estimate.value == pytest.approx(0.5)
        assert estimate.n_observations == 4

    def test_refuses_an_empty_corpus(self):
        with pytest.raises(NoiseFloorError, match="empty"):
            dnf_incident_floor(pd.DataFrame({"status_raw": pd.Series([], dtype=str)}))

    def test_refuses_a_frame_missing_status_raw(self):
        with pytest.raises(NoiseFloorError, match="missing"):
            dnf_incident_floor(pd.DataFrame({"other": [1]}))


class TestTeammatePairFloor:
    def test_zero_variance_when_teammates_always_finish_identically_spaced(self):
        # Every pair finishes exactly where they started (grid == finish),
        # so any variance in the diffs across pairs must come from the pairs
        # differing from each other, not from noise within a pair. Two pairs
        # with the same 1-position gap: residual (within-pair) variance is 0.
        rows = []
        for rnd in range(1, 6):
            rows.append(_row(round=rnd, grid_position=1, classified_position=1))
            rows.append(_row(round=rnd, grid_position=2, classified_position=2))
        frame = pd.DataFrame(rows)
        estimate = teammate_pair_floor(frame, max_grid_gap=3)
        assert estimate.value == pytest.approx(0.0, abs=1e-9)

    def test_a_grid_gap_beyond_the_threshold_is_excluded(self):
        rows = [
            _row(round=1, grid_position=1, classified_position=1),
            _row(round=1, grid_position=10, classified_position=10),
        ]
        frame = pd.DataFrame(rows)
        with pytest.raises(NoiseFloorError, match="too few"):
            teammate_pair_floor(frame, max_grid_gap=3)

    def test_dnf_rows_do_not_enter_the_pairing(self):
        rows = [
            _row(round=r, grid_position=1, classified_position=1) for r in range(1, 6)
        ] + [
            _row(round=r, grid_position=2, classified_position=2, constructor_id="team_a")
            for r in range(1, 6)
        ]
        # A third, retired driver on a different team must not corrupt a
        # same-team pair count.
        rows.append(
            _row(
                round=1,
                constructor_id="team_b",
                classification_outcome="not_classified",
                status_raw="Engine",
                grid_position=5,
                classified_position=5,
            )
        )
        frame = pd.DataFrame(rows)
        estimate = teammate_pair_floor(frame, max_grid_gap=3)
        assert estimate.n_observations == 5

    def test_refuses_a_negative_grid_gap(self):
        with pytest.raises(NoiseFloorError, match="non-negative"):
            teammate_pair_floor(pd.DataFrame([_row()]), max_grid_gap=-1)


class TestEstimateNoiseFloor:
    def test_headline_is_the_wider_of_the_two_bounds(self):
        raw = synthetic_corpus(seasons=(2013, 2014, 2015, 2016), rounds=5)
        report = estimate_noise_floor(raw)
        assert set(report.estimates) == {"dnf_incident_bound", "near_identical_teammate_variance"}
        assert report.headline_value == max(e.value for e in report.estimates.values())
        assert report.headline_method in report.estimates

    def test_report_serialises_to_json_compatible_dict(self):
        import json

        raw = synthetic_corpus(seasons=(2013, 2014, 2015, 2016), rounds=5)
        report = estimate_noise_floor(raw)
        json.dumps(report.as_dict())  # must not raise

    def test_both_estimates_are_bounded_between_zero_and_one(self):
        raw = synthetic_corpus(seasons=(2013, 2014, 2015, 2016), rounds=5)
        report = estimate_noise_floor(raw)
        for estimate in report.estimates.values():
            assert 0.0 <= estimate.value <= 1.0


class TestErrorRelativeToFloor:
    def _report(self, floor_positions: float) -> NoiseFloorReport:
        teammate = NoiseFloorEstimate(
            method="near_identical_teammate_variance",
            value=0.3,
            n_observations=100,
            description="synthetic",
            position_units=floor_positions,
        )
        return NoiseFloorReport(
            estimates={"near_identical_teammate_variance": teammate},
            headline_method="near_identical_teammate_variance",
            headline_value=0.3,
        )

    def test_matches_hand_computation(self):
        context = error_relative_to_floor(5.0, self._report(3.0))
        assert context["model_mae_positions"] == 5.0
        assert context["floor_positions"] == 3.0
        assert context["model_error_minus_floor_positions"] == pytest.approx(2.0)

    def test_a_model_at_the_floor_has_zero_room_left(self):
        context = error_relative_to_floor(3.0, self._report(3.0))
        assert context["model_error_minus_floor_positions"] == pytest.approx(0.0)

    def test_refuses_a_report_with_no_position_unit_estimate(self):
        dnf_only = NoiseFloorReport(
            estimates={
                "dnf_incident_bound": NoiseFloorEstimate(
                    method="dnf_incident_bound",
                    value=0.1,
                    n_observations=10,
                    description="synthetic",
                )
            },
            headline_method="dnf_incident_bound",
            headline_value=0.1,
        )
        with pytest.raises(NoiseFloorError, match="position-unit"):
            error_relative_to_floor(5.0, dnf_only)
