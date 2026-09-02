"""T-216 — the telemetry benefit as a separate arm (FR-213, FR-214, FR-215, AC-19).

What is worth testing here is not whether telemetry helps. That is a number,
the corpus decides it, and it is recorded in `results/`. What is worth testing
is that the number cannot be produced dishonestly:

* both arms rest on one holdout, so a "benefit" cannot be a difference in races;
* the delta is paired inside the bootstrap, so it cannot be widened into a null
  by subtracting two independent intervals;
* the reduced season count reaches the record, so the arm's score cannot be
  read as if it came from the whole corpus;
* a covered corpus below the FR-501 floor fails rather than shrinking;
* and an arm with no practice data attached is reported as unmeasured rather
  than as a delta of roughly zero, which is the failure that would look most
  like a finding.

Nothing here reads `data/raw/` or touches the network. The corpus is synthetic,
the practice frames are written into `tmp_path`, and the ranker is fitted with
a dozen trees — the structure is what is under test, not the accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.corpus import WindowTooNarrowError
from f1pred.data.collect import write_checkpoint
from f1pred.data.telemetry import (
    TELEMETRY_START_SEASON,
    frame_from_rows,
    write_practice_checkpoint,
)
from f1pred.evaluate.bootstrap import BootstrapError, bootstrap_delta, bootstrap_metrics
from f1pred.evaluate.metrics import METRICS, PRIMARY_METRIC, RACE_KEYS
from f1pred.evaluate.split import SplitError
from f1pred.evaluate.telemetry_arm import (
    PRIMARY_ARM,
    TELEMETRY_ARM,
    Arm,
    TelemetryArmError,
    cited_coverage,
    coverage_disagreements,
    covered_seasons,
    covered_split,
    covered_window,
    model_name,
    render_finding,
    run,
    telemetry_source_columns,
)
from f1pred.features.registry import REGISTRY, Availability, FeatureGroup
from f1pred.models.ranker import Hyperparameters
from tests.fixtures.races import synthetic_corpus

#: Two seasons before the source starts, so the record's "8 of 10 seasons"
#: contrast is a real one rather than an artefact of a fixture that happens to
#: begin in 2018.
SEASONS = tuple(range(2016, 2026))
ROUNDS = 4
DRIVERS = 10
REPLICATES = 64

#: A dozen trees and no early stopping. The fit is not what is being tested,
#: and the default 400 would make this file slow enough that somebody would
#: stop running it.
FAST = Hyperparameters(
    n_estimators=12, max_depth=3, early_stopping_rounds=None, calibration_seasons=2
)


def make_practice(season: int) -> pd.DataFrame:
    """One practice row per driver per round, in the declared practice schema."""
    return frame_from_rows(
        [
            {
                "season": season,
                "round": rnd,
                "driver_code": f"D{i:02d}",
                "practice_sessions": 1,
                "practice_laps": 20,
                "long_run_pace_s": 90.0 + i * 0.1,
                "long_run_laps": 12,
                "best_lap_s": 85.0 + i * 0.1,
                "stint_pace_trend_s_per_lap": 0.02,
                "longest_stint_laps": 12,
                "stints": 3,
            }
            for rnd in range(1, ROUNDS + 1)
            for i in range(DRIVERS)
        ]
    )


def write_corpus(directory: Path, seasons: tuple[int, ...]) -> Path:
    frame = synthetic_corpus(seasons=seasons, rounds=ROUNDS, transfer=(2019, 2))
    for season, rows in frame.groupby("season", sort=True):
        write_checkpoint(rows.reset_index(drop=True), directory, int(season))
    return directory


@pytest.fixture(scope="module")
def bare_root(tmp_path_factory) -> Path:
    """The corpus with no practice checkpoints: the source is not attached."""
    return write_corpus(tmp_path_factory.mktemp("raw_bare"), SEASONS)


@pytest.fixture(scope="module")
def covered_root(tmp_path_factory) -> Path:
    """The corpus plus practice data for every season the source covers."""
    directory = write_corpus(tmp_path_factory.mktemp("raw_covered"), SEASONS)
    for season in SEASONS:
        if season >= TELEMETRY_START_SEASON:
            write_practice_checkpoint(make_practice(season), directory, season)
    return directory


@pytest.fixture(scope="module")
def measured(covered_root, tmp_path_factory) -> tuple[dict[str, Any], tuple[Arm, Arm]]:
    out = tmp_path_factory.mktemp("results_covered")
    return run(
        root=covered_root, params=FAST, replicates=REPLICATES, results_dir=out
    )


@pytest.fixture(scope="module")
def unmeasured(bare_root, tmp_path_factory) -> tuple[dict[str, Any], tuple[Arm, Arm]]:
    out = tmp_path_factory.mktemp("results_bare")
    return run(root=bare_root, params=FAST, replicates=REPLICATES, results_dir=out)


def race_keys(scores: pd.DataFrame) -> set[tuple[int, int]]:
    return set(map(tuple, scores[list(RACE_KEYS)].to_numpy().tolist()))


class TestCoveredSeasons:
    def test_only_seasons_the_source_covers_are_admitted(self, covered_root):
        assert covered_seasons(covered_root) == tuple(range(2018, 2026))

    def test_the_start_season_is_the_sources_own_declaration(self, covered_root):
        """Not a literal 2018 in this module: the source owns that constant.

        If the collector's first season ever moves, this arm must move with it
        rather than keep measuring seasons the source no longer covers.
        """
        assert min(covered_seasons(covered_root)) == TELEMETRY_START_SEASON
        assert 2017 in SEASONS
        assert 2017 not in covered_seasons(covered_root)

    def test_a_corpus_with_no_covered_season_raises(self, tmp_path):
        assert covered_seasons(tmp_path) == ()
        with pytest.raises(TelemetryArmError, match="no arm to measure"):
            covered_window(covered_seasons(tmp_path))

    def test_the_two_arms_are_named_after_their_feature_sets(self):
        assert TELEMETRY_ARM == f"primary+{FeatureGroup.TELEMETRY.value}"
        assert model_name(PRIMARY_ARM) != model_name(TELEMETRY_ARM)


class TestConstitutionalFloor:
    """FR-501's minima, which the covered corpus meets with zero margin."""

    def test_eight_covered_seasons_leave_exactly_five_training(self):
        split = covered_split(tuple(range(2018, 2026)))
        assert len(split.seasons) == 8
        assert split.training_seasons == tuple(range(2018, 2023))
        assert split.holdout == (2023, 2024, 2025)
        assert split.describe()["meets_constitutional_minimum"] is True

    def test_seven_covered_seasons_are_refused(self):
        with pytest.raises(WindowTooNarrowError):
            covered_window(tuple(range(2018, 2025)))

    def test_a_thinner_holdout_is_refused_rather_than_accommodated(self):
        with pytest.raises(SplitError):
            covered_split(tuple(range(2018, 2026)), holdout_seasons=2)

    def test_a_thin_covered_corpus_fails_instead_of_reporting_a_thin_arm(
        self, covered_root
    ):
        """The seasons exist on disk; the arm still refuses to rest on seven."""
        with pytest.raises(WindowTooNarrowError):
            run(root=covered_root, seasons=range(2018, 2025), params=FAST)


class TestOneHoldoutForBothArms:
    def test_both_arms_scored_the_identical_races(self, measured):
        _record, (primary, telemetry) = measured
        assert race_keys(primary.per_race) == race_keys(telemetry.per_race)
        assert len(primary.per_race) == len(telemetry.per_race)

    def test_the_grid_baseline_scores_the_same_in_both_arms(self, measured):
        """A property of the races alone, so a difference would mean two holdouts."""
        record, _arms = measured
        scores = [
            record["arms"][arm]["evaluation"]["predictors"]["grid_order"]["metrics"][
                PRIMARY_METRIC.name
            ]["point"]
            for arm in (PRIMARY_ARM, TELEMETRY_ARM)
        ]
        assert scores[0] == pytest.approx(scores[1])

    def test_both_arms_report_the_same_holdout_races_and_rows(self, measured):
        record, _arms = measured
        first, second = (record["arms"][a]["evaluation"] for a in record["arms"])
        assert first["holdout_races"] == second["holdout_races"]
        assert first["holdout_rows"] == second["holdout_rows"]
        assert first["split"] == second["split"]

    def test_the_models_carry_distinct_names(self, measured):
        """A shared name would silently overwrite one arm's line in the harness."""
        _record, (primary, telemetry) = measured
        assert primary.model.name != telemetry.model.name
        assert primary.model.name in primary.evaluation["predictors"]
        assert telemetry.model.name in telemetry.evaluation["predictors"]

    def test_the_second_arm_is_the_first_plus_the_telemetry_group(self, measured):
        _record, (primary, telemetry) = measured
        added = set(telemetry.experiment.feature_names) - set(
            primary.experiment.feature_names
        )
        declared = {
            f.name
            for f in REGISTRY.group_features(FeatureGroup.TELEMETRY)
            if f.availability is Availability.PRE_RACE
        }
        assert added == declared
        assert set(primary.experiment.feature_names) < set(
            telemetry.experiment.feature_names
        )

    def test_the_added_features_are_named_in_the_record(self, measured):
        record, _arms = measured
        named = {f["name"] for f in record["telemetry_features"]}
        assert named == {
            f.name
            for f in REGISTRY.group_features(FeatureGroup.TELEMETRY)
            if f.availability is Availability.PRE_RACE
        }


class TestPairedDelta:
    def test_the_recorded_delta_is_the_paired_bootstrap(self, measured):
        record, (primary, telemetry) = measured
        recomputed = bootstrap_delta(
            telemetry.per_race,
            primary.per_race,
            metrics=METRICS,
            replicates=REPLICATES,
        )
        for metric in METRICS:
            assert (
                record["telemetry_minus_primary"][metric.name]
                == recomputed[metric.name].as_dict()
            )

    def test_the_delta_runs_telemetry_minus_primary(self, measured):
        record, _arms = measured
        arms = record["arms"]
        expected = (
            arms[TELEMETRY_ARM]["metrics"][PRIMARY_METRIC.name]["point"]
            - arms[PRIMARY_ARM]["metrics"][PRIMARY_METRIC.name]["point"]
        )
        assert record["telemetry_minus_primary"][PRIMARY_METRIC.name][
            "point"
        ] == pytest.approx(expected)

    def test_a_delta_over_different_races_is_refused(self, measured):
        """The structural guarantee that the pairing is real, not assumed."""
        _record, (primary, telemetry) = measured
        with pytest.raises(BootstrapError, match="different races"):
            bootstrap_delta(
                telemetry.per_race.iloc[1:],
                primary.per_race,
                metrics=METRICS,
                replicates=8,
            )

    def test_pairing_is_narrower_than_subtracting_two_intervals(self, measured):
        """Why the pairing is not a detail: the unpaired interval is wider.

        Both arms are scored on the same races, so a race that is hard for one
        is hard for the other. Bootstrapping them separately and subtracting
        throws that away and inflates the uncertainty — in the direction that
        turns a real difference into a null.
        """
        record, (primary, telemetry) = measured
        levels = [
            bootstrap_metrics(arm.per_race, metrics=METRICS, replicates=REPLICATES)[
                PRIMARY_METRIC.name
            ]
            for arm in (telemetry, primary)
        ]
        unpaired = (levels[0].high - levels[1].low) - (levels[0].low - levels[1].high)
        paired = record["telemetry_minus_primary"][PRIMARY_METRIC.name]
        assert paired["ci_high"] - paired["ci_low"] < unpaired


class TestReducedSeasonCount:
    """FR-215's mandatory half: the count travels with the number."""

    def test_the_record_states_how_many_seasons_the_arm_rests_on(self, measured):
        record, _arms = measured
        seasons = record["seasons"]
        assert seasons["telemetry_covered"] == list(range(2018, 2026))
        assert seasons["n_telemetry_covered"] == 8

    def test_the_record_states_the_whole_corpus_for_contrast(self, measured):
        record, _arms = measured
        seasons = record["seasons"]
        assert seasons["n_collected_corpus_seasons"] == len(SEASONS)
        assert seasons["seasons_this_arm_cannot_speak_about"] == [2016, 2017]
        assert "8 of the corpus's 10" in seasons["reduced_season_count"]

    def test_the_record_states_the_training_and_holdout_seasons(self, measured):
        record, _arms = measured
        assert record["seasons"]["training_seasons"] == list(range(2018, 2023))
        assert record["seasons"]["holdout_seasons"] == [2023, 2024, 2025]
        assert record["split"]["training_seasons"] == record["seasons"][
            "training_seasons"
        ]
        assert record["split"]["meets_constitutional_minimum"] is True

    def test_the_finding_states_the_count_beside_the_benefit(self, measured):
        record, _arms = measured
        assert "8 of the corpus's 10 seasons" in record["finding"]
        assert "2018-2025" in record["finding"]

    def test_the_arm_disclaims_comparison_with_the_headline_run(self, measured):
        record, _arms = measured
        assert "not the project's model" in record["seasons"]["not_the_headline_run"]

    def test_the_written_file_carries_the_count(self, measured, covered_root, tmp_path):
        record, _arms = measured
        target = tmp_path / "telemetry_arm.json"
        target.write_text(json.dumps(record, indent=2, default=str))
        reloaded = json.loads(target.read_text())
        assert reloaded["seasons"]["n_telemetry_covered"] == 8
        assert reloaded["seasons"]["n_collected_corpus_seasons"] == len(SEASONS)

    def test_the_file_is_named_after_the_holdout_it_reports(
        self, covered_root, tmp_path_factory
    ):
        out = tmp_path_factory.mktemp("results_named")
        run(root=covered_root, params=FAST, replicates=8, results_dir=out)
        assert (out / "telemetry_arm_2023_2025.json").exists()


class TestBothArmsRecordTheirSources:
    def test_each_arm_records_its_absent_sources(self, measured):
        record, _arms = measured
        for arm in (PRIMARY_ARM, TELEMETRY_ARM):
            assert "absent_sources" in record["arms"][arm]
        # The practice columns are attached in this run, so they are not absent.
        assert not record["absent_telemetry_sources"]
        assert record["telemetry_attached"] is True

    def test_the_source_columns_are_derived_from_the_group(self):
        columns = telemetry_source_columns()
        assert "long_run_pace_s" in columns
        assert "stint_pace_trend_s_per_lap" in columns
        assert "weather_prior_wet_probability" not in columns


class TestTelemetryNotAttached:
    """The failure that looks most like a finding: a delta of nothing at all."""

    def test_the_record_says_the_arm_measured_nothing(self, unmeasured):
        record, _arms = unmeasured
        assert record["telemetry_attached"] is False

    def test_the_absent_source_columns_are_named(self, unmeasured):
        record, _arms = unmeasured
        assert "long_run_pace_s" in record["absent_telemetry_sources"]
        assert "long_run_pace_s" in record["arms"][TELEMETRY_ARM]["absent_sources"]

    def test_no_telemetry_feature_holds_a_value(self, unmeasured):
        record, _arms = unmeasured
        assert record["telemetry_features"]
        assert all(
            f["holdout_rows_with_a_value"] == 0 for f in record["telemetry_features"]
        )

    def test_the_finding_refuses_to_report_a_benefit(self, unmeasured):
        record, _arms = unmeasured
        assert record["finding"].startswith("NOT MEASURED")
        assert "NO MEASURED BENEFIT" not in record["finding"]
        assert "TELEMETRY HELPS" not in record["finding"]

    def test_the_features_still_entered_the_second_arm(self, unmeasured):
        """Null columns, not missing ones: FR-214's null is never imputed away."""
        _record, (primary, telemetry) = unmeasured
        assert len(telemetry.experiment.feature_names) > len(
            primary.experiment.feature_names
        )


class TestCoverageReporting:
    def test_coverage_is_reported_for_every_collected_season(self, measured):
        record, _arms = measured
        per_season = record["coverage"]["per_season"]
        assert sorted(per_season) == [str(s) for s in SEASONS]

    def test_uncovered_seasons_are_reported_as_uncovered(self, measured):
        record, _arms = measured
        for season in ("2016", "2017"):
            entry = record["coverage"]["per_season"][season]
            assert entry["coverage"] == 0.0
            assert entry["source_covers_season"] is False

    def test_covered_seasons_report_their_share_of_corpus_rows(self, measured):
        record, _arms = measured
        for season in ("2018", "2025"):
            entry = record["coverage"]["per_season"][season]
            assert entry["coverage"] == pytest.approx(1.0)
            assert entry["covered"] == entry["corpus_rows"]

    def test_an_absent_citation_is_reported_rather_than_assumed(self, measured):
        record, _arms = measured
        assert record["coverage"]["cited_file_found"] is False
        assert record["coverage"]["agrees_with_cited_file"] is None
        assert record["coverage"]["cited_file"].endswith(
            "telemetry_coverage_2018_2025.json"
        )

    def test_a_citation_is_read_from_the_name_t214_writes(self, tmp_path):
        seasons = tuple(range(2018, 2026))
        path = tmp_path / "telemetry_coverage_2018_2025.json"
        path.write_text(json.dumps({"coverage_per_season": {"2018": {"coverage": 1.0}}}))
        found, table = cited_coverage(seasons, results_dir=tmp_path)
        assert found == path
        assert table == {"2018": {"coverage": 1.0}}

    def test_agreement_with_the_citation_is_checked_not_asserted(self):
        mine = {"2018": {"coverage": 1.0}, "2019": {"coverage": 0.9}}
        assert coverage_disagreements(mine, {"2018": {"coverage": 1.0}}) == {}
        assert coverage_disagreements(mine, None) == {}

    def test_a_disagreeing_citation_is_named(self):
        mine = {"2018": {"coverage": 1.0}, "2019": {"coverage": 0.9}}
        found = coverage_disagreements(mine, {"2019": {"coverage": 0.5}})
        assert list(found) == ["2019"]
        assert found["2019"] == {"this_run": 0.9, "cited": 0.5}


class TestTheFinding:
    def test_a_null_result_is_stated_as_a_result(self, measured):
        record, _arms = measured
        delta = record["telemetry_minus_primary"][PRIMARY_METRIC.name]
        finding = render_finding(record)
        if delta["excludes_zero"]:
            assert "TELEMETRY HELPS" in finding or "TELEMETRY HURTS" in finding
        else:
            assert "NO MEASURED BENEFIT" in finding
            assert "measured null" in finding

    def test_the_finding_quotes_both_arms(self, measured):
        record, _arms = measured
        finding = render_finding(record)
        assert PRIMARY_ARM in finding and TELEMETRY_ARM in finding
        assert "paired" in finding
