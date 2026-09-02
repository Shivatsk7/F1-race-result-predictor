"""T-407 — the oracle-variant comparison (FR-207, FR-304, FR-504).

What is checked here is the *shape* of the comparison, never the size of the
gap. The magnitude belongs to the corpus and is reported in
`results/oracle_vs_primary_2023_2025.json`; asserting it against a synthetic
fixture would pin a number that means nothing and would break every time the
fixture's seed moved.

So the tests are about the properties that make the number honest: that both
arms are scored on one holdout by one call, that the delta is paired rather
than assembled by subtracting two independent intervals, that the leak is named
feature by feature, and that the oracle arm is labelled inadmissible wherever a
reader might mistake it for a result.

Nothing here reads `data/raw/` or touches the network. The corpus is
`synthetic_corpus` written into `tmp_path`, the way `tests/test_dataset.py`
does it, and the ranker is given few trees because none of these properties are
properties of accuracy.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401  (populates REGISTRY)
from f1pred.corpus import CorpusWindow
from f1pred.data.collect import write_checkpoint
from f1pred.dataset import prepare
from f1pred.evaluate.metrics import PRIMARY_METRIC
from f1pred.evaluate.oracle_arm import (
    INADMISSIBLE,
    ORACLE_ARM,
    PRIMARY_ARM,
    OracleArmError,
    OracleComparison,
    assert_arms_comparable,
    compare,
    describe_leak,
    render_comparison,
)
from f1pred.evaluate.split import ChronologicalSplit
from f1pred.features.build import build_oracle, build_primary
from f1pred.features.registry import REGISTRY, Availability
from f1pred.models.ranker import Hyperparameters
from tests.fixtures.races import synthetic_corpus

SEASONS = tuple(range(2012, 2022))
ROUNDS = 3
METRIC = PRIMARY_METRIC.name

#: Ten seasons of fixture, three held out. Enough to clear the FR-501 floors
#: and to leave the ranker a calibration slice inside training.
HOLDOUT = SEASONS[-3:]


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """A bare synthetic corpus on disk. No weather, no practice: this module
    has no opinion about enrichment, and both arms are missing it equally."""
    directory = tmp_path_factory.mktemp("raw_oracle")
    corpus = synthetic_corpus(seasons=SEASONS, rounds=ROUNDS, transfer=(2014, 2))
    for season, frame in corpus.groupby("season", sort=True):
        write_checkpoint(frame.reset_index(drop=True), directory, int(season))
    return directory


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    return ChronologicalSplit(
        window=CorpusWindow(width=len(SEASONS), earliest_available=SEASONS[0]),
        holdout_seasons=3,
        season_end_dates={s: date(s, 12, 1) for s in SEASONS},
    )


def quick() -> Hyperparameters:
    """Few trees, no early stopping. Every property here is structural."""
    return Hyperparameters(n_estimators=30, max_depth=3, early_stopping_rounds=None)


@pytest.fixture(scope="module")
def comparison(root, split, tmp_path_factory) -> OracleComparison:
    """One real run of the whole path, reused by every test that reads it."""
    return compare(
        root=root,
        split=split,
        params=quick(),
        replicates=200,
        path=tmp_path_factory.mktemp("results") / "oracle_vs_primary.json",
    )


@pytest.fixture(scope="module")
def record(comparison) -> dict:
    return comparison.record


class TestTheTwoArmsAreOneExperiment:
    """FR-304: identical holdout, identical code path, or the delta is fiction."""

    def test_both_arms_appear_in_the_record_under_their_own_names(self, record):
        assert PRIMARY_ARM in record["predictors"]
        assert ORACLE_ARM in record["predictors"]
        assert PRIMARY_ARM != ORACLE_ARM

    def test_both_arms_scored_the_same_races(self, record):
        primary = record["predictors"][PRIMARY_ARM]["races_scored"]
        oracle = record["predictors"][ORACLE_ARM]["races_scored"]
        assert primary == oracle == record["holdout_races"]

    def test_the_holdout_is_the_split_the_run_declared(self, record, split):
        assert record["split"]["holdout_seasons"] == list(split.holdout)
        assert record["seasons"] == sorted(SEASONS)

    def test_the_grid_baseline_is_present_so_every_delta_has_a_reference(
        self, record
    ):
        # FR-504 measures against `grid_order`; the harness refuses without it.
        assert "grid_order" in record["predictors"]

    def test_two_arms_sharing_a_name_are_refused(self, root, split):
        """The harness keys by name, so a shared name silently drops an arm."""
        with pytest.raises(OracleArmError, match="overwrite"):
            compare(
                root=root,
                split=split,
                params=quick(),
                primary_name="same",
                oracle_name="same",
                write=False,
            )


class TestArmsComparable:
    """The refusals that stand between "two feature sets" and "two corpora"."""

    @pytest.fixture(scope="class")
    def arms(self, root):
        return (
            prepare(root=root, builder=build_primary),
            prepare(root=root, builder=build_oracle),
        )

    def test_the_extra_features_are_returned_and_non_empty(self, arms):
        extra = assert_arms_comparable(*arms)
        assert extra
        assert set(extra).isdisjoint(arms[0].feature_names)

    def test_a_corpus_of_a_different_size_is_refused(self, root, arms):
        short = prepare(list(SEASONS[:5]), root=root, builder=build_oracle)
        with pytest.raises(OracleArmError, match="different corpora"):
            assert_arms_comparable(arms[0], short)

    def test_an_oracle_set_that_adds_nothing_is_refused(self, arms):
        """A build defect must not read as "leakage buys nothing"."""
        with pytest.raises(OracleArmError, match="adds nothing"):
            assert_arms_comparable(arms[0], arms[0])

    def test_a_perturbed_shared_feature_is_refused(self, arms):
        primary, oracle = arms
        tampered = oracle.frame.copy()
        name = primary.feature_names[0]
        tampered[name] = tampered[name] + 1.0
        with pytest.raises(OracleArmError, match="different values"):
            assert_arms_comparable(
                primary, type(oracle)(
                    frame=tampered,
                    matrix=oracle.matrix,
                    feature_names=oracle.feature_names,
                    absent_sources=oracle.absent_sources,
                )
            )

    def test_a_dropped_primary_feature_is_refused(self, arms):
        primary, oracle = arms
        kept = tuple(n for n in oracle.feature_names if n != primary.feature_names[0])
        with pytest.raises(OracleArmError, match="missing primary feature"):
            assert_arms_comparable(
                primary, type(oracle)(
                    frame=oracle.frame,
                    matrix=oracle.matrix,
                    feature_names=kept,
                    absent_sources=oracle.absent_sources,
                )
            )


class TestTheLeakIsNamed:
    """FR-207: a reader must be able to see *what* the oracle arm knows."""

    def test_the_record_names_every_added_feature(self, record):
        leak = record["leak"]["features_in_oracle_not_in_primary"]
        assert leak
        assert record["leak"]["n_features_added"] == len(leak)
        assert all(entry["name"] and entry["description"] for entry in leak)

    def test_every_added_feature_is_declared_post_race(self, record):
        for entry in record["leak"]["features_in_oracle_not_in_primary"]:
            assert entry["availability"] == Availability.POST_RACE.value
            assert REGISTRY.get(entry["name"]).availability is Availability.POST_RACE

    def test_the_primary_arm_carries_none_of_them(self, record):
        added = {
            e["name"] for e in record["leak"]["features_in_oracle_not_in_primary"]
        }
        assert added.isdisjoint(record["arms"]["primary"]["features"])
        assert added <= set(record["arms"]["oracle"]["features"])

    def test_the_description_is_read_off_the_registry(self):
        # Written here would go stale silently; read back, it cannot.
        name = "finished_race"
        assert describe_leak([name])[0]["description"] == (
            REGISTRY.get(name).description
        )


class TestThePairedDelta:
    """D-5, FR-504: the delta is measured, not assembled from two intervals."""

    def test_the_delta_is_reported_for_every_metric(self, record):
        assert set(record["oracle_minus_primary"]) == {
            m for m in record["predictors"][PRIMARY_ARM]["metrics"]
        }

    def test_the_point_estimate_is_oracle_minus_primary(self, record):
        arms = record["arms"]
        expected = (
            arms["oracle"]["metrics"][METRIC]["point"]
            - arms["primary"]["metrics"][METRIC]["point"]
        )
        assert record["oracle_minus_primary"][METRIC]["point"] == pytest.approx(
            expected, abs=1e-9
        )

    def test_the_interval_is_narrower_than_subtracting_two_independent_ones(
        self, record
    ):
        """Pairing is the whole point of computing this here rather than by hand.

        Subtracting the two arms' own intervals gives a width of `w_a + w_b`.
        The paired width is at most `sqrt(v_a + v_b - 2c)` in the same units,
        which is strictly smaller than `sqrt(v_a) + sqrt(v_b)` for any positive
        variances — before the positive correlation between two predictors
        scored on the same races narrows it further.
        """
        arms = record["arms"]
        naive = sum(
            arms[a]["metrics"][METRIC]["ci_high"] - arms[a]["metrics"][METRIC]["ci_low"]
            for a in ("primary", "oracle")
        )
        paired = record["oracle_minus_primary"][METRIC]
        assert paired["ci_high"] - paired["ci_low"] < naive

    def test_the_headline_block_says_it_is_paired(self, record):
        assert record["inflation"]["paired"] is True
        assert record["inflation"]["metric"] == METRIC


class TestTheRecordDescribesItself:
    """FR-506: the file has to stand on its own, without this module beside it."""

    def test_each_arm_reports_its_own_features_rows_and_absent_sources(self, record):
        for arm in ("primary", "oracle"):
            entry = record["arms"][arm]
            assert entry["n_features"] == len(entry["features"])
            assert entry["corpus_rows"] > entry["training_rows"] > 0
            assert isinstance(entry["absent_sources"], list)
            # Bare fixture corpus: no weather archive, no practice checkpoints.
            assert "weather_prior_wet_probability" in entry["absent_sources"]

    def test_the_oracle_set_is_a_strict_superset_of_the_primary_set(self, record):
        primary = set(record["arms"]["primary"]["features"])
        oracle = set(record["arms"]["oracle"]["features"])
        assert primary < oracle

    def test_both_arms_carry_an_interval_on_every_metric(self, record):
        for arm in ("primary", "oracle"):
            for entry in record["arms"][arm]["metrics"].values():
                assert entry["ci_low"] <= entry["point"] <= entry["ci_high"]
                assert entry["confidence"] == pytest.approx(0.95)

    def test_the_statement_is_plain_language_in_the_metrics_units(self, record):
        statement = record["inflation"]["statement"]
        assert METRIC in statement
        assert f"{record['inflation']['delta']:+.4f}" in statement
        assert "holdout" in statement

    def test_the_oracle_arm_is_labelled_inadmissible(self, record):
        assert record["admissibility"]["oracle_arm"] == INADMISSIBLE
        assert "INADMISSIBLE" in record["admissibility"]["oracle_arm"]
        assert "FR-207" in record["admissibility"]["oracle_arm"]
        assert "Admissible" in record["admissibility"]["primary_arm"]

    def test_the_verdict_block_is_annotated_as_counting_the_oracle_arm(self, record):
        """FR-507's verdict counts every predictor it was handed, oracle included.

        A reader skimming this file for a headline would otherwise find
        "beats the grid" attached to an arm that read the finishing order.
        """
        note = record["admissibility"]["verdict_block"]
        assert "oracle arm" in note and "reading the result" in note
        assert ORACLE_ARM in {
            entry["name"] for entry in record["verdict"]["beaten_by"]
        } | {entry["name"] for entry in record["verdict"]["not_beaten_by"]}

    def test_the_written_file_is_the_record(self, comparison):
        assert comparison.path is not None and comparison.path.exists()
        written = json.loads(comparison.path.read_text())
        assert written["inflation"]["statement"] == (
            comparison.record["inflation"]["statement"]
        )
        assert written["leak"] == comparison.record["leak"]

    def test_the_rendered_summary_names_the_leak_and_the_caveat(self, record):
        text = render_comparison(record)
        assert "INADMISSIBLE" in text
        assert record["inflation"]["statement"] in text
        for entry in record["leak"]["features_in_oracle_not_in_primary"]:
            assert entry["name"] in text


class TestTheModelsBehindIt:
    def test_each_model_was_fitted_on_its_own_arms_features(self, comparison):
        assert comparison.primary.feature_names == tuple(
            comparison.record["arms"]["primary"]["features"]
        )
        assert comparison.oracle.feature_names == tuple(
            comparison.record["arms"]["oracle"]["features"]
        )

    def test_neither_model_saw_a_holdout_season(self, comparison, split):
        for model in (comparison.primary, comparison.oracle):
            assert set(model.training_seasons).isdisjoint(split.holdout)
            assert max(model.training_seasons) < split.boundary_season

    def test_both_models_trained_on_the_same_rows(self, comparison):
        # Same corpus, same split: a difference here would mean the arms were
        # built from frames of different length and the gap is not the leak.
        assert comparison.primary.training_rows == comparison.oracle.training_rows

    def test_nothing_was_written_when_write_is_off(self, root, split):
        out = compare(
            root=root, split=split, params=quick(), replicates=64, write=False
        )
        assert out.path is None


def test_the_run_does_not_read_the_real_corpus(root):
    """Guards the guard: every fixture above must resolve inside `tmp_path`."""
    assert not str(root).startswith(str(Path.cwd() / "data"))
    assert pd.read_parquet(next(iter(root.glob("season_*.parquet")))).size > 0
