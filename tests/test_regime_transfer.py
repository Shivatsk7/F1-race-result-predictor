"""Tests for the regulation-transfer experiment (FR-511, FR-511a, FR-512, AC-12).

The tests that matter here are the ones about *labelling and refusal*, not
about the size of the penalty. The penalty is measured against the real corpus
and committed to `results/`; what has to be guaranteed in a test is that a
transition which falls below the FR-501 training floor is labelled
`DIAGNOSTIC`, reports the count that made it short, and cannot reach the
headline — and that the split type this experiment needs still refuses to train
on the season it is about to be scored on.

Nothing here reads `data/raw/` or the network. The corpus is
`tests/fixtures/races.py::synthetic_corpus` and the fits are deliberately tiny:
the arithmetic of the experiment is what is under test, and a fixture that took
a minute to fit would be run less often than one that takes a second.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.config import REGIMES, RegimeSeverity
from f1pred.corpus import CorpusWindow
from f1pred.dataset import experiment
from f1pred.evaluate.bootstrap import BootstrapError, bootstrap_delta
from f1pred.evaluate.metrics import PRIMARY_METRIC
from f1pred.evaluate.regime_transfer import (
    BASELINE_NAME,
    COMPARATOR_ARM,
    DIAGNOSTIC_LABEL,
    MODEL_NAME,
    TRANSFER_ARM,
    TransferError,
    TransferSplit,
    reset_seasons,
    run,
    transitions,
    unpaired_delta,
)
from f1pred.evaluate.split import (
    MIN_TRAINING_SEASONS,
    ChronologicalSplit,
    SplitError,
)
from f1pred.models.ranker import Hyperparameters
from tests.fixtures.races import synthetic_corpus

#: The real archive, as `corpus.archive_seasons` resolves it for 2025.
ARCHIVE = tuple(range(2010, 2026))

#: The fixture archive: short enough to fit in seconds, long enough to hold
#: one diagnostic transition (2014) and one compliant one (2017).
FIXTURE_SEASONS = tuple(range(2010, 2019))


def per_race(seasons: tuple[int, ...], values: tuple[float, ...]) -> pd.DataFrame:
    """A minimal per-race score frame, in the shape the bootstrap consumes."""
    return pd.DataFrame(
        {
            "season": [s for s in seasons for _ in values],
            "round": [i for _ in seasons for i, _ in enumerate(values, start=1)],
            PRIMARY_METRIC.name: [v for _ in seasons for v in values],
        }
    )


class TestTransferSplit:
    def test_training_must_precede_the_evaluated_season(self):
        with pytest.raises(TransferError, match="at or after the evaluated season"):
            TransferSplit(training_seasons=(2010, 2011, 2014), evaluated_season=2014)

    def test_a_split_with_no_training_seasons_is_refused(self):
        with pytest.raises(TransferError, match="nothing before it"):
            TransferSplit(training_seasons=(), evaluated_season=2010)

    def test_seasons_out_of_order_are_refused(self):
        with pytest.raises(TransferError, match="chronological order"):
            TransferSplit(training_seasons=(2012, 2010), evaluated_season=2014)

    def test_from_archive_takes_everything_before_and_nothing_after(self):
        split = TransferSplit.from_archive(ARCHIVE, 2014)
        assert split.training_seasons == (2010, 2011, 2012, 2013)
        assert split.holdout == (2014,)
        assert split.boundary_season == 2014

    def test_a_short_transition_is_labelled_diagnostic_with_its_count(self):
        """FR-511a: 2014 trains on four seasons, one short of the floor."""
        described = TransferSplit.from_archive(ARCHIVE, 2014).describe()
        assert described["n_training_seasons"] == 4
        assert described["training_seasons"] == [2010, 2011, 2012, 2013]
        assert described["evaluated_season"] == 2014
        assert described["meets_constitutional_minimum"] is False
        assert described["label"] == DIAGNOSTIC_LABEL

    def test_a_compliant_transition_carries_no_label(self):
        described = TransferSplit.from_archive(ARCHIVE, 2017).describe()
        assert described["n_training_seasons"] == 7
        assert described["meets_constitutional_minimum"] is True
        assert described["label"] is None

    def test_the_floor_is_the_one_split_py_declares(self):
        """Imported, not retyped: a change to the floor must reach this file."""
        described = TransferSplit.from_archive(ARCHIVE, 2017).describe()
        assert described["minimum_required"]["training"] == MIN_TRAINING_SEASONS

        exactly_at_the_floor = TransferSplit(
            training_seasons=tuple(range(2000, 2000 + MIN_TRAINING_SEASONS)),
            evaluated_season=2010,
        )
        one_short = TransferSplit(
            training_seasons=tuple(range(2000, 2000 + MIN_TRAINING_SEASONS - 1)),
            evaluated_season=2010,
        )
        assert exactly_at_the_floor.meets_constitutional_minimum
        assert not one_short.meets_constitutional_minimum
        assert one_short.label == DIAGNOSTIC_LABEL

    def test_leakage_from_the_evaluated_season_is_refused(self):
        split = TransferSplit.from_archive(ARCHIVE, 2017)
        training = pd.DataFrame({"season": [2015, 2016, 2017]})
        with pytest.raises(TransferError, match="at or after the evaluated season"):
            split.assert_no_leakage(training)

    def test_leakage_from_a_later_season_is_refused(self):
        split = TransferSplit.from_archive(ARCHIVE, 2017)
        with pytest.raises(TransferError, match=r"\[2019\]"):
            split.assert_no_leakage(pd.DataFrame({"season": [2016, 2019]}))

    def test_clean_training_passes_the_leakage_check(self):
        split = TransferSplit.from_archive(ARCHIVE, 2017)
        split.assert_no_leakage(pd.DataFrame({"season": [2010, 2013, 2016]}))

    def test_split_drops_every_season_after_the_evaluated_one(self):
        """The frame handed in is the whole archive, including 2015-2025."""
        frame = pd.DataFrame({"season": list(ARCHIVE)})
        training, holdout = TransferSplit.from_archive(ARCHIVE, 2014).split(frame)
        assert sorted(training["season"]) == [2010, 2011, 2012, 2013]
        assert sorted(holdout["season"]) == [2014]

    def test_the_masks_agree_with_the_declared_seasons(self):
        frame = pd.DataFrame({"season": list(ARCHIVE)})
        split = TransferSplit.from_archive(ARCHIVE, 2022)
        assert int(split.train_mask(frame).sum()) == 12
        assert int(split.holdout_mask(frame).sum()) == 1

    def test_a_frame_without_a_season_column_is_refused(self):
        with pytest.raises(TransferError, match="no `season` column"):
            TransferSplit.from_archive(ARCHIVE, 2017).split(pd.DataFrame({"x": [1]}))

    def test_the_chronological_split_cannot_express_this_experiment(self):
        """Which is why `TransferSplit` exists, and why it lives here.

        A single-season holdout is refused by `ChronologicalSplit` on purpose
        (FR-501, Constitution IV.2). This test pins that refusal: if it ever
        starts passing, the headline path has been weakened to accommodate a
        diagnostic, which FR-511a exists to prevent.
        """
        with pytest.raises(SplitError, match="below the Constitution IV.2 floor"):
            ChronologicalSplit(window=CorpusWindow(width=16), holdout_seasons=1)
        assert not isinstance(
            TransferSplit.from_archive(ARCHIVE, 2017), ChronologicalSplit
        )


class TestTransitions:
    def test_the_archive_yields_the_three_declared_resets(self):
        assert reset_seasons(ARCHIVE) == (2014, 2017, 2022)

    def test_moderate_regimes_are_not_transitions(self):
        """2011 and 2019 are declared MODERATE; FR-511 is about resets."""
        moderate = {
            r.start_season
            for r in REGIMES
            if r.severity is RegimeSeverity.MODERATE
        }
        assert moderate.isdisjoint(reset_seasons(ARCHIVE))

    def test_the_reset_at_the_corpus_start_is_not_testable(self):
        """2010 is a major reset and the corpus start; nothing precedes it."""
        assert 2010 not in reset_seasons(ARCHIVE)

    def test_a_future_reset_enters_the_experiment_when_its_season_arrives(self):
        """D-10 put 2026 in the table so this needs no code change."""
        assert 2026 not in reset_seasons(ARCHIVE)
        assert 2026 in reset_seasons(tuple(range(2010, 2027)))

    def test_the_comparator_is_the_identical_protocol_one_season_earlier(self):
        ground_effect = next(t for t in transitions(ARCHIVE) if t.reset_season == 2022)
        assert ground_effect.transfer.evaluated_season == 2022
        assert ground_effect.transfer.training_seasons == tuple(range(2010, 2022))
        assert ground_effect.comparator.evaluated_season == 2021
        assert ground_effect.comparator.training_seasons == tuple(range(2010, 2021))

    def test_the_headline_counts_two_transitions_and_not_three(self):
        """FR-511a and AC-12: 2014 is reported, and does not count."""
        found = transitions(ARCHIVE)
        assert len(found) == 3
        compliant = [t.reset_season for t in found if t.supports_headline_claim]
        diagnostic = [t.reset_season for t in found if not t.supports_headline_claim]
        assert compliant == [2017, 2022]
        assert diagnostic == [2014]
        assert all(t.label == DIAGNOSTIC_LABEL for t in found if t.reset_season == 2014)

    def test_a_transition_is_short_if_either_arm_is(self):
        """The 2014 comparator trains on three seasons — shorter still."""
        v6 = next(t for t in transitions(ARCHIVE) if t.reset_season == 2014)
        assert v6.transfer.n_training_seasons == 4
        assert v6.comparator.n_training_seasons == 3
        assert not v6.meets_constitutional_minimum

    def test_an_empty_archive_has_no_transitions(self):
        with pytest.raises(TransferError, match="empty archive"):
            reset_seasons(())


class TestPenaltyPairing:
    """The two arms are scored on different races, and the record says so."""

    def test_the_arms_cannot_be_paired(self):
        """`bootstrap_delta` refuses mismatched race sets, and is right to.

        This is the test that justifies `unpaired_delta` existing at all: if
        the pairing were available, using an unpaired interval would be
        overstating uncertainty for no reason.
        """
        transfer = per_race((2022,), (0.6, 0.7, 0.8))
        comparator = per_race((2021,), (0.6, 0.7, 0.8))
        with pytest.raises(BootstrapError, match="different races"):
            bootstrap_delta(transfer, comparator, metrics=(PRIMARY_METRIC,))

    def test_unpaired_delta_is_the_difference_of_the_two_means(self):
        transfer = per_race((2022,), (0.4, 0.6))
        comparator = per_race((2021,), (0.7, 0.9))
        interval = unpaired_delta(
            transfer, comparator, metrics=(PRIMARY_METRIC,), replicates=200
        )[PRIMARY_METRIC.name]
        assert interval.point == pytest.approx(0.3)
        assert interval.low <= interval.point <= interval.high

    def test_an_arm_with_no_scored_races_has_no_penalty(self):
        with pytest.raises(TransferError, match="no scored races"):
            unpaired_delta(
                per_race((), ()), per_race((2021,), (0.5,)), metrics=(PRIMARY_METRIC,)
            )


@pytest.fixture(scope="module")
def record() -> dict:
    """One whole experiment over a synthetic archive, at two decay settings.

    Nine seasons, three rounds each: enough for the 2014 transition (four
    training seasons, DIAGNOSTIC) and the 2017 one (seven, compliant) to both
    be built, which is the pairing AC-12 requires the real run to report. The
    trees are few and early stopping is off so the eight fits cost seconds.
    """
    corpus = synthetic_corpus(seasons=FIXTURE_SEASONS, rounds=3, transfer=(2013, 2))
    return run(
        experiment=experiment(corpus),
        decays=(0.0, 0.5),
        params=Hyperparameters(
            n_estimators=20, max_depth=3, early_stopping_rounds=None
        ),
        replicates=200,
        write=False,
    )


class TestRun:
    def test_a_single_decay_setting_is_refused(self):
        """FR-512: measured at zero and non-zero, or not claimed."""
        with pytest.raises(TransferError, match="FR-512"):
            run(decays=(0.0,), write=False)
        with pytest.raises(TransferError, match="FR-512"):
            run(decays=(0.25, 0.5), write=False)

    def test_it_runs_every_transition_at_every_decay(self, record):
        assert record["decay_settings"] == [0.0, 0.5]
        assert [r["recency_decay"] for r in record["runs"]] == [0.0, 0.5]
        for run_record in record["runs"]:
            assert [t["reset_season"] for t in run_record["transitions"]] == [
                2014,
                2017,
            ]

    def test_every_arm_reports_its_actual_training_season_count(self, record):
        """FR-511a, per arm rather than per experiment."""
        for run_record in record["runs"]:
            for t in run_record["transitions"]:
                for arm in (TRANSFER_ARM, COMPARATOR_ARM):
                    described = t["arms"][arm]["split"]
                    assert described["n_training_seasons"] == len(
                        described["training_seasons"]
                    )
                    assert t["n_training_seasons"][arm] == described[
                        "n_training_seasons"
                    ]

    def test_the_short_transition_is_labelled_and_kept_out_of_the_headline(
        self, record
    ):
        head = record["headline"]
        assert head["diagnostic_transitions"] == [2014]
        assert head["transitions"] == [2017]
        assert head["n"] == 1  # one compliant transition in this short archive
        v6 = next(t for t in record["runs"][0]["transitions"] if t["reset_season"] == 2014)
        assert v6["label"] == DIAGNOSTIC_LABEL
        assert v6["supports_headline_claim"] is False
        assert v6["n_training_seasons"][TRANSFER_ARM] == 4
        assert DIAGNOSTIC_LABEL in head["statement"]

    def test_no_arm_trained_on_the_season_it_was_scored_on(self, record):
        for run_record in record["runs"]:
            for t in run_record["transitions"]:
                for arm in t["arms"].values():
                    described = arm["split"]
                    assert described["evaluated_season"] not in (
                        described["training_seasons"]
                    )
                    assert max(described["training_seasons"]) < described[
                        "evaluated_season"
                    ]

    def test_the_grid_baseline_is_scored_on_every_evaluated_season(self, record):
        """A penalty is unreadable without knowing how predictable the season was."""
        metric = record["primary_metric"]
        for run_record in record["runs"]:
            for t in run_record["transitions"]:
                for arm in t["arms"].values():
                    assert metric in arm["predictors"][BASELINE_NAME]["metrics"]
                    assert metric in arm["predictors"][MODEL_NAME]["metrics"]
                assert metric in t["penalty"]["all_metrics"][BASELINE_NAME]

    def test_the_penalty_records_that_the_pairing_is_unavailable(self, record):
        for run_record in record["runs"]:
            for t in run_record["transitions"]:
                penalty = t["penalty"]
                assert penalty["pairing"] == "unavailable"
                assert "different races" in penalty["pairing_reason"]
                assert penalty["metric"] == record["primary_metric"]
                assert penalty[MODEL_NAME]["ci_low"] <= penalty[MODEL_NAME]["point"]
                assert penalty[MODEL_NAME]["point"] <= penalty[MODEL_NAME]["ci_high"]

    def test_the_corpus_is_the_archive_and_not_the_rolling_window(self, record):
        """FR-511: an observation must not age out of this experiment."""
        assert record["corpus"]["bound_to_rolling_window"] is False
        assert record["corpus"]["seasons"] == list(FIXTURE_SEASONS)

    def test_recency_weighting_is_measured_rather_than_asserted(self, record):
        """FR-512: the comparison is in the record, whichever way it went."""
        effect = record["recency_weighting"]
        assert effect["reference_decay"] == 0.0
        assert {c["reset_season"] for c in effect["comparisons"]} == {2014, 2017}
        for comparison in effect["comparisons"]:
            assert comparison["recency_decay"] == 0.5
            assert comparison["reduced_the_penalty"] is (
                comparison["penalty_at_this_decay"]
                < comparison["penalty_at_zero_decay"]
            )
        assert isinstance(
            effect["reduced_the_penalty_on_every_compliant_transition"], bool
        )
