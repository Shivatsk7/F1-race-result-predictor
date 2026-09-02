"""The in-sample check the harness does not do (FR-405, FR-902).

These tests do not assert a gap size. The gap is a property of the corpus and
the configuration, and pinning it to a number here would make an ordinary
feature change look like a test failure. What is asserted is that the
diagnostic *can distinguish* a memorising fit from a stopped one, that its
reference arm is a predictor incapable of memorising, and that its finding is
computed from the arms rather than written by hand — the three things that
would let it report "fine" while the model was reproducing its training races.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.corpus import CorpusWindow
from f1pred.dataset import experiment
from f1pred.evaluate.fit_diagnostic import (
    STOPPED_ARM,
    UNSTOPPED_ARM,
    WIDE_GAP,
    in_sample_and_holdout,
    render,
    run,
)
from f1pred.evaluate.split import ChronologicalSplit
from f1pred.models.ranker import Hyperparameters, train
from tests.fixtures.races import synthetic_corpus

SEASONS = tuple(range(2013, 2021))


@pytest.fixture(scope="module")
def built():
    return experiment(synthetic_corpus(seasons=SEASONS, rounds=4, transfer=(2016, 2)))


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    return ChronologicalSplit(
        window=CorpusWindow(width=len(SEASONS)),
        holdout_seasons=3,
        season_end_dates={s: date(s, 12, 1) for s in SEASONS},
    )


@pytest.fixture(scope="module")
def record(built, split, tmp_path_factory):
    return run(
        experiment=built,
        split=split,
        params=Hyperparameters(n_estimators=120, max_depth=6),
        path=tmp_path_factory.mktemp("results") / "fit.json",
    )


class TestScoring:
    def test_both_sides_of_the_boundary_are_scored(self, built, split):
        model = train(
            built.frame,
            split=split,
            features=built.feature_names,
            params=Hyperparameters(n_estimators=40, max_depth=3),
        )
        scores = in_sample_and_holdout(model, built.frame, split=split)
        assert set(scores) == {"in_sample", "holdout", "gap"}
        assert scores["gap"] == pytest.approx(scores["in_sample"] - scores["holdout"])

    def test_the_in_sample_side_really_is_the_training_seasons(self, built, split):
        """A diagnostic that scored the holdout twice would report a gap of
        zero and look reassuring."""
        training, holdout = split.split(built.frame)
        assert set(training["season"]) == set(split.training_seasons)
        assert set(holdout["season"]) == set(split.holdout)
        assert not set(training["season"]) & set(holdout["season"])


class TestArms:
    def test_both_arms_are_run(self, record):
        assert set(record["arms"]) == {STOPPED_ARM, UNSTOPPED_ARM}

    def test_the_unstopped_arm_keeps_every_tree(self, record):
        arm = record["arms"][UNSTOPPED_ARM]
        assert arm["early_stopping_rounds"] is None
        assert arm["trees_kept"] == arm["n_estimators_ceiling"]

    def test_the_stopped_arm_is_the_configured_one(self, record):
        assert record["arms"][STOPPED_ARM]["early_stopping_rounds"] is not None

    def test_stopping_narrows_the_gap(self, record):
        """The claim the board rests on, re-measured rather than quoted."""
        assert (
            record["arms"][STOPPED_ARM]["gap"]
            < record["arms"][UNSTOPPED_ARM]["gap"]
        )

    def test_a_memorising_arm_is_labelled(self, record):
        for arm in record["arms"].values():
            assert arm["memorising"] == (arm["gap"] > WIDE_GAP)


class TestReference:
    def test_the_grid_is_scored_on_the_same_split(self, record):
        ref = record["grid_order_reference"]
        assert set(ref) == {"in_sample", "holdout", "gap"}
        assert ref["gap"] == pytest.approx(ref["in_sample"] - ref["holdout"])

    def test_the_reference_cannot_memorise(self, record):
        """Why the grid is the right control: it has no parameters to fit, so
        whatever gap it shows belongs to the corpus and not to a model. Without
        it, a gap could always be explained away as easier training races."""
        assert abs(record["grid_order_reference"]["gap"]) < WIDE_GAP


class TestRecord:
    def test_the_finding_is_computed_not_written(self, record):
        arms = record["arms"]
        assert f"{arms[UNSTOPPED_ARM]['in_sample']:.4f}" in record["finding"]
        assert f"{arms[STOPPED_ARM]['holdout']:.4f}" in record["finding"]

    def test_the_record_names_its_split_and_corpus(self, record, split):
        assert record["split"]["training_seasons"] == list(split.training_seasons)
        assert record["corpus_rows"] > 0

    def test_it_renders(self, record):
        text = render(record)
        assert "in-sample" in text
        assert "grid_order" in text

    def test_the_file_is_written_and_round_trips(self, built, split, tmp_path):
        path = tmp_path / "fit.json"
        written = run(
            experiment=built,
            split=split,
            params=Hyperparameters(n_estimators=40, max_depth=3),
            path=path,
        )
        assert json.loads(path.read_text())["finding"] == written["finding"]
