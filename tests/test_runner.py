"""The headline run (AC-3, AC-4, FR-506).

What is worth testing here is not the score — that is measured against the real
corpus and committed to `results/`, and a fixture that produced a particular
number would only be testing the fixture. What is worth testing is that the run
cannot quietly become a different run: that every baseline FR-301 and FR-302
name is present, that the model and the baselines are scored on one holdout
through one path, and that the record says which corpus produced it.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.corpus import CorpusWindow
from f1pred.dataset import experiment
from f1pred.evaluate.harness import REFERENCE_BASELINE, EvaluationError, render_verdict
from f1pred.evaluate.runner import baselines, run, train_model
from f1pred.evaluate.split import ChronologicalSplit
from f1pred.models.ranker import Hyperparameters
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


def quick() -> Hyperparameters:
    return Hyperparameters(n_estimators=30, max_depth=3)


@pytest.fixture(scope="module")
def scored(built, split, tmp_path_factory):
    """One run, reused. The random control is off: FR-303's thousand draws are
    the harness's own test, exercised in `test_baselines.py`, and repeating
    them here would cost a minute to re-establish a settled fact."""
    path = tmp_path_factory.mktemp("results") / "headline.json"
    record, model = run(
        experiment=built,
        split=split,
        params=quick(),
        random_control=False,
        path=path,
    )
    return record, model, path


class TestBaselineSet:
    def test_both_named_baselines_are_present(self):
        """FR-301 and FR-302. A missing one renders the same verdict."""
        assert {b.name for b in baselines()} == {"grid_order", "standings_order"}

    def test_the_reference_baseline_is_among_them(self):
        assert REFERENCE_BASELINE in {b.name for b in baselines()}


class TestHeadlineRun:
    def test_the_model_and_every_baseline_are_scored(self, scored):
        record, model, _ = scored
        assert set(record["predictors"]) == {"grid_order", "standings_order", model.name}

    def test_all_predictors_face_the_same_races(self, scored):
        """FR-304: one holdout, one scoring path, no exceptions for the model."""
        record, _, _ = scored
        counts = {p["races_scored"] for p in record["predictors"].values()}
        assert len(counts) == 1

    def test_every_predictor_but_the_reference_carries_a_delta(self, scored):
        record, _, _ = scored
        for name, entry in record["predictors"].items():
            if name == REFERENCE_BASELINE:
                continue
            assert f"delta_vs_{REFERENCE_BASELINE}" in entry

    def test_the_verdict_is_computed_and_renders(self, scored):
        record, _, _ = scored
        assert "verdict" in record
        text = render_verdict(record)
        assert "grid_order" in text
        assert ("BEATS THE GRID" in text) or ("DOES NOT BEAT" in text)

    def test_the_model_never_trains_on_the_holdout(self, scored, split):
        record, model, _ = scored
        assert model.training_seasons == split.training_seasons
        assert max(model.training_seasons) < split.boundary_season


class TestTheRecord:
    def test_the_model_record_travels_with_the_result(self, scored):
        """FR-404's sixth field can only be filled after the harness runs, so
        the record proves the two halves were joined."""
        record, _, _ = scored
        assert record["model"]["evaluation_metrics"]
        assert record["model"]["code_commit"]
        assert record["model"]["training_rows"] > 0

    def test_the_intervals_are_labelled_uncalibrated(self, scored):
        record, _, _ = scored
        assert record["model"]["interval"]["status"] == "UNCALIBRATED"

    def test_the_record_says_which_enrichment_the_run_had(self, scored):
        """A run without weather is a different experiment from one with it,
        and afterwards only this field distinguishes them."""
        record, _, _ = scored
        assert "absent_sources" in record
        assert "weather_prior_wet_probability" in record["absent_sources"]

    def test_the_record_names_the_corpus_it_rests_on(self, scored, split):
        record, _, _ = scored
        assert record["split"]["training_seasons"] == list(split.training_seasons)
        assert record["split"]["holdout_seasons"] == list(split.holdout)
        assert record["corpus_rows"] > 0

    def test_the_file_is_written_and_is_machine_readable(self, scored):
        """FR-506."""
        record, _, path = scored
        assert path.exists()
        assert json.loads(path.read_text())["primary_metric"] == record["primary_metric"]

    def test_the_filename_names_the_holdout_it_reports(self, built, split, tmp_path):
        """A file called `model.json` is one rerun from describing a different
        holdout than the report citing it."""
        record, _ = run(
            experiment=built,
            split=split,
            params=quick(),
            random_control=False,
            path=tmp_path / "x.json",
        )
        from f1pred.evaluate.runner import _filename

        assert _filename("model_vs_baselines", split) == "model_vs_baselines_2018_2020.json"


class TestRefusals:
    def test_a_run_without_the_grid_baseline_is_refused(self, built, split):
        """FR-504 measures every delta against the grid, so it is not optional.

        Reached by calling the harness directly, because `runner.run` is
        written so that a caller cannot drop it — which is the point.
        """
        from f1pred.evaluate.harness import evaluate

        model = train_model(built, split=split, params=quick())
        with pytest.raises(EvaluationError, match=REFERENCE_BASELINE):
            evaluate(built.frame, (model,), split=split)

    def test_the_model_is_trained_on_the_experiment_s_own_features(self, built, split):
        model = train_model(built, split=split, params=quick())
        assert model.feature_names == built.feature_names
