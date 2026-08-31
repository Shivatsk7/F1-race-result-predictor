"""T-301..T-311 — baselines and the evaluation harness (FR-301..305, FR-500..507).

The bar has to be set before the model exists, and it has to be set by code
that cannot flatter it. Most of what is tested here is that the harness refuses
things: a random split, a delta over mismatched races, an evaluation with no
grid baseline to compare against.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from f1pred.corpus import CorpusWindow
from f1pred.data.target import build_target
from f1pred.evaluate.bootstrap import (
    BootstrapError,
    Interval,
    bootstrap_delta,
    bootstrap_metrics,
)
from f1pred.evaluate.harness import (
    REFERENCE_BASELINE,
    EvaluationError,
    evaluate,
    render_verdict,
    score_predictor,
)
from f1pred.evaluate.metrics import (
    METRICS,
    METRICS_BY_NAME,
    PRIMARY_METRIC,
    MetricError,
    aggregate,
    per_race_scores,
)
from f1pred.evaluate.split import (
    MIN_HOLDOUT_SEASONS,
    MIN_TOTAL_SEASONS,
    MIN_TRAINING_SEASONS,
    ChronologicalSplit,
    SplitError,
)
from f1pred.models.base import Predictor, rank_to_positions
from f1pred.models.baselines import (
    BaselineError,
    GridOrderBaseline,
    RandomOrderControl,
    StandingsOrderBaseline,
    championship_points_prior,
)
from tests.fixtures.races import synthetic_corpus

SEASONS = tuple(range(2012, 2028))


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    df = synthetic_corpus(seasons=SEASONS, rounds=4)
    df = build_target(df).sort_values(
        ["season", "round", "driver_id"], kind="stable"
    ).reset_index(drop=True)
    df["championship_points_prior"] = championship_points_prior(df)
    return df


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    # Width 14 spans 2012-2025 given the fixture's end dates. The window
    # resolves against a clock, so a width that reaches before
    # `earliest_available` is refused rather than silently truncated.
    return ChronologicalSplit(
        window=CorpusWindow(width=14, earliest_available=2012),
        holdout_seasons=3,
        season_end_dates={s: pd.Timestamp(f"{s}-12-01").date() for s in SEASONS},
    )


class TestMetrics:
    """FR-500, FR-502."""

    def test_there_are_exactly_nine(self):
        assert len(METRICS) == 9

    def test_the_primary_metric_is_pinned_and_is_spearman(self):
        # FR-500: wherever anything says "the primary metric", it means this.
        assert PRIMARY_METRIC is METRICS_BY_NAME["spearman"]

    def test_a_perfect_prediction_scores_perfectly(self):
        keys = pd.DataFrame({"season": [2020] * 5, "round": [1] * 5})
        actual = pd.Series([1, 2, 3, 4, 5])
        got = aggregate(per_race_scores(keys, actual, actual))
        assert got["mae"] == 0.0
        assert got["exact_accuracy"] == 1.0
        assert got["spearman"] == pytest.approx(1.0)
        assert got["podium_set_accuracy"] == 1.0

    def test_a_reversed_prediction_anticorrelates(self):
        keys = pd.DataFrame({"season": [2020] * 5, "round": [1] * 5})
        actual = pd.Series([1, 2, 3, 4, 5])
        got = aggregate(per_race_scores(keys, actual[::-1].reset_index(drop=True), actual))
        assert got["spearman"] == pytest.approx(-1.0)

    def test_podium_set_is_stricter_than_top_three_overlap(self):
        # Two of three right: partial credit on one, zero on the other.
        keys = pd.DataFrame({"season": [2020] * 5, "round": [1] * 5})
        actual = pd.Series([1, 2, 3, 4, 5])
        predicted = pd.Series([1, 2, 4, 3, 5])
        got = aggregate(per_race_scores(keys, predicted, actual))
        assert got["top_3_set_accuracy"] == pytest.approx(2 / 3)
        assert got["podium_set_accuracy"] == 0.0

    def test_races_are_weighted_equally_regardless_of_field_size(self):
        # FR-500 averages unweighted across races. A 20-car race and a 4-car
        # race count the same, or seasons get weighted by grid size.
        keys = pd.DataFrame(
            {"season": [2020] * 24, "round": [1] * 20 + [2] * 4}
        )
        actual = pd.Series(list(range(1, 21)) + list(range(1, 5)))
        predicted = pd.Series(list(range(1, 21)) + list(range(4, 0, -1)))
        scores = per_race_scores(keys, predicted, actual)
        assert len(scores) == 2
        assert aggregate(scores)["spearman"] == pytest.approx(0.0)

    def test_mismatched_lengths_are_refused(self):
        keys = pd.DataFrame({"season": [2020], "round": [1]})
        with pytest.raises(MetricError, match="length mismatch"):
            per_race_scores(keys, pd.Series([1, 2]), pd.Series([1]))


class TestBaselines:
    """FR-301, FR-302, FR-303, FR-304."""

    def test_all_three_satisfy_the_predictor_protocol(self):
        for b in (
            GridOrderBaseline(),
            StandingsOrderBaseline(),
            RandomOrderControl().predictors()[0],
        ):
            assert isinstance(b, Predictor)

    def test_grid_baseline_predicts_the_grid(self, corpus):
        race = corpus[(corpus.season == 2014) & (corpus["round"] == 1)]
        got = GridOrderBaseline().predict(race)
        expected = race["grid_position"].rank(method="first").astype("int64")
        pd.testing.assert_series_equal(got, expected, check_names=False)

    def test_predictions_are_a_permutation_of_the_field(self, corpus):
        race = corpus[(corpus.season == 2014) & (corpus["round"] == 1)]
        for b in (GridOrderBaseline(), StandingsOrderBaseline()):
            got = b.predict(race)
            assert sorted(got) == list(range(1, len(race) + 1)), b.name

    def test_standings_baseline_puts_the_points_leader_first(self, corpus):
        race = corpus[(corpus.season == 2014) & (corpus["round"] == 4)].copy()
        race.loc[race.index[3], "championship_points_prior"] = 999.0
        got = StandingsOrderBaseline().predict(race)
        assert got.loc[race.index[3]] == 1

    def test_a_missing_column_raises_rather_than_substituting(self, corpus):
        race = corpus[(corpus.season == 2014) & (corpus["round"] == 1)].drop(
            columns=["grid_position"]
        )
        with pytest.raises(BaselineError, match="harness bug"):
            GridOrderBaseline().predict(race)

    def test_championship_points_exclude_the_current_race(self, corpus):
        first = corpus[(corpus.season == 2013) & (corpus["round"] == 1)]
        assert (first["championship_points_prior"] == 0.0).all()

    def test_the_random_control_refuses_to_be_thin(self):
        # FR-303: fewer than 1000 draws makes the floor meaningless.
        with pytest.raises(BaselineError, match="at least 1000"):
            RandomOrderControl(draws=10)

    def test_random_draws_are_seeded_and_reproducible(self, corpus):
        race = corpus[(corpus.season == 2014) & (corpus["round"] == 1)]
        a = RandomOrderControl().predictors()[7].predict(race)
        b = RandomOrderControl().predictors()[7].predict(race)
        pd.testing.assert_series_equal(a, b)

    def test_positions_are_dense_with_no_ties(self):
        # A tie would put two cars in P3 and none in P4, and every metric
        # would score against a position that does not exist.
        values = pd.Series([5.0, 5.0, 5.0, 1.0])
        got = rank_to_positions(values, ascending=True)
        assert sorted(got) == [1, 2, 3, 4]


class TestChronologicalSplit:
    """FR-403, FR-501, FR-501a, plan D-4."""

    def test_no_api_accepts_a_random_split(self):
        params = set(inspect.signature(ChronologicalSplit).parameters)
        for forbidden in ("test_size", "shuffle", "random_state", "stratify"):
            assert forbidden not in params

    def test_it_takes_a_window_not_a_season_list(self):
        # D-9: a literal season list is correct for exactly one year.
        assert "window" in inspect.signature(ChronologicalSplit).parameters

    def test_holdout_is_the_most_recent_seasons(self, split):
        assert split.holdout == split.seasons[-3:]
        assert max(split.training_seasons) < min(split.holdout)

    def test_the_constitutional_minima_are_enforced(self):
        window = CorpusWindow(width=14, earliest_available=2012)
        dates = {s: pd.Timestamp(f"{s}-12-01").date() for s in SEASONS}
        with pytest.raises(SplitError, match="below the Constitution"):
            ChronologicalSplit(window=window, holdout_seasons=2, season_end_dates=dates)

    def test_the_minima_are_the_declared_constants(self):
        assert (MIN_TOTAL_SEASONS, MIN_TRAINING_SEASONS, MIN_HOLDOUT_SEASONS) == (8, 5, 3)

    def test_splitting_never_puts_a_season_on_both_sides(self, corpus, split):
        train, holdout = split.split(corpus)
        assert not set(train.season) & set(holdout.season)

    def test_leakage_assertion_catches_a_holdout_season_in_training(self, corpus, split):
        train, holdout = split.split(corpus)
        with pytest.raises(SplitError, match="at or after the holdout boundary"):
            split.assert_no_leakage(pd.concat([train, holdout]))

    def test_a_clean_training_set_passes_the_assertion(self, corpus, split):
        train, _ = split.split(corpus)
        split.assert_no_leakage(train)

    def test_describe_records_the_shortfall_field(self, split):
        # FR-501a: recorded in every results file, never absorbed.
        got = split.describe()
        assert got["meets_constitutional_minimum"] is True
        assert got["minimum_required"]["training"] == MIN_TRAINING_SEASONS


class TestBootstrap:
    """FR-503, FR-504, plan D-5."""

    def test_intervals_bracket_the_point_estimate(self, corpus, split):
        _, holdout = split.split(corpus)
        scores = score_predictor(GridOrderBaseline(), holdout)
        got = bootstrap_metrics(scores, replicates=200)
        for name, interval in got.items():
            assert interval.low <= interval.point <= interval.high, name

    def test_it_resamples_races_not_rows(self):
        # D-5: the number of draws per replicate must equal the number of
        # races. Rows would give dishonestly narrow intervals.
        source = inspect.getsource(bootstrap_metrics)
        assert "_replicate_draws(len(scores)" in source

    def test_a_delta_over_mismatched_races_is_refused(self, corpus, split):
        _, holdout = split.split(corpus)
        a = score_predictor(GridOrderBaseline(), holdout)
        b = score_predictor(GridOrderBaseline(), holdout).iloc[:-1]
        with pytest.raises(BootstrapError, match="different races"):
            bootstrap_delta(a, b, replicates=50)

    def test_a_predictor_against_itself_has_a_zero_delta(self, corpus, split):
        _, holdout = split.split(corpus)
        scores = score_predictor(GridOrderBaseline(), holdout)
        got = bootstrap_delta(scores, scores, replicates=100)
        assert got["spearman"].point == pytest.approx(0.0)
        assert not got["spearman"].excludes_zero

    def test_excludes_zero_is_the_significance_test(self):
        assert Interval(0.1, 0.02, 0.18).excludes_zero
        assert not Interval(0.1, -0.02, 0.22).excludes_zero


class TestHarness:
    """FR-304, FR-305, FR-505, FR-506, FR-507."""

    @staticmethod
    @pytest.fixture(scope="class")
    def record(corpus, split):
        return evaluate(
            corpus,
            [GridOrderBaseline(), StandingsOrderBaseline()],
            split=split,
            replicates=200,
        )

    def test_every_predictor_is_scored_by_the_same_function(self, corpus, split):
        # FR-304, structurally: there is one scoring path and it does not know
        # what kind of predictor it holds.
        _, holdout = split.split(corpus)
        for b in (GridOrderBaseline(), StandingsOrderBaseline()):
            assert len(score_predictor(b, holdout)) > 0

    def test_baselines_are_first_class_in_the_output(self, record):
        # FR-305: same structure as everything else, not a footnote.
        assert set(record["predictors"]) >= {"grid_order", "standings_order"}
        for entry in record["predictors"].values():
            assert len(entry["metrics"]) == 9

    def test_an_evaluation_without_the_grid_baseline_is_refused(self, corpus, split):
        with pytest.raises(EvaluationError, match="not optional"):
            evaluate(corpus, [StandingsOrderBaseline()], split=split, replicates=50)

    def test_every_metric_carries_an_interval(self, record):
        for entry in record["predictors"].values():
            for metric in entry["metrics"].values():
                assert {"point", "ci_low", "ci_high"} <= set(metric)

    def test_deltas_are_measured_against_the_grid(self, record):
        assert f"delta_vs_{REFERENCE_BASELINE}" in record["predictors"]["standings_order"]

    def test_segmentation_covers_all_three_axes(self, record):
        # FR-505: aggregate performance must not conceal systematic failure.
        for axis in ("season", "circuit_type", "overtaking_tier"):
            assert record["segmentation"]["grid_order"][axis]

    def test_the_verdict_states_both_directions(self, record):
        v = record["verdict"]
        assert "any_predictor_beats_the_grid" in v
        text = render_verdict(record)
        assert ("BEATS THE GRID" in text) or ("DOES NOT BEAT THE GRID" in text)

    def test_the_verdict_is_computed_not_written(self, record):
        v = record["verdict"]
        expected = bool(v["beaten_by"])
        assert v["any_predictor_beats_the_grid"] is expected

    def test_the_record_is_json_serialisable(self, record):
        assert json.loads(json.dumps(record, default=str))["primary_metric"] == "spearman"

    def test_the_split_travels_with_the_results(self, record):
        # FR-510: W=16 means different seasons in different years.
        assert record["split"]["holdout_seasons"]
        assert record["split"]["window"]["seasons"]
