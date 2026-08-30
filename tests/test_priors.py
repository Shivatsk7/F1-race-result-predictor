"""T-204 — strict-prior computation (FR-203).

The two traps these tests exist for are named in `priors.py`: the teammate
trap, where a shift inside a constructor group hands a driver their teammate's
result from the race being predicted, and the sort trap, where `shift(1)` means
"previous race" only if the frame is in chronological order.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.features.priors import (
    PriorError,
    prior_race_count,
    prior_rate,
    prior_stat,
    race_ordinal,
    sort_canonically,
    within_race_percentile,
    within_season_zscore,
)
from tests.fixtures.races import synthetic_corpus


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return sort_canonically(synthetic_corpus())


class TestStrictPriors:
    def test_first_appearance_has_no_prior(self, corpus):
        prior = prior_stat(corpus, entity="driver_id", value="classified_position")
        first_race = corpus["round"].eq(1) & corpus["season"].eq(corpus["season"].min())
        assert prior[first_race].isna().all()

    def test_prior_excludes_the_current_race(self, corpus):
        # A driver's prior mean at race n must equal the mean of races 1..n-1,
        # computed independently of the implementation under test.
        driver = corpus["driver_id"].iloc[0]
        prior = prior_stat(corpus, entity="driver_id", value="classified_position")
        rows = corpus[corpus["driver_id"] == driver].reset_index(drop=True)
        values = rows["classified_position"].astype(float)
        got = prior[corpus["driver_id"] == driver].reset_index(drop=True)
        for n in range(1, len(rows)):
            assert got[n] == pytest.approx(values[:n].mean())

    def test_changing_a_race_result_cannot_move_its_own_prior(self, corpus):
        before = prior_stat(corpus, entity="driver_id", value="classified_position")
        altered = corpus.copy()
        last = altered.index[-1]
        altered.loc[last, "classified_position"] = 99
        after = prior_stat(altered, entity="driver_id", value="classified_position")
        assert before[last] == after[last] or (
            pd.isna(before[last]) and pd.isna(after[last])
        )

    def test_window_bounds_the_lookback(self, corpus):
        wide = prior_stat(corpus, entity="driver_id", value="classified_position")
        narrow = prior_stat(
            corpus, entity="driver_id", value="classified_position", window=2
        )
        assert not wide.equals(narrow)


class TestTeammateTrap:
    """A constructor has two cars in every race. A row-level shift leaks."""

    def test_both_cars_of_a_team_get_the_same_constructor_prior(self, corpus):
        prior = prior_stat(
            corpus, entity="constructor_id", value="points", within_race="sum"
        )
        distinct = (
            corpus.assign(_p=prior)
            .groupby(["season", "round", "constructor_id"], dropna=False)["_p"]
            .nunique(dropna=False)
        )
        assert (distinct <= 1).all()

    def test_a_teams_own_race_does_not_enter_its_prior(self, corpus):
        prior = prior_stat(
            corpus, entity="constructor_id", value="points", within_race="sum"
        )
        team = corpus["constructor_id"].iloc[0]
        rows = corpus[corpus["constructor_id"] == team]
        per_race = rows.groupby(["season", "round"], sort=False)["points"].sum()
        got = rows.assign(_p=prior).groupby(["season", "round"], sort=False)["_p"].first()
        expected = per_race.expanding().mean().shift(1)
        pd.testing.assert_series_equal(
            got.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestSortTrap:
    def test_an_unsorted_frame_is_refused(self, corpus):
        shuffled = corpus.sample(frac=1.0, random_state=1)
        with pytest.raises(PriorError, match="not in canonical"):
            prior_stat(shuffled, entity="driver_id", value="points")

    def test_sorting_is_idempotent(self, corpus):
        assert sort_canonically(sort_canonically(corpus)).equals(sort_canonically(corpus))


class TestRaceOrdinal:
    def test_one_ordinal_per_race_shared_by_every_row(self, corpus):
        ordinal = race_ordinal(corpus)
        per_race = corpus.assign(_o=ordinal).groupby(["season", "round"])["_o"].nunique()
        assert (per_race == 1).all()

    def test_ordinals_are_dense_and_chronological(self, corpus):
        ordinal = race_ordinal(corpus)
        assert sorted(ordinal.unique()) == list(range(ordinal.nunique()))
        assert ordinal.is_monotonic_increasing


class TestRatesAndCounts:
    def test_prior_rate_is_the_prior_mean_of_the_predicate(self, corpus):
        wins = corpus["classified_position"] == 1
        rate = prior_rate(corpus, entity="driver_id", mask=wins)
        assert rate.dropna().between(0.0, 1.0).all()

    def test_prior_race_count_starts_at_zero(self, corpus):
        count = prior_race_count(corpus, entity="driver_id")
        assert count.min() == 0
        # Every driver's first race, and only those, has a zero count.
        firsts = corpus.groupby("driver_id", sort=False).head(1).index
        assert set(count[count == 0].index) == set(firsts)


class TestRelativeForms:
    """FR-209's three admissible expressions."""

    def test_percentile_is_bounded_and_orientated(self, corpus):
        pct = within_race_percentile(
            corpus, corpus["classified_position"].astype(float), higher_is_better=False
        )
        assert pct.between(0.0, 1.0).all()
        # The race winner is the best-placed car, so it takes the top percentile.
        winners = corpus["classified_position"] == 1
        assert (pct[winners] == 1.0).all()

    def test_percentile_leaves_unknown_values_unknown(self, corpus):
        values = corpus["classified_position"].astype(float)
        values.iloc[0] = None
        pct = within_race_percentile(corpus, values, higher_is_better=False)
        assert pd.isna(pct.iloc[0])

    def test_zscore_is_centred_within_each_season(self, corpus):
        z = within_season_zscore(corpus, corpus["points"].astype(float))
        by_season = z.groupby(corpus["season"]).mean()
        assert by_season.abs().max() < 1e-9

    def test_zscore_of_a_constant_season_is_zero_not_infinite(self, corpus):
        flat = pd.Series(3.0, index=corpus.index)
        assert (within_season_zscore(corpus, flat) == 0.0).all()
