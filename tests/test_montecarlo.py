"""T-501..T-504 — the Monte Carlo race simulator (FR-601..605, FR-751, FR-751a).

Fitting is exercised on the shared synthetic corpus rather than the real one:
these tests need to run offline in well under a second, and what they check —
determinism, the probability-sum identity, and that parameters actually come
from the data handed in — does not depend on the corpus being real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.reference.circuits import OvertakingTier
from f1pred.simulate.montecarlo import (
    POSITION_PROB_PREFIX,
    SimulationError,
    SimulationParameters,
    aggregate_positions,
    assert_probabilities_sum_to_one,
    fit_simulation_parameters,
    pace_noise_std_from_interval,
    simulate_race,
)
from tests.fixtures.races import synthetic_corpus

PACE_NOISE_STD = 1.5


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return synthetic_corpus(seasons=(2013, 2014, 2015, 2016), rounds=5)


@pytest.fixture(scope="module")
def params(corpus: pd.DataFrame) -> SimulationParameters:
    return fit_simulation_parameters(corpus, pace_noise_std=PACE_NOISE_STD)


def _one_race(corpus: pd.DataFrame, *, season: int, round_: int) -> pd.DataFrame:
    race = corpus[(corpus["season"] == season) & (corpus["round"] == round_)].copy()
    race["predicted_position"] = race["grid_position"].astype(float)
    return race


class TestFitSimulationParameters:
    def test_refuses_a_corpus_missing_required_columns(self):
        with pytest.raises(SimulationError, match="missing"):
            fit_simulation_parameters(pd.DataFrame({"season": [2020]}), pace_noise_std=1.0)

    def test_refuses_a_non_positive_pace_noise_std(self, corpus: pd.DataFrame):
        with pytest.raises(SimulationError, match="pace_noise_std"):
            fit_simulation_parameters(corpus, pace_noise_std=0.0)

    def test_refuses_an_empty_corpus(self):
        empty = synthetic_corpus(seasons=(2013,), rounds=1).iloc[0:0]
        with pytest.raises(SimulationError, match="empty"):
            fit_simulation_parameters(empty, pace_noise_std=1.0)

    def test_retirement_rate_matches_the_corpus_by_construction(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        # The fixture retires exactly the last two of every ten-driver field.
        assert params.corpus_retirement_rate == pytest.approx(0.20, abs=0.01)

    def test_every_constructor_and_circuit_seen_gets_a_rate(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        for constructor_id in corpus["constructor_id"].unique():
            assert constructor_id in params.constructor_retirement_rate
        for circuit_id in corpus["circuit_id"].unique():
            assert circuit_id in params.circuit_retirement_multiplier
            assert circuit_id in params.circuit_safety_car_rate

    def test_tier_swap_rate_is_declared_for_every_tier_present(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        tiers_present = {
            {"albert_park": OvertakingTier.MODERATE, "monaco": OvertakingTier.VERY_HARD,
             "monza": OvertakingTier.EASY, "spa": OvertakingTier.EASY}[c]
            for c in corpus["circuit_id"].unique()
        }
        assert tiers_present <= set(params.tier_swap_rate)
        for rate in params.tier_swap_rate.values():
            assert 0.0 <= rate <= 1.0

    def test_no_observed_weather_column_leaves_the_multiplier_neutral(
        self, params: SimulationParameters
    ):
        # synthetic_corpus carries no weather_actual_precip_mm at all.
        assert params.weather_fitted is False
        assert params.wet_pace_multiplier == 1.0

    def test_weather_multiplier_is_fitted_when_the_column_is_present(
        self, corpus: pd.DataFrame
    ):
        rng = np.random.default_rng(0)
        with_weather = corpus.copy()
        race_keys = with_weather[["season", "round"]].drop_duplicates()
        wet_races = set(
            map(tuple, race_keys.sample(frac=0.3, random_state=1).to_numpy())
        )
        is_wet = with_weather[["season", "round"]].apply(tuple, axis=1).isin(wet_races)
        precip = np.where(is_wet, rng.uniform(2.0, 10.0, len(with_weather)), 0.0)
        with_weather["weather_actual_precip_mm"] = precip

        fitted = fit_simulation_parameters(with_weather, pace_noise_std=PACE_NOISE_STD)
        assert fitted.weather_fitted is True
        assert fitted.corpus_wet_rate == pytest.approx(0.3, abs=0.05)


class TestPaceNoiseStdFromInterval:
    def test_converts_a_declared_coverage(self):
        std = pace_noise_std_from_interval(half_width=1.2816, coverage=0.80)
        assert std == pytest.approx(1.0, abs=1e-3)

    def test_refuses_an_undeclared_coverage(self):
        with pytest.raises(SimulationError, match="z-score"):
            pace_noise_std_from_interval(half_width=1.0, coverage=0.66)

    def test_refuses_a_non_positive_half_width(self):
        with pytest.raises(SimulationError, match="half-width"):
            pace_noise_std_from_interval(half_width=0.0, coverage=0.80)


class TestSimulateRace:
    def test_refuses_a_race_frame_missing_required_columns(self, params: SimulationParameters):
        with pytest.raises(SimulationError, match="missing"):
            simulate_race(pd.DataFrame({"driver_id": ["a"]}), params)

    def test_refuses_fewer_than_two_drivers(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2013, round_=1).iloc[:1]
        with pytest.raises(SimulationError, match="fewer than two"):
            simulate_race(race, params, n_simulations=10)

    def test_refuses_a_race_frame_spanning_two_circuits(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2013, round_=1)
        race = race.copy()
        race.iloc[0, race.columns.get_loc("circuit_id")] = "a_different_circuit"
        with pytest.raises(SimulationError, match="single circuit_id"):
            simulate_race(race, params, n_simulations=10)

    def test_determinism_same_seed_gives_identical_output(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2014, round_=2)
        first = simulate_race(race, params, n_simulations=200, seed=42)
        second = simulate_race(race, params, n_simulations=200, seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_give_different_output(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2014, round_=2)
        first = simulate_race(race, params, n_simulations=200, seed=1)
        second = simulate_race(race, params, n_simulations=200, seed=2)
        assert not first["expected_finish"].equals(second["expected_finish"])

    def test_probabilities_sum_to_one_for_every_driver(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2015, round_=3)
        result = simulate_race(race, params, n_simulations=500, seed=7)
        prob_cols = [c for c in result.columns if c.startswith(POSITION_PROB_PREFIX)]
        assert len(prob_cols) == len(race)
        totals = result[prob_cols].sum(axis=1)
        assert (totals - 1.0).abs().max() < 1e-6

    def test_win_podium_points_probabilities_are_emitted_and_consistent(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2015, round_=3)
        result = simulate_race(race, params, n_simulations=500, seed=7)
        for column in ("win_probability", "podium_probability", "points_probability", "dnf_probability"):
            assert column in result.columns
            assert result[column].between(0.0, 1.0).all()
        # Podium is a superset of a win, and points a superset of a podium.
        assert (result["podium_probability"] >= result["win_probability"] - 1e-9).all()
        assert (result["points_probability"] >= result["podium_probability"] - 1e-9).all()
        assert result["win_probability"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_pole_sitter_wins_more_often_than_the_back_of_the_grid(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2013, round_=1)
        result = simulate_race(race, params, n_simulations=1000, seed=3)
        by_grid = result.set_index("grid_position").sort_index()
        best, worst = by_grid.index.min(), by_grid.index.max()
        assert by_grid.loc[best, "win_probability"].max() >= by_grid.loc[worst, "win_probability"].max()

    def test_very_hard_tier_shuffles_less_than_easy_tier_on_average(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        # synthetic_corpus cycles circuits (rnd - 1) % 4 over
        # ("albert_park", "monaco", "monza", "spa"): round 2 is always
        # Monaco (VERY_HARD), round 3 always Monza (EASY). Same grid
        # construction, same pace gaps: the harder tier should preserve grid
        # order more often across draws.
        monaco_race = _one_race(corpus, season=2013, round_=2)
        monza_race = _one_race(corpus, season=2013, round_=3)

        monaco_result = simulate_race(monaco_race, params, n_simulations=800, seed=11)
        monza_result = simulate_race(monza_race, params, n_simulations=800, seed=11)

        def mean_abs_shuffle(result: pd.DataFrame) -> float:
            return float((result["expected_finish"] - result["grid_position"]).abs().mean())

        assert mean_abs_shuffle(monaco_result) <= mean_abs_shuffle(monza_result) + 1.0


class TestAssertProbabilitiesSumToOne:
    def test_passes_on_a_correctly_aggregated_result(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2016, round_=4)
        result = simulate_race(race, params, n_simulations=300, seed=5)
        assert_probabilities_sum_to_one(result)  # must not raise

    def test_fails_when_a_probability_is_corrupted(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2016, round_=4)
        result = simulate_race(race, params, n_simulations=300, seed=5).copy()
        result.loc[0, f"{POSITION_PROB_PREFIX}01"] += 0.5
        with pytest.raises(SimulationError, match="FR-605"):
            assert_probabilities_sum_to_one(result)

    def test_fails_when_a_required_column_is_missing(self):
        with pytest.raises(SimulationError, match="missing"):
            assert_probabilities_sum_to_one(pd.DataFrame({f"{POSITION_PROB_PREFIX}01": [1.0]}), n_drivers=2)


class TestWeatherRoleSeparation:
    """T-510a: a pre-race prior may drive the simulation; an observed,
    race-day reading may not — even when it rides along on the same frame."""

    def test_observed_weather_is_not_among_the_columns_the_simulator_reads(self):
        from f1pred.simulate.montecarlo import REQUIRED_RACE_COLUMNS

        assert "weather_actual_precip_mm" not in REQUIRED_RACE_COLUMNS
        assert "weather_actual_temp_max_c" not in REQUIRED_RACE_COLUMNS

    def test_an_observed_weather_column_on_the_race_frame_changes_nothing(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        race = _one_race(corpus, season=2015, round_=3)
        without = simulate_race(race, params, n_simulations=300, seed=9)

        with_observed = race.copy()
        # A same-race observed reading, wildly different from anything the
        # pre-race prior would say — if the simulator ever read this column,
        # the result would move.
        with_observed["weather_actual_precip_mm"] = 500.0
        result = simulate_race(with_observed, params, n_simulations=300, seed=9)

        pd.testing.assert_frame_equal(without, result)

    def test_the_pre_race_prior_does_change_the_simulation(
        self, corpus: pd.DataFrame, params: SimulationParameters
    ):
        import dataclasses

        # This fixture's corpus carries no observed weather, so its fitted
        # multiplier is the neutral 1.0 (correctly — there was nothing to fit
        # from) and a 1.0 multiplier is a genuine no-op regardless of how wet
        # a race is drawn to be. Substitute a real multiplier to exercise the
        # prior's *effect*, which is the thing this test checks.
        wet_params = dataclasses.replace(
            params, wet_pace_multiplier=3.0, weather_fitted=True
        )
        race = _one_race(corpus, season=2015, round_=3)
        dry = race.copy()
        dry["weather_prior_wet"] = 0.0
        wet = race.copy()
        wet["weather_prior_wet"] = 1.0

        dry_result = simulate_race(dry, wet_params, n_simulations=2000, seed=13)
        wet_result = simulate_race(wet, wet_params, n_simulations=2000, seed=13)

        assert not dry_result["expected_finish"].equals(wet_result["expected_finish"])
