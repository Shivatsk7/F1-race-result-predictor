"""AC-10: no prediction anywhere in the system is emitted as a bare point.

The guard `assert_carries_uncertainty` has existed since P4 and is called at
exactly one place — `TrainedRanker.predict_with_interval`. That is correct as
far as it goes, and it is not the criterion. AC-10 says *anywhere*, and the
system grew two more publication paths afterwards: the Monte Carlo table and,
in P6, `f1pred predict`.

So these tests walk the exits rather than the guard: every frame this project
hands a reader is checked for a point estimate standing on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.models.base import BarePointEstimateError, assert_carries_uncertainty

#: Anything a reader could mistake for "the answer".
POINT_COLUMNS = ("predicted_position", "expected_finish")


class TestTheGuardItself:
    def test_a_point_with_both_bounds_passes(self):
        frame = pd.DataFrame(
            {
                "predicted_position": [1.0, 2.0],
                "predicted_position_lower": [1.0, 1.0],
                "predicted_position_upper": [4.0, 6.0],
            }
        )
        assert_carries_uncertainty(frame)

    def test_a_point_with_no_bounds_is_refused(self):
        with pytest.raises(BarePointEstimateError):
            assert_carries_uncertainty(pd.DataFrame({"predicted_position": [1.0]}))

    def test_a_null_bound_is_refused_because_it_only_looks_like_an_interval(self):
        frame = pd.DataFrame(
            {
                "predicted_position": [1.0, 2.0],
                "predicted_position_lower": [1.0, np.nan],
                "predicted_position_upper": [4.0, 6.0],
            }
        )
        with pytest.raises(BarePointEstimateError, match="null interval bound"):
            assert_carries_uncertainty(frame)

    def test_a_frame_with_no_prediction_in_it_passes_trivially(self):
        assert_carries_uncertainty(pd.DataFrame({"driver_id": ["norris"]}))


class TestEveryPublicationPath:
    """The exits, one by one. A new one added without uncertainty fails here."""

    @pytest.fixture(scope="class")
    def simulated(self):
        """Fitted from a synthetic corpus, the same way the real path fits."""
        from f1pred.simulate.montecarlo import fit_simulation_parameters, simulate_race
        from tests.fixtures.races import synthetic_corpus

        corpus = synthetic_corpus(seasons=(2013, 2014, 2015, 2016), rounds=5)
        params = fit_simulation_parameters(corpus, pace_noise_std=1.5)
        race = corpus[(corpus["season"] == 2016) & (corpus["round"] == 5)].copy()
        race["predicted_position"] = race["grid_position"].astype(float)
        return simulate_race(race, params, n_simulations=50, seed=1)

    def test_the_simulation_table_carries_a_full_distribution(self, simulated):
        """FR-605's distribution is a superset of an interval, not an alternative.

        `expected_finish` is a point estimate, so if the per-position columns
        ever stopped being emitted the table would be publishing a bare one.
        """
        assert "expected_finish" in simulated.columns
        distribution = [c for c in simulated.columns if c.startswith("prob_p")]
        assert len(distribution) == len(simulated), (
            "the table publishes an expected finishing position without a "
            "position distribution behind it"
        )
        assert np.allclose(simulated[distribution].sum(axis=1), 1.0, atol=1e-6)

    def test_the_simulation_table_carries_event_probabilities_too(self, simulated):
        for column in ("win_probability", "podium_probability", "points_probability"):
            assert column in simulated.columns
            assert simulated[column].notna().all()

    def test_the_ranker_cannot_publish_without_calling_the_guard(self):
        """A rewrite that dropped the guard call would pass every other test."""
        import inspect

        from f1pred.models.ranker import TrainedRanker

        source = inspect.getsource(TrainedRanker.predict_with_interval)
        assert "assert_carries_uncertainty" in source

    def test_the_cli_prediction_table_carries_an_interval_column(self):
        """`f1pred predict` renders `interval` per row; the code must build it."""
        import inspect

        from f1pred import predict as predict_module

        source = inspect.getsource(predict_module.predict_race)
        assert '"interval"' in source
        assert "predict_with_interval" in source, (
            "the prediction path no longer routes through the guarded method"
        )


class TestTheRankerOutput:
    def test_a_real_prediction_frame_passes_the_guard(self):
        """End to end on the trained model, when one is on disk."""
        from f1pred import artifacts
        from f1pred.models.ranker import TrainedRanker

        if not artifacts.model_is_present():
            pytest.skip("no trained model on disk; run `f1pred train`")
        model = TrainedRanker.load(artifacts.MODEL_STEM)
        race = pd.DataFrame(
            {name: np.linspace(0.1, 0.9, 8) for name in model.feature_names}
        )
        assert_carries_uncertainty(model.predict_with_interval(race))
