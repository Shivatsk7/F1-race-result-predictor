import unittest

import numpy as np
import pandas as pd

from src.race_simulator import (
    apply_driver_variability,
    apply_track_adjustment,
    overtake_probability,
    run_monte_carlo,
)


def build_prediction_frame(circuit: str = "Baku", grid_weight: float = 0.45) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver": "VER",
                "team": "Red Bull",
                "grid_position": 1,
                "predicted_position_raw": 2.0,
                "track_archetype": "street",
                "track_grid_weight": grid_weight,
                "is_street": 1,
                "overtaking_difficulty": 3,
                "track_position_importance": 3,
                "safety_car_risk": 5,
                "pit_delta_index": 3,
                "tire_deg_index": 2,
                "weather_volatility": 3,
                "driver_elo_race": 1650,
                "driver_recent_gain_avg": 0.5,
                "constructor_reliability_season": 0.96,
                "circuit": circuit,
            },
            {
                "driver": "NOR",
                "team": "McLaren",
                "grid_position": 4,
                "predicted_position_raw": 1.5,
                "track_archetype": "street",
                "track_grid_weight": grid_weight,
                "is_street": 1,
                "overtaking_difficulty": 3,
                "track_position_importance": 3,
                "safety_car_risk": 5,
                "pit_delta_index": 3,
                "tire_deg_index": 2,
                "weather_volatility": 3,
                "driver_elo_race": 1600,
                "driver_recent_gain_avg": 0.3,
                "constructor_reliability_season": 0.95,
                "circuit": circuit,
            },
            {
                "driver": "LEC",
                "team": "Ferrari",
                "grid_position": 2,
                "predicted_position_raw": 3.0,
                "track_archetype": "street",
                "track_grid_weight": grid_weight,
                "is_street": 1,
                "overtaking_difficulty": 3,
                "track_position_importance": 3,
                "safety_car_risk": 5,
                "pit_delta_index": 3,
                "tire_deg_index": 2,
                "weather_volatility": 3,
                "driver_elo_race": 1580,
                "driver_recent_gain_avg": 0.1,
                "constructor_reliability_season": 0.93,
                "circuit": circuit,
            },
            {
                "driver": "HAD",
                "team": "Racing Bulls",
                "grid_position": 3,
                "predicted_position_raw": 7.0,
                "track_archetype": "street",
                "track_grid_weight": grid_weight,
                "is_street": 1,
                "overtaking_difficulty": 3,
                "track_position_importance": 3,
                "safety_car_risk": 5,
                "pit_delta_index": 3,
                "tire_deg_index": 2,
                "weather_volatility": 3,
                "driver_elo_race": 1450,
                "driver_recent_gain_avg": -0.4,
                "constructor_reliability_season": 0.87,
                "circuit": circuit,
            },
        ]
    )


class RaceSimulatorTests(unittest.TestCase):
    def test_track_adjustment_anchors_more_on_high_grid_tracks(self):
        low_track = build_prediction_frame(grid_weight=0.20)
        low_track["track_archetype"] = "chaos"
        low_track["benchmark_group"] = "chaos"
        low_track["is_street"] = 0
        low_track["track_position_importance"] = 3
        low_track["overtaking_difficulty"] = 3
        low_track["tire_deg_index"] = 3

        high_track = build_prediction_frame(grid_weight=0.85)
        high_track["track_archetype"] = "high_grid"
        high_track["benchmark_group"] = "high_grid"
        high_track["is_street"] = 1
        high_track["track_position_importance"] = 5
        high_track["overtaking_difficulty"] = 5
        high_track["tire_deg_index"] = 2

        low_grid_weight = apply_track_adjustment(low_track)
        high_grid_weight = apply_track_adjustment(high_track)

        driver = "HAD"
        low_gap = abs(
            low_grid_weight.loc[low_grid_weight["driver"] == driver, "adjusted_pace"].iloc[0]
            - low_grid_weight.loc[low_grid_weight["driver"] == driver, "grid_position"].iloc[0]
        )
        high_gap = abs(
            high_grid_weight.loc[high_grid_weight["driver"] == driver, "adjusted_pace"].iloc[0]
            - high_grid_weight.loc[high_grid_weight["driver"] == driver, "grid_position"].iloc[0]
        )

        self.assertLess(high_gap, low_gap)

    def test_overtake_probability_increases_with_pace_and_track_openness(self):
        hard_track = overtake_probability(0.5, overtaking_difficulty_score=0.95, overtaking_skill=0.5)
        easy_track = overtake_probability(1.5, overtaking_difficulty_score=0.20, overtaking_skill=0.8)

        self.assertGreater(easy_track, hard_track)

    def test_monte_carlo_outputs_calibrated_probability_columns(self):
        frame = apply_driver_variability(apply_track_adjustment(build_prediction_frame()))
        results = run_monte_carlo(frame, n_simulations=300, random_state=7)
        prob_columns = [column for column in results.columns if column.startswith("prob_p")]

        self.assertEqual(len(results), 4)
        self.assertEqual(len(prob_columns), 4)
        self.assertTrue({"expected_finish", "win_probability", "podium_probability"}.issubset(results.columns))
        np.testing.assert_allclose(results[prob_columns].sum(axis=1).to_numpy(), np.ones(4), atol=0.02)
        self.assertTrue((results["finish_p95"] >= results["finish_p05"]).all())


if __name__ == "__main__":
    unittest.main()
