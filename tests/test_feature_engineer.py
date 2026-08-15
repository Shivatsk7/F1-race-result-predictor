import unittest

import pandas as pd

from src.feature_engineer import engineer_features, select_available_features


def build_sample_records() -> pd.DataFrame:
    rows = []
    for round_number, circuit in enumerate(["Bahrain", "Jeddah", "Melbourne", "Suzuka"], start=1):
        rows.extend(
            [
                {
                    "year": 2024,
                    "round": round_number,
                    "race_name": f"{circuit} Grand Prix",
                    "circuit": circuit,
                    "driver": "VER",
                    "driver_number": 1,
                    "team": "Red Bull",
                    "grid_position": 1,
                    "qualifying_gap_to_pole": 0.0,
                    "race_position": 1,
                    "status": "Finished",
                    "points": 25,
                    "weather_dry": 1,
                    "weather_wet": 0,
                    "weather_mixed": 0,
                },
                {
                    "year": 2024,
                    "round": round_number,
                    "race_name": f"{circuit} Grand Prix",
                    "circuit": circuit,
                    "driver": "PER",
                    "driver_number": 11,
                    "team": "Red Bull",
                    "grid_position": 2,
                    "qualifying_gap_to_pole": 0.2,
                    "race_position": 3,
                    "status": "+1 Lap",
                    "points": 15,
                    "weather_dry": 1,
                    "weather_wet": 0,
                    "weather_mixed": 0,
                },
                {
                    "year": 2024,
                    "round": round_number,
                    "race_name": f"{circuit} Grand Prix",
                    "circuit": circuit,
                    "driver": "NOR",
                    "driver_number": 4,
                    "team": "McLaren",
                    "grid_position": 3,
                    "qualifying_gap_to_pole": 0.3,
                    "race_position": 2,
                    "status": "Finished",
                    "points": 18,
                    "weather_dry": 1,
                    "weather_wet": 0,
                    "weather_mixed": 0,
                },
                {
                    "year": 2024,
                    "round": round_number,
                    "race_name": f"{circuit} Grand Prix",
                    "circuit": circuit,
                    "driver": "PIA",
                    "driver_number": 81,
                    "team": "McLaren",
                    "grid_position": 4,
                    "qualifying_gap_to_pole": 0.45,
                    "race_position": 4,
                    "status": "Accident",
                    "points": 12,
                    "weather_dry": 1,
                    "weather_wet": 0,
                    "weather_mixed": 0,
                },
            ]
        )
    return pd.DataFrame(rows)


class FeatureEngineeringTests(unittest.TestCase):
    def test_keep_unlabeled_rows_for_inference(self):
        historical = build_sample_records()
        historical["_is_prediction_row"] = False

        prediction_row = historical.iloc[[0]].copy()
        prediction_row["round"] = 5
        prediction_row["race_name"] = "Miami Grand Prix"
        prediction_row["circuit"] = "Miami"
        prediction_row["race_position"] = None
        prediction_row["status"] = None
        prediction_row["points"] = None
        prediction_row["_is_prediction_row"] = True

        engineered = engineer_features(
            pd.concat([historical, prediction_row], ignore_index=True),
            keep_unlabeled=True,
            save_processed=False,
        )

        pred_rows = engineered[engineered["_is_prediction_row"]]
        self.assertEqual(len(pred_rows), 1)
        self.assertFalse(pred_rows[select_available_features(engineered)].isna().any().any())

    def test_track_weighting_is_higher_for_monaco_than_bahrain(self):
        records = build_sample_records()
        monaco_rows = records.iloc[:4].copy()
        monaco_rows["round"] = 5
        monaco_rows["race_name"] = "Monaco Grand Prix"
        monaco_rows["circuit"] = "Circuit de Monaco"

        engineered = engineer_features(
            pd.concat([records, monaco_rows], ignore_index=True),
            save_processed=False,
        )

        bahrain_weight = engineered[engineered["circuit"] == "Bahrain"]["track_grid_weight"].iloc[0]
        monaco_weight = engineered[engineered["circuit"] == "Circuit de Monaco"]["track_grid_weight"].iloc[0]
        self.assertGreater(monaco_weight, bahrain_weight)

    def test_madrid_metadata_resolves_without_history(self):
        madrid = pd.DataFrame(
            [
                {
                    "year": 2026,
                    "round": 1,
                    "race_name": "Madrid Grand Prix",
                    "circuit": "Madrid",
                    "driver": "VER",
                    "driver_number": 1,
                    "team": "Red Bull",
                    "grid_position": 1,
                    "qualifying_gap_to_pole": 0.0,
                    "race_position": None,
                    "status": None,
                    "points": None,
                    "weather_dry": 1,
                    "weather_wet": 0,
                    "weather_mixed": 0,
                }
            ]
        )

        engineered = engineer_features(madrid, keep_unlabeled=True, save_processed=False)
        row = engineered.iloc[0]
        self.assertEqual(row["track_archetype"], "street")
        self.assertEqual(row["benchmark_group"], "balanced")
        self.assertTrue(row[select_available_features(engineered)].notna().all())


if __name__ == "__main__":
    unittest.main()
