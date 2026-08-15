import unittest

from src.feature_engineer import engineer_features, prepare_training_data
from tests.test_feature_engineer import build_sample_records

try:
    from src.model import RacePredictor
except ModuleNotFoundError:
    RacePredictor = None


@unittest.skipIf(RacePredictor is None, "xgboost is not installed in this Python environment")
class ModelTests(unittest.TestCase):
    def test_training_uses_latest_races_as_holdout(self):
        raw_df = engineer_features(build_sample_records(), save_processed=False)
        X, y, _ = prepare_training_data(raw_df)

        predictor = RacePredictor(
            ranker_params={
                "n_estimators": 25,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "objective": "rank:pairwise",
                "random_state": 42,
                "n_jobs": 1,
            },
            regressor_params={
                "n_estimators": 25,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": 1,
            },
        )
        metrics = predictor.train(
            X,
            y,
            race_keys=raw_df[["year", "round", "benchmark_group", "benchmark_group_code"]],
        )

        train_races = predictor.evaluation_data["train_races"]
        test_races = predictor.evaluation_data["test_races"]

        self.assertTrue(set(train_races).isdisjoint(test_races))
        self.assertLess(max(train_races), min(test_races))
        self.assertIn("top_3_accuracy", metrics)
        self.assertIn("spearman_mean", metrics)

    def test_predict_with_confidence_returns_grouped_positions(self):
        raw_df = engineer_features(build_sample_records(), save_processed=False)
        X, y, _ = prepare_training_data(raw_df)

        predictor = RacePredictor(
            ranker_params={
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "objective": "rank:pairwise",
                "random_state": 42,
                "n_jobs": 1,
            },
            regressor_params={
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "objective": "reg:squarederror",
                "random_state": 42,
                "n_jobs": 1,
            },
        )
        predictor.train(
            X,
            y,
            race_keys=raw_df[["year", "round", "benchmark_group", "benchmark_group_code"]],
        )

        race_slice = raw_df[raw_df["round"] == 4].reset_index(drop=True)
        prediction_frame = predictor.predict_with_confidence(
            race_slice,
            race_keys=race_slice[["year", "round", "benchmark_group", "benchmark_group_code"]],
        )

        self.assertEqual(sorted(prediction_frame["predicted_position"].tolist()), [1, 2, 3, 4])
        self.assertTrue((prediction_frame["predicted_position_upper"] >= prediction_frame["predicted_position_lower"]).all())


if __name__ == "__main__":
    unittest.main()
