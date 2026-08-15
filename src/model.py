"""Track-aware ranking model for F1 race result prediction."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    CROSS_VAL_FOLDS,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    XGBOOST_RANKER_PARAMS,
    XGBOOST_REGRESSOR_PARAMS,
)

logger = logging.getLogger(__name__)


class RacePredictor:
    """Track-aware ranker with calibrated finish-position intervals."""

    def __init__(
        self,
        ranker_params: dict | None = None,
        regressor_params: dict | None = None,
    ):
        self.ranker_params = ranker_params or XGBOOST_RANKER_PARAMS
        self.regressor_params = regressor_params or XGBOOST_REGRESSOR_PARAMS
        self.ranker: xgb.XGBRanker | None = None
        self.position_model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
        self.is_trained = False
        self.interval_profiles: dict[str, dict[str, float]] = {}
        self.global_interval_profile: dict[str, float] = {"q10": -2.0, "q90": 2.0}
        self.cv_mae_mean: float | None = None
        self.evaluation_data: dict | None = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        race_keys: pd.DataFrame | None = None,
    ) -> dict:
        """Train the ranker/regressor stack with chronological validation."""
        logger.info("Training model on %s samples with %s features...", len(X), X.shape[1])

        X, y, race_keys = self._align_training_inputs(X, y, race_keys)
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test, train_keys, test_keys = self._time_based_split(X, y, race_keys)
        train_groups = self._group_sizes(train_keys)

        self.ranker = xgb.XGBRanker(**self.ranker_params)
        self.ranker.fit(
            X_train,
            self._rank_targets(y_train, train_keys),
            group=train_groups,
            verbose=False,
        )

        self.position_model = xgb.XGBRegressor(**self.regressor_params)
        self.position_model.fit(X_train, y_train, verbose=False)

        prediction_frame = self._predict_frame(X_test, test_keys)
        metrics = self._calculate_metrics(y_test, prediction_frame, test_keys)

        cv_scores = self._time_series_cv_scores(X, y, race_keys)
        if cv_scores:
            metrics["cv_mae_mean"] = float(np.mean(cv_scores))
            metrics["cv_mae_std"] = float(np.std(cv_scores))
            self.cv_mae_mean = metrics["cv_mae_mean"]
        else:
            self.cv_mae_mean = metrics["mae"]

        self._fit_interval_profiles(y_test, prediction_frame["predicted_position_raw"], test_keys)

        self.evaluation_data = {
            "X_test": X_test.copy(),
            "y_test": y_test.copy(),
            "prediction_frame": prediction_frame.copy(),
            "test_keys": test_keys.copy(),
            "train_races": self._race_id_list(train_keys),
            "test_races": self._race_id_list(test_keys),
            "benchmark_metrics": metrics.get("benchmark_metrics", {}),
        }

        self.is_trained = True
        logger.info(
            "Training complete. MAE: %.3f, Spearman: %.3f, Top-3: %.3f",
            metrics["mae"],
            metrics["spearman_mean"],
            metrics["top_3_accuracy"],
        )
        return metrics

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return expected finishing positions from the auxiliary regressor."""
        if self.position_model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.position_model.predict(X[self.feature_names])

    def predict(self, X: pd.DataFrame, race_keys: pd.DataFrame | None = None) -> np.ndarray:
        """Predict race finishing positions using grouped ranking."""
        prediction_frame = self._predict_frame(X, self._infer_race_keys(X, race_keys))
        return prediction_frame["predicted_position"].to_numpy()

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        race_keys: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Predict positions with track-calibrated prediction intervals."""
        keys = self._infer_race_keys(X, race_keys)
        prediction_frame = self._predict_frame(X, keys)
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        for idx, row in prediction_frame.iterrows():
            benchmark_code = str(int(keys.iloc[idx]["benchmark_group_code"])) if "benchmark_group_code" in keys.columns else "global"
            profile = self.interval_profiles.get(benchmark_code, self.global_interval_profile)
            raw_prediction = row["predicted_position_raw"]
            lower_bounds.append(np.clip(raw_prediction + profile["q10"], 1, 20))
            upper_bounds.append(np.clip(raw_prediction + profile["q90"], 1, 20))

        prediction_frame["predicted_position_lower"] = np.round(lower_bounds, 1)
        prediction_frame["predicted_position_upper"] = np.round(upper_bounds, 1)
        return prediction_frame[
            [
                "predicted_position",
                "predicted_position_raw",
                "predicted_position_lower",
                "predicted_position_upper",
            ]
        ]

    def get_feature_importance(self) -> pd.DataFrame:
        """Return ranker feature importance."""
        if not self.is_trained or self.ranker is None:
            raise RuntimeError("Model must be trained first")

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.ranker.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, name: str = "race_predictor") -> Path:
        """Save the trained ranker, regressor, and metadata."""
        if not self.is_trained or self.ranker is None or self.position_model is None:
            raise RuntimeError("Model must be trained first")

        ranker_path = MODELS_DIR / f"{name}.joblib"
        regressor_path = MODELS_DIR / f"{name}_position_model.joblib"
        meta_path = MODELS_DIR / f"{name}_meta.json"

        joblib.dump(self.ranker, ranker_path)
        joblib.dump(self.position_model, regressor_path)

        meta = {
            "feature_names": self.feature_names,
            "ranker_params": self.ranker_params,
            "regressor_params": self.regressor_params,
            "is_trained": self.is_trained,
            "cv_mae_mean": self.cv_mae_mean,
            "interval_profiles": self.interval_profiles,
            "global_interval_profile": self.global_interval_profile,
            "random_state": RANDOM_STATE,
            "model_type": "xgboost_ranker_regressor",
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        logger.info("Model saved to %s", ranker_path)
        return ranker_path

    def load(self, name: str = "race_predictor") -> "RacePredictor":
        """Load the trained ranker, regressor, and metadata."""
        ranker_path = MODELS_DIR / f"{name}.joblib"
        regressor_path = MODELS_DIR / f"{name}_position_model.joblib"
        meta_path = MODELS_DIR / f"{name}_meta.json"

        self.ranker = joblib.load(ranker_path)
        self.position_model = joblib.load(regressor_path)

        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

        self.feature_names = meta["feature_names"]
        self.ranker_params = meta["ranker_params"]
        self.regressor_params = meta["regressor_params"]
        self.is_trained = meta["is_trained"]
        self.cv_mae_mean = meta.get("cv_mae_mean")
        self.interval_profiles = meta.get("interval_profiles", {})
        self.global_interval_profile = meta.get("global_interval_profile", {"q10": -2.0, "q90": 2.0})

        logger.info("Model loaded from %s", ranker_path)
        return self

    def _predict_frame(self, X: pd.DataFrame, race_keys: pd.DataFrame) -> pd.DataFrame:
        """Create grouped ranking predictions for one or more races."""
        if self.ranker is None or self.position_model is None:
            raise RuntimeError("Model must be trained before prediction")

        feature_frame = X[self.feature_names].reset_index(drop=True)
        race_keys = race_keys.reset_index(drop=True)

        rank_scores = self.ranker.predict(feature_frame)
        expected_finish = self.position_model.predict(feature_frame)
        grouped_positions = np.zeros(len(feature_frame), dtype=int)

        for _, row_indices in self._race_groups_from_keys(race_keys).items():
            sort_frame = pd.DataFrame(
                {
                    "row_idx": row_indices,
                    "rank_score": rank_scores[row_indices],
                    "expected_finish": expected_finish[row_indices],
                }
            ).sort_values(
                ["rank_score", "expected_finish"],
                ascending=[False, True],
                ignore_index=True,
            )
            sort_frame["predicted_position"] = np.arange(1, len(sort_frame) + 1)
            grouped_positions[sort_frame["row_idx"]] = sort_frame["predicted_position"].to_numpy()

        return pd.DataFrame(
            {
                "predicted_position": grouped_positions,
                "predicted_position_raw": np.clip(expected_finish, 1, 20),
                "rank_score": rank_scores,
            }
        )

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        prediction_frame: pd.DataFrame,
        race_keys: pd.DataFrame,
    ) -> dict:
        """Calculate ranking-aware metrics and benchmark slices."""
        predicted_positions = prediction_frame["predicted_position"].to_numpy()
        expected_positions = prediction_frame["predicted_position_raw"].to_numpy()
        y_values = y_true.to_numpy()

        metrics = {
            "mae": mean_absolute_error(y_values, predicted_positions),
            "rmse": math.sqrt(mean_squared_error(y_values, predicted_positions)),
            "expected_finish_mae": mean_absolute_error(y_values, expected_positions),
            "exact_accuracy": (y_values == predicted_positions).mean(),
            "within_1_accuracy": (np.abs(y_values - predicted_positions) <= 1).mean(),
            "within_2_accuracy": (np.abs(y_values - predicted_positions) <= 2).mean(),
            "within_3_accuracy": (np.abs(y_values - predicted_positions) <= 3).mean(),
            "podium_accuracy": (
                ((y_values <= 3) & (predicted_positions <= 3))
                | ((y_values > 3) & (predicted_positions > 3))
            ).mean(),
        }

        top3_scores: list[float] = []
        top5_scores: list[float] = []
        spearman_scores: list[float] = []
        benchmark_rows: list[dict[str, float | str]] = []

        grouped_keys = self._race_groups_from_keys(race_keys)
        for race_id, row_indices in grouped_keys.items():
            actual_race = pd.Series(y_values[row_indices])
            predicted_race = pd.Series(predicted_positions[row_indices])
            benchmark_group = race_keys.iloc[row_indices]["benchmark_group"].iloc[0] if "benchmark_group" in race_keys.columns else "unknown"

            top3 = _top_n_overlap(actual_race, predicted_race, n=3)
            top5 = _top_n_overlap(actual_race, predicted_race, n=5)
            spearman = actual_race.corr(predicted_race, method="spearman")
            spearman = 0.0 if pd.isna(spearman) else float(spearman)

            top3_scores.append(top3)
            top5_scores.append(top5)
            spearman_scores.append(spearman)
            benchmark_rows.append(
                {
                    "race_id": f"{race_id[0]}-{race_id[1]}",
                    "benchmark_group": benchmark_group,
                    "mae": float(mean_absolute_error(actual_race, predicted_race)),
                    "top_3_accuracy": top3,
                    "top_5_accuracy": top5,
                    "spearman": spearman,
                }
            )

        metrics["top_3_accuracy"] = float(np.mean(top3_scores)) if top3_scores else 0.0
        metrics["top_5_accuracy"] = float(np.mean(top5_scores)) if top5_scores else 0.0
        metrics["spearman_mean"] = float(np.mean(spearman_scores)) if spearman_scores else 0.0
        metrics["benchmark_metrics"] = _summarize_benchmarks(pd.DataFrame(benchmark_rows))
        return metrics

    def _fit_interval_profiles(
        self,
        y_true: pd.Series,
        expected_positions: pd.Series | np.ndarray,
        race_keys: pd.DataFrame,
    ) -> None:
        """Fit conformal-style residual bands by benchmark group."""
        residual_frame = race_keys.copy().reset_index(drop=True)
        residual_frame["residual"] = y_true.reset_index(drop=True) - pd.Series(expected_positions).reset_index(drop=True)

        self.global_interval_profile = {
            "q10": float(residual_frame["residual"].quantile(0.10)),
            "q90": float(residual_frame["residual"].quantile(0.90)),
        }
        self.interval_profiles = {}

        if "benchmark_group_code" not in residual_frame.columns:
            return

        for benchmark_code, group in residual_frame.groupby("benchmark_group_code"):
            if len(group) < 8:
                continue
            self.interval_profiles[str(int(benchmark_code))] = {
                "q10": float(group["residual"].quantile(0.10)),
                "q90": float(group["residual"].quantile(0.90)),
            }

    def _time_series_cv_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        race_keys: pd.DataFrame,
    ) -> list[float]:
        """Evaluate with expanding-window CV over race weekends."""
        unique_races = self._race_id_list(race_keys)
        if len(unique_races) < 3:
            return []

        splitter = TimeSeriesSplit(n_splits=min(CROSS_VAL_FOLDS, len(unique_races) - 1))
        row_races = list(map(tuple, race_keys[["year", "round"]].to_numpy()))
        scores: list[float] = []

        for train_indices, test_indices in splitter.split(unique_races):
            train_set = {unique_races[index] for index in train_indices}
            test_set = {unique_races[index] for index in test_indices}
            train_mask = np.array([race in train_set for race in row_races], dtype=bool)
            test_mask = np.array([race in test_set for race in row_races], dtype=bool)

            train_keys = race_keys.loc[train_mask].reset_index(drop=True)
            test_keys = race_keys.loc[test_mask].reset_index(drop=True)
            train_X = X.loc[train_mask].reset_index(drop=True)
            test_X = X.loc[test_mask].reset_index(drop=True)
            train_y = y.loc[train_mask].reset_index(drop=True)
            test_y = y.loc[test_mask].reset_index(drop=True)

            ranker = xgb.XGBRanker(**self.ranker_params)
            ranker.fit(
                train_X,
                self._rank_targets(train_y, train_keys),
                group=self._group_sizes(train_keys),
                verbose=False,
            )

            regressor = xgb.XGBRegressor(**self.regressor_params)
            regressor.fit(train_X, train_y, verbose=False)

            temp_predictor = RacePredictor(self.ranker_params, self.regressor_params)
            temp_predictor.ranker = ranker
            temp_predictor.position_model = regressor
            temp_predictor.feature_names = self.feature_names or X.columns.tolist()
            prediction_frame = temp_predictor._predict_frame(test_X, test_keys)
            scores.append(mean_absolute_error(test_y, prediction_frame["predicted_position"]))

        return scores

    def _time_based_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        race_keys: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """Split by whole race weekends, reserving the latest races for holdout."""
        unique_races = self._race_id_list(race_keys)
        if len(unique_races) < 2:
            raise ValueError("Need at least two race weekends for chronological evaluation.")

        n_test_races = max(1, math.ceil(len(unique_races) * TEST_SIZE))
        n_test_races = min(n_test_races, len(unique_races) - 1)
        train_races = set(unique_races[:-n_test_races])
        test_races = set(unique_races[-n_test_races:])
        row_races = list(map(tuple, race_keys[["year", "round"]].to_numpy()))

        train_mask = np.array([race in train_races for race in row_races], dtype=bool)
        test_mask = np.array([race in test_races for race in row_races], dtype=bool)

        return (
            X.loc[train_mask].reset_index(drop=True),
            X.loc[test_mask].reset_index(drop=True),
            y.loc[train_mask].reset_index(drop=True),
            y.loc[test_mask].reset_index(drop=True),
            race_keys.loc[train_mask].reset_index(drop=True),
            race_keys.loc[test_mask].reset_index(drop=True),
        )

    @staticmethod
    def _align_training_inputs(
        X: pd.DataFrame,
        y: pd.Series,
        race_keys: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Align, sort, and validate training inputs."""
        X = X.reset_index(drop=True).copy()
        y = y.reset_index(drop=True).copy()

        if race_keys is None:
            race_keys = pd.DataFrame({"year": np.zeros(len(X), dtype=int), "round": np.arange(len(X))})
        else:
            race_keys = race_keys.reset_index(drop=True).copy()

        order = race_keys.sort_values(["year", "round"]).index.to_numpy()
        return (
            X.iloc[order].reset_index(drop=True),
            y.iloc[order].reset_index(drop=True),
            race_keys.iloc[order].reset_index(drop=True),
        )

    @staticmethod
    def _group_sizes(race_keys: pd.DataFrame) -> list[int]:
        """Return XGBoost ranker group sizes for each race weekend."""
        return [
            len(group)
            for _, group in race_keys.groupby(["year", "round"], sort=False)
        ]

    @staticmethod
    def _race_groups_from_keys(race_keys: pd.DataFrame) -> dict[tuple[int, int], list[int]]:
        """Return row indices grouped by race."""
        groups: dict[tuple[int, int], list[int]] = {}
        for idx, row in race_keys.reset_index(drop=True).iterrows():
            groups.setdefault((int(row["year"]), int(row["round"])), []).append(idx)
        return groups

    @staticmethod
    def _race_id_list(race_keys: pd.DataFrame) -> list[tuple[int, int]]:
        """Return sorted unique race IDs."""
        return list(
            race_keys[["year", "round"]]
            .drop_duplicates()
            .sort_values(["year", "round"])
            .itertuples(index=False, name=None)
        )

    @staticmethod
    def _rank_targets(y: pd.Series, race_keys: pd.DataFrame) -> np.ndarray:
        """Convert finishing positions into ranking labels."""
        labels = np.zeros(len(y), dtype=float)
        for _, row_indices in RacePredictor._race_groups_from_keys(race_keys).items():
            actual = y.iloc[row_indices].to_numpy()
            field_size = len(row_indices)
            labels[row_indices] = field_size + 1 - actual
        return labels

    @staticmethod
    def _infer_race_keys(X: pd.DataFrame, race_keys: pd.DataFrame | None) -> pd.DataFrame:
        """Infer race grouping information from the feature frame when possible."""
        if race_keys is not None:
            return race_keys[["year", "round", *[col for col in ["benchmark_group", "benchmark_group_code"] if col in race_keys.columns]]].copy()

        keys = pd.DataFrame(index=X.index)
        keys["year"] = X["year"] if "year" in X.columns else 0
        keys["round"] = X["round"] if "round" in X.columns else 1
        if "benchmark_group" in X.columns:
            keys["benchmark_group"] = X["benchmark_group"]
        if "benchmark_group_code" in X.columns:
            keys["benchmark_group_code"] = X["benchmark_group_code"]
        return keys.reset_index(drop=True)


def _top_n_overlap(actual_positions: pd.Series, predicted_positions: pd.Series, *, n: int) -> float:
    """Compute overlap accuracy for the top-N finishers."""
    actual_top = set(actual_positions.nsmallest(min(n, len(actual_positions))).index)
    predicted_top = set(predicted_positions.nsmallest(min(n, len(predicted_positions))).index)
    return len(actual_top & predicted_top) / max(1, min(n, len(actual_positions)))


def _summarize_benchmarks(benchmark_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Aggregate evaluation metrics by benchmark group."""
    if benchmark_frame.empty:
        return {}

    summary = (
        benchmark_frame.groupby("benchmark_group")[["mae", "top_3_accuracy", "top_5_accuracy", "spearman"]]
        .mean()
        .reset_index()
    )
    return {
        row["benchmark_group"]: {
            "mae": float(row["mae"]),
            "top_3_accuracy": float(row["top_3_accuracy"]),
            "top_5_accuracy": float(row["top_5_accuracy"]),
            "spearman": float(row["spearman"]),
        }
        for _, row in summary.iterrows()
    }


def plot_feature_importance(predictor: RacePredictor, top_n: int = 15) -> Path:
    """Plot and save feature importance chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    importance_df = predictor.get_feature_importance().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=importance_df,
        y="feature",
        x="importance",
        ax=ax,
        hue="feature",
        palette="viridis",
        legend=False,
    )
    ax.set_title("Track-Aware Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")

    plt.tight_layout()
    path = PLOTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Feature importance plot saved to %s", path)
    return path


def plot_predictions_vs_actual(predictor: RacePredictor, X: pd.DataFrame, y: pd.Series, race_keys: pd.DataFrame | None = None) -> Path:
    """Plot predicted vs actual race positions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prediction_frame = predictor.predict_with_confidence(X, race_keys=race_keys)
    predictions = prediction_frame["predicted_position"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y, predictions, alpha=0.4, s=24, color="#e10600")
    ax.plot([0, 20], [0, 20], "k--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Actual Position", fontsize=12)
    ax.set_ylabel("Predicted Position", fontsize=12)
    ax.set_title("Predicted vs Actual Race Positions", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.set_xticks(range(1, 21))
    ax.set_yticks(range(1, 21))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "predictions_vs_actual.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Predictions plot saved to %s", path)
    return path


def plot_position_distribution(predictor: RacePredictor, X: pd.DataFrame, y: pd.Series, race_keys: pd.DataFrame | None = None) -> Path:
    """Plot distribution of prediction errors by actual finishing position."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prediction_frame = predictor.predict_with_confidence(X, race_keys=race_keys)
    predictions = prediction_frame["predicted_position"]
    errors = np.abs(y.to_numpy() - predictions.to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(errors, bins=range(0, 20), color="#e10600", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Absolute Error (positions)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Prediction Errors", fontweight="bold")
    axes[0].axvline(np.mean(errors), color="black", linestyle="--", label=f"Mean: {np.mean(errors):.2f}")
    axes[0].legend()

    error_by_pos = pd.DataFrame({"actual": y.to_numpy(), "error": errors})
    avg_error = error_by_pos.groupby("actual")["error"].mean()
    axes[1].bar(avg_error.index, avg_error.values, color="#e10600", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Actual Position")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("Prediction Error by Actual Position", fontweight="bold")

    plt.tight_layout()
    path = PLOTS_DIR / "error_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Error analysis plot saved to %s", path)
    return path
