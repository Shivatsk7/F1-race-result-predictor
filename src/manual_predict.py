"""Prediction helpers for manually supplied starting grids."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data_collector import load_cached_data
from src.feature_engineer import engineer_features
from src.model import RacePredictor
from src.race_simulator import run_monte_carlo

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "year",
    "round",
    "race_name",
    "circuit",
    "driver",
    "driver_number",
    "team",
    "grid_position",
]


def predict_from_grid_csv(
    input_path: str | Path,
    *,
    model_name: str = "race_predictor",
    historical_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict a race from a manually prepared starting-grid CSV."""
    input_path = Path(input_path)
    grid_df = pd.read_csv(input_path)
    return predict_from_grid_dataframe(grid_df, model_name=model_name, historical_df=historical_df)


def predict_from_grid_dataframe(
    grid_df: pd.DataFrame,
    *,
    model_name: str = "race_predictor",
    historical_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict a race from a dataframe describing the starting grid."""
    results = _build_prediction_context(
        grid_df,
        model_name=model_name,
        historical_df=historical_df,
    )

    results = results.sort_values(
        ["predicted_position", "predicted_position_raw", "grid_position"],
        na_position="last",
    ).reset_index(drop=True)
    results.index = results.index + 1
    results.index.name = "predicted_rank"
    return results[_deterministic_output_columns(results)]


def simulate_from_grid_csv(
    input_path: str | Path,
    *,
    model_name: str = "race_predictor",
    historical_df: pd.DataFrame | None = None,
    n_simulations: int = 1000,
    random_state: int = 42,
    enable_overtaking: bool = True,
    enable_strategy: bool = True,
) -> pd.DataFrame:
    """Run Monte Carlo race simulations from a manually prepared grid CSV."""
    input_path = Path(input_path)
    grid_df = pd.read_csv(input_path)
    return simulate_from_grid_dataframe(
        grid_df,
        model_name=model_name,
        historical_df=historical_df,
        n_simulations=n_simulations,
        random_state=random_state,
        enable_overtaking=enable_overtaking,
        enable_strategy=enable_strategy,
    )


def simulate_from_grid_dataframe(
    grid_df: pd.DataFrame,
    *,
    model_name: str = "race_predictor",
    historical_df: pd.DataFrame | None = None,
    n_simulations: int = 1000,
    random_state: int = 42,
    enable_overtaking: bool = True,
    enable_strategy: bool = True,
) -> pd.DataFrame:
    """Simulate a race from a dataframe describing the starting grid."""
    prediction_context = _build_prediction_context(
        grid_df,
        model_name=model_name,
        historical_df=historical_df,
    )
    simulation_results = run_monte_carlo(
        prediction_context,
        n_simulations=n_simulations,
        random_state=random_state,
        enable_overtaking=enable_overtaking,
        enable_strategy=enable_strategy,
    )
    return simulation_results


def _prepare_prediction_rows(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize manual grid input into model-ready race rows."""
    df = grid_df.copy()

    if "qualifying_gap_to_pole" not in df.columns:
        if "qualifying_time" not in df.columns:
            raise ValueError(
                "Input must include either 'qualifying_gap_to_pole' or 'qualifying_time'."
            )
        df["qualifying_gap_to_pole"] = _compute_gap_to_pole(df["qualifying_time"])

    for weather_column, default_value in {
        "weather_dry": 1,
        "weather_wet": 0,
        "weather_mixed": 0,
    }.items():
        if weather_column not in df.columns:
            df[weather_column] = default_value

    df["race_position"] = None
    df["status"] = None
    df["points"] = None
    return df


def _build_prediction_context(
    grid_df: pd.DataFrame,
    *,
    model_name: str,
    historical_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build deterministic predictions enriched with simulator features."""
    _validate_grid_dataframe(grid_df)

    predictor = RacePredictor().load(model_name)
    prediction_rows = _prepare_prediction_rows(grid_df)

    target_year = int(prediction_rows["year"].iloc[0])
    target_round = int(prediction_rows["round"].iloc[0])

    if historical_df is None:
        historical_df = load_cached_data()

    historical_df = historical_df[
        (historical_df["year"] < target_year)
        | ((historical_df["year"] == target_year) & (historical_df["round"] < target_round))
    ].copy()
    historical_df["_is_prediction_row"] = False

    prediction_rows["_is_prediction_row"] = True
    combined = pd.concat([historical_df, prediction_rows], ignore_index=True, sort=False)
    engineered = engineer_features(combined, keep_unlabeled=True, save_processed=False)

    pred_features = engineered[engineered["_is_prediction_row"]].copy()
    pred_features = pred_features.sort_values(["grid_position", "driver"], na_position="last").reset_index(drop=True)

    prediction_frame = predictor.predict_with_confidence(
        pred_features,
        race_keys=pred_features[["year", "round", "benchmark_group", "benchmark_group_code"]],
    )
    simulator_feature_columns = [
        "year",
        "round",
        "race_name",
        "circuit",
        "track_archetype",
        "benchmark_group",
        "track_grid_weight",
        "is_street",
        "overtaking_difficulty",
        "track_position_importance",
        "safety_car_risk",
        "pit_delta_index",
        "tire_deg_index",
        "high_downforce_bias",
        "weather_volatility",
        "driver_elo_race",
        "driver_recent_gain_avg",
        "team_performance_index",
        "constructor_reliability_season",
    ]
    available_simulator_columns = [
        column for column in simulator_feature_columns if column in pred_features.columns
    ]
    return pd.concat(
        [
            pred_features[["driver", "team", "grid_position"] + available_simulator_columns].reset_index(drop=True),
            prediction_frame.reset_index(drop=True),
        ],
        axis=1,
    )


def _deterministic_output_columns(results: pd.DataFrame) -> list[str]:
    """Return the stable deterministic prediction schema."""
    columns = [
        "driver",
        "team",
        "grid_position",
        "predicted_position",
        "predicted_position_raw",
        "predicted_position_lower",
        "predicted_position_upper",
    ]
    return [column for column in columns if column in results.columns]


def _compute_gap_to_pole(time_series: pd.Series) -> pd.Series:
    """Convert qualifying lap times into gaps to pole."""
    parsed_times = time_series.apply(_parse_time_to_seconds)
    pole_seconds = parsed_times.min()
    return parsed_times - pole_seconds


def _parse_time_to_seconds(value) -> float:
    """Parse a lap time like 1:29.158 into total seconds."""
    if pd.isna(value):
        raise ValueError("Qualifying times must not be empty when used to compute gap to pole.")

    text = str(value).strip()
    parts = text.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"Unsupported qualifying time format: {value}")


def _validate_grid_dataframe(grid_df: pd.DataFrame) -> None:
    """Validate that a manual grid dataframe is usable."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in grid_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if grid_df.empty:
        raise ValueError("Grid input is empty.")

    unique_targets = grid_df[["year", "round", "race_name", "circuit"]].drop_duplicates()
    if len(unique_targets) != 1:
        raise ValueError("Grid input must contain rows for exactly one race.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
