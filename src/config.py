"""Configuration for the Race Result Predictor."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

# Create directories
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REFERENCE_DIR, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data collection settings
HISTORICAL_YEARS_START = 2018
HISTORICAL_YEARS_END = 2024
CACHE_DIR = DATA_DIR / "fastf1_cache"
TRACK_METADATA_PATH = REFERENCE_DIR / "track_metadata.csv"

# Feature settings
FEATURES = [
    # Qualifying / starting position
    "grid_position",
    "qualifying_gap_to_pole",
    "grid_position_sq",
    "quali_vs_teammate",
    "track_grid_weight",
    "weighted_grid_position",
    "weighted_quali_gap",
    "street_grid_pressure",
    "grid_track_interaction",
    "overqualified_by_grid",
    "underqualified_by_grid",
    # Driver form
    "driver_elo_race",
    "driver_elo_quali",
    "driver_win_rate_season",
    "driver_podium_rate_season",
    "driver_points_per_race_season",
    "driver_recent_finish_avg",
    "driver_recent_finish_last5",
    "driver_recent_grid_avg",
    "driver_recent_points_last3",
    "driver_recent_gain_avg",
    # Constructor form
    "team_performance_index",
    "constructor_points_per_race_season",
    "constructor_reliability_season",
    "constructor_recent_finish_avg",
    # Circuit history
    "driver_avg_finish_at_circuit",
    "driver_avg_grid_at_circuit",
    "driver_best_finish_at_circuit",
    "constructor_avg_finish_at_circuit",
    "strength_rank_prior",
    "track_archetype_code",
    "benchmark_group_code",
    "is_street",
    "is_temporary",
    "is_permanent",
    "is_hybrid",
    "overtaking_difficulty",
    "track_position_importance",
    "safety_car_risk",
    "pit_delta_index",
    "tire_deg_index",
    "avg_speed_index",
    "high_downforce_bias",
    "traction_bias",
    "braking_severity",
    "surface_evolution_rate",
    "weather_volatility",
    # Contextual
    "weather_dry",
    "weather_wet",
    "weather_mixed",
]

TARGET = "race_position"

# Model settings
MODEL_TYPE = "xgboost-ranker"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5

# XGBoost ranker hyperparameters
XGBOOST_RANKER_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "objective": "rank:pairwise",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Auxiliary position regressor for interval calibration
XGBOOST_REGRESSOR_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 2,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
