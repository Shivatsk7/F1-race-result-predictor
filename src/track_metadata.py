"""Track metadata helpers for circuit-aware modeling."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from src.config import TRACK_METADATA_PATH

TRACK_NUMERIC_COLUMNS = [
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
]

TRACK_CATEGORICAL_COLUMNS = [
    "circuit_key",
    "track_archetype",
    "benchmark_group",
    "circuit_type",
]

TRACK_ALL_COLUMNS = TRACK_CATEGORICAL_COLUMNS + TRACK_NUMERIC_COLUMNS

TRACK_ARCHETYPE_CODES = {
    "high_grid": 0,
    "balanced": 1,
    "power": 2,
    "chaos": 3,
    "street": 4,
    "unknown": 5,
}

BENCHMARK_GROUP_CODES = {
    "high_grid": 0,
    "balanced": 1,
    "power": 2,
    "chaos": 3,
    "unknown": 4,
}


def normalize_track_name(value: object) -> str:
    """Normalize circuit names for metadata joins."""
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_track_metadata() -> pd.DataFrame:
    """Load and normalize the track metadata table."""
    metadata = pd.read_csv(TRACK_METADATA_PATH)
    metadata["alias_norm"] = metadata["alias"].map(normalize_track_name)
    return metadata.drop_duplicates("alias_norm", keep="first").reset_index(drop=True)


def resolve_track_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach circuit-aware metadata, preserving any user-supplied overrides."""
    metadata = load_track_metadata()
    result = df.copy()

    circuit_text = result.get("circuit", pd.Series("", index=result.index)).fillna("")
    race_text = result.get("race_name", pd.Series("", index=result.index)).fillna("")
    inferred_lookup = circuit_text.where(circuit_text != "", race_text).map(normalize_track_name)
    if "circuit_lookup" not in result.columns:
        result["circuit_lookup"] = inferred_lookup
    else:
        result["circuit_lookup"] = result["circuit_lookup"].fillna("").astype(str)
        result.loc[result["circuit_lookup"].str.strip() == "", "circuit_lookup"] = inferred_lookup

    metadata_for_join = metadata.drop(columns=["alias"]).rename(
        columns={column: f"{column}_metadata" for column in TRACK_ALL_COLUMNS}
    )
    merged = result.merge(
        metadata_for_join,
        left_on="circuit_lookup",
        right_on="alias_norm",
        how="left",
    )

    for column in TRACK_ALL_COLUMNS:
        metadata_column = f"{column}_metadata"
        if column in result.columns:
            result[column] = result[column].combine_first(merged[metadata_column])
        else:
            result[column] = merged[metadata_column]

    result = _fill_track_defaults(result)
    result["track_archetype_code"] = result["track_archetype"].map(TRACK_ARCHETYPE_CODES).fillna(
        TRACK_ARCHETYPE_CODES["unknown"]
    )
    result["benchmark_group_code"] = result["benchmark_group"].map(BENCHMARK_GROUP_CODES).fillna(
        BENCHMARK_GROUP_CODES["unknown"]
    )
    return result


def _fill_track_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing metadata with medians or unknown categories."""
    metadata = load_track_metadata()

    for column in TRACK_NUMERIC_COLUMNS:
        default_value = float(metadata[column].median())
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default_value)

    defaults = {
        "circuit_key": "unknown",
        "track_archetype": "unknown",
        "benchmark_group": "unknown",
        "circuit_type": "unknown",
    }
    for column, default_value in defaults.items():
        df[column] = df[column].fillna(default_value)

    return df
