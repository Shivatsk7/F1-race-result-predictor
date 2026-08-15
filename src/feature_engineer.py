"""Feature engineering for F1 race result prediction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import FEATURES, PROCESSED_DIR
from src.track_metadata import resolve_track_metadata

logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    *,
    keep_unlabeled: bool = False,
    save_processed: bool = True,
) -> pd.DataFrame:
    """Apply track-aware feature engineering to raw race records."""
    logger.info("Engineering features from %s raw records...", len(df))

    if df.empty:
        return df.copy()

    df = df.copy()
    df = _prepare_base_columns(df)
    df = resolve_track_metadata(df)
    df = _add_driver_form_features(df)
    df = _add_constructor_form_features(df)
    df = _add_skill_ratings(df)
    df = _add_circuit_history_features(df)
    df = _add_track_weighting_features(df)
    df = _add_strength_context(df)
    df = _fill_feature_na(df)

    if not keep_unlabeled:
        df = df.dropna(subset=["race_position"]).copy()

    if save_processed and not keep_unlabeled:
        df.to_parquet(PROCESSED_DIR / "training_data.parquet", index=False)

    logger.info(
        "Feature engineering complete. %s records with %s usable features.",
        len(df),
        len(select_available_features(df)),
    )
    return df


def select_available_features(df: pd.DataFrame) -> list[str]:
    """Return the configured feature list available in a dataframe."""
    return [feature for feature in FEATURES if feature in df.columns]


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare training matrices from an engineered dataframe."""
    train_df = df.dropna(subset=["race_position"]).copy()
    available_features = select_available_features(train_df)

    X = train_df[available_features].copy()
    y = train_df["race_position"].astype(int).copy()

    logger.info("Training data shape: X=%s, y=%s", X.shape, y.shape)
    logger.info("Features used: %s", available_features)
    return X, y, available_features


def _prepare_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ordering and key numeric fields."""
    ordered = df.sort_values(["year", "round", "driver"]).reset_index(drop=True)
    for column in ["grid_position", "qualifying_gap_to_pole", "race_position", "points"]:
        if column in ordered.columns:
            ordered[column] = pd.to_numeric(ordered[column], errors="coerce")

    ordered["position_points"] = pd.to_numeric(ordered.get("points"), errors="coerce").fillna(0)
    return ordered


def _add_driver_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver season-to-date and recent-form features."""
    driver_groups = df.groupby(["year", "driver"])
    driver_history = df.groupby("driver")

    df["driver_win_rate_season"] = driver_groups["race_position"].transform(
        lambda series: _prior_rate(series, lambda values: values == 1)
    )
    df["driver_podium_rate_season"] = driver_groups["race_position"].transform(
        lambda series: _prior_rate(series, lambda values: values <= 3)
    )
    df["driver_points_per_race_season"] = driver_groups["position_points"].transform(_prior_mean)

    df["driver_recent_finish_avg"] = driver_history["race_position"].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["driver_recent_finish_last5"] = driver_history["race_position"].transform(
        lambda series: series.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df["driver_recent_grid_avg"] = driver_history["grid_position"].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["driver_recent_points_last3"] = driver_history["position_points"].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
    )

    position_gain = df["grid_position"] - df["race_position"]
    df["driver_recent_gain_avg"] = position_gain.groupby(df["driver"]).transform(
        lambda series: series.shift(1).rolling(window=5, min_periods=1).mean()
    )

    return df


def _add_constructor_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add constructor season-to-date and recent-form features."""
    team_round = (
        df.groupby(["year", "round", "team"], as_index=False)
        .agg(
            round_points=("position_points", "sum"),
            round_reliability=("status", lambda values: _finished_flag(values).mean()),
            round_finish_avg=("race_position", "mean"),
        )
        .sort_values(["year", "round", "team"])
        .reset_index(drop=True)
    )

    team_groups = team_round.groupby(["year", "team"])
    team_history = team_round.groupby("team")

    team_round["constructor_points_per_race_season"] = team_groups["round_points"].transform(_prior_mean)
    team_round["constructor_reliability_season"] = team_groups["round_reliability"].transform(_prior_mean)
    team_round["constructor_recent_finish_avg"] = team_history["round_finish_avg"].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
    )
    team_round["constructor_recent_points_last3"] = team_history["round_points"].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
    )
    team_round["team_performance_index"] = (
        0.45 * team_round["constructor_points_per_race_season"].fillna(0)
        + 0.30 * (21 - team_round["constructor_recent_finish_avg"].fillna(10))
        + 0.25 * (20 * team_round["constructor_reliability_season"].fillna(0.75))
    )

    return df.merge(
        team_round[
            [
                "year",
                "round",
                "team",
                "constructor_points_per_race_season",
                "constructor_reliability_season",
                "constructor_recent_finish_avg",
                "team_performance_index",
            ]
        ],
        on=["year", "round", "team"],
        how="left",
    )


def _add_skill_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Add Elo-style prior ratings for race execution and qualifying pace."""
    df["driver_elo_race"] = _compute_sequential_elo(
        df,
        entity_column="driver",
        outcome_column="race_position",
        default_rating=1500.0,
        k_factor=24.0,
    )
    df["driver_elo_quali"] = _compute_sequential_elo(
        df,
        entity_column="driver",
        outcome_column="grid_position",
        default_rating=1500.0,
        k_factor=20.0,
    )
    return df


def _add_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver and constructor circuit-history features."""
    df["driver_avg_finish_at_circuit"] = (
        df.groupby(["driver", "circuit_key"])["race_position"]
        .transform(lambda series: series.expanding().mean().shift(1))
    )
    df["driver_avg_grid_at_circuit"] = (
        df.groupby(["driver", "circuit_key"])["grid_position"]
        .transform(lambda series: series.expanding().mean().shift(1))
    )
    df["driver_best_finish_at_circuit"] = (
        df.groupby(["driver", "circuit_key"])["race_position"]
        .transform(lambda series: series.expanding().min().shift(1))
    )
    df["constructor_avg_finish_at_circuit"] = (
        df.groupby(["team", "circuit_key"])["race_position"]
        .transform(lambda series: series.expanding().mean().shift(1))
    )
    return df


def _add_track_weighting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add dynamic track-aware weighting for grid and qualifying performance."""
    normalized_overtaking = (df["overtaking_difficulty"] - 1) / 4
    normalized_position = (df["track_position_importance"] - 1) / 4
    normalized_pit = (df["pit_delta_index"] - 1) / 4
    normalized_deg = (df["tire_deg_index"] - 1) / 4

    track_logit = (
        -0.35
        + 1.40 * df["is_street"]
        + 1.10 * normalized_overtaking
        + 1.10 * normalized_position
        + 0.60 * normalized_pit
        - 0.50 * normalized_deg
    )
    df["track_grid_weight"] = 1 / (1 + np.exp(-track_logit))
    df["grid_position_sq"] = df["grid_position"] ** 2
    df["weighted_grid_position"] = df["track_grid_weight"] * np.log1p(df["grid_position"])
    df["weighted_quali_gap"] = df["track_grid_weight"] * df["qualifying_gap_to_pole"].fillna(0)
    df["street_grid_pressure"] = df["grid_position"] * df["is_street"]
    df["grid_track_interaction"] = df["grid_position"] * df["track_position_importance"]
    df["quali_vs_teammate"] = (
        df.groupby(["year", "round", "team"])["grid_position"]
        .transform(lambda series: series.rank(method="dense"))
    )
    return df


def _add_strength_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior strength ranks and over/under-qualification signals."""
    race_groups = ["year", "round"]

    df["driver_skill_pct"] = _group_percentile(df, race_groups, "driver_elo_race", ascending=False)
    df["driver_quali_skill_pct"] = _group_percentile(df, race_groups, "driver_elo_quali", ascending=False)
    df["driver_form_pct"] = _group_percentile(df, race_groups, "driver_recent_points_last3", ascending=False)
    df["team_strength_pct"] = _group_percentile(df, race_groups, "team_performance_index", ascending=False)

    df["prior_strength_score"] = (
        0.35 * df["driver_skill_pct"]
        + 0.20 * df["driver_quali_skill_pct"]
        + 0.20 * df["driver_form_pct"]
        + 0.25 * df["team_strength_pct"]
    )
    df["strength_rank_prior"] = df.groupby(race_groups)["prior_strength_score"].transform(
        lambda series: series.rank(method="dense", ascending=False)
    )
    df["overqualified_by_grid"] = df["strength_rank_prior"] - df["grid_position"]
    df["underqualified_by_grid"] = df["grid_position"] - df["strength_rank_prior"]
    return df


def _fill_feature_na(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing feature values using labeled-history medians."""
    fill_source = df[df["race_position"].notna()].copy()
    if fill_source.empty:
        fill_source = df

    for column in select_available_features(df):
        if pd.api.types.is_numeric_dtype(df[column]):
            median = fill_source[column].median()
            fill_value = 0 if pd.isna(median) else median
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fill_value)

    return df


def _prior_rate(series: pd.Series, predicate) -> pd.Series:
    """Return the pre-event success rate for each row in a sorted series."""
    prior_count = pd.Series(np.arange(len(series)), index=series.index, dtype=float)
    prior_success = predicate(series).astype(float).cumsum().shift(1, fill_value=0)
    return prior_success.divide(prior_count.replace(0, np.nan))


def _prior_mean(series: pd.Series) -> pd.Series:
    """Return the pre-event running mean for each row in a sorted series."""
    prior_count = pd.Series(np.arange(len(series)), index=series.index, dtype=float)
    prior_total = series.astype(float).cumsum().shift(1, fill_value=0)
    return prior_total.divide(prior_count.replace(0, np.nan))


def _finished_flag(status_series: pd.Series) -> pd.Series:
    """Classify officially finished or lapped cars as finished races."""
    normalized = status_series.fillna("").astype(str)
    return normalized.str.contains("Finished|Lap", case=False, regex=True).astype(float)


def _group_percentile(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    *,
    ascending: bool,
) -> pd.Series:
    """Return within-race percentiles where higher is always better."""
    rank_pct = df.groupby(group_columns)[value_column].transform(
        lambda series: series.rank(method="average", pct=True, ascending=ascending)
    )
    return 1 - rank_pct if ascending else rank_pct


def _compute_sequential_elo(
    df: pd.DataFrame,
    *,
    entity_column: str,
    outcome_column: str,
    default_rating: float,
    k_factor: float,
) -> pd.Series:
    """Compute prior Elo-style ratings across race weekends."""
    ratings: dict[str, float] = {}
    priors = pd.Series(index=df.index, dtype=float)

    for (_, _), group in df.groupby(["year", "round"], sort=True):
        competitors = list(group.itertuples())
        updates = {getattr(row, entity_column): 0.0 for row in competitors}

        for row in competitors:
            entity = getattr(row, entity_column)
            priors.loc[row.Index] = ratings.get(entity, default_rating)

        if outcome_column not in group.columns or group[outcome_column].notna().sum() < 2:
            continue

        for i, row_i in enumerate(competitors):
            entity_i = getattr(row_i, entity_column)
            rating_i = ratings.get(entity_i, default_rating)
            outcome_i = getattr(row_i, outcome_column)
            if pd.isna(outcome_i):
                continue

            for row_j in competitors[i + 1:]:
                entity_j = getattr(row_j, entity_column)
                rating_j = ratings.get(entity_j, default_rating)
                outcome_j = getattr(row_j, outcome_column)
                if pd.isna(outcome_j):
                    continue

                expected_i = 1.0 / (1.0 + 10 ** ((rating_j - rating_i) / 400.0))
                if outcome_i < outcome_j:
                    actual_i = 1.0
                elif outcome_i > outcome_j:
                    actual_i = 0.0
                else:
                    actual_i = 0.5

                delta = k_factor * (actual_i - expected_i)
                updates[entity_i] += delta
                updates[entity_j] -= delta

        normalization = max(1, len(competitors) - 1)
        for entity, delta in updates.items():
            ratings[entity] = ratings.get(entity, default_rating) + delta / normalization

    return priors.fillna(default_rating)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_collector import collect_all_seasons

    raw_df = collect_all_seasons()
    processed_df = engineer_features(raw_df)
    print(processed_df.head())
