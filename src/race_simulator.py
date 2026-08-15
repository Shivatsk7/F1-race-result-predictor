"""Probabilistic race simulation layers built on top of the base pace model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE


POSITION_PROB_PREFIX = "prob_p"


@dataclass(frozen=True)
class SimulationConfig:
    """Controls the Monte Carlo race simulation."""

    n_simulations: int = 1000
    random_state: int = RANDOM_STATE
    enable_overtaking: bool = True
    enable_strategy: bool = True
    min_pace_noise: float = 0.20
    max_pace_noise: float = 2.50


def compute_base_pace(prediction_frame: pd.DataFrame) -> pd.Series:
    """Convert the current model output into a continuous pace score.

    Lower values are faster. The preferred source is the auxiliary regressor's
    expected finish, which preserves more signal than the integer rank.
    """
    if "predicted_position_raw" in prediction_frame.columns:
        pace = prediction_frame["predicted_position_raw"]
    elif "predicted_position" in prediction_frame.columns:
        pace = prediction_frame["predicted_position"]
    elif "grid_position" in prediction_frame.columns:
        pace = prediction_frame["grid_position"]
    else:
        raise ValueError(
            "Simulation input must include predicted_position_raw, "
            "predicted_position, or grid_position."
        )

    field_size = len(prediction_frame)
    return pd.to_numeric(pace, errors="coerce").fillna(field_size).clip(1, field_size)


def apply_track_adjustment(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Apply track-specific grid anchoring to base pace.

    Formula:
        adjusted_pace = pace_score + 0.70 * grid_weight * (grid_position - pace_score)

    On Monaco/Singapore this pulls outcomes closer to starting position. On
    Brazil/Baku it leaves more room for race pace and stochastic recovery.
    """
    df = prediction_frame.copy().reset_index(drop=True)
    field_size = len(df)

    df["pace_score"] = compute_base_pace(df)
    df["track_type"] = df.apply(_infer_track_type, axis=1)
    df["overtaking_difficulty_score"] = _scale_metadata(df, "overtaking_difficulty", default=0.50)
    df["grid_weight"] = _resolve_grid_weight(df)
    df["safety_car_probability"] = _scale_metadata(df, "safety_car_risk", default=0.35)
    df["tire_degradation_level"] = _scale_metadata(df, "tire_deg_index", default=0.50)
    df["pit_stop_delta_level"] = _scale_metadata(df, "pit_delta_index", default=0.50)
    df["weather_randomness"] = _scale_metadata(df, "weather_volatility", default=0.30)

    grid_position = pd.to_numeric(df["grid_position"], errors="coerce").fillna(field_size)
    grid_pull = df["grid_weight"] * (grid_position - df["pace_score"])
    random_track_penalty = 0.25 * (df["safety_car_probability"] + df["weather_randomness"] - 0.70)
    df["adjusted_pace"] = (df["pace_score"] + 0.70 * grid_pull + random_track_penalty).clip(1, field_size)
    return df


def apply_driver_variability(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach driver-specific variability, overtaking skill, and DNF risk."""
    df = prediction_frame.copy().reset_index(drop=True)
    field_size = len(df)

    pace_strength = _rank_percentile(-df["pace_score"])
    elo_strength = _rank_percentile(_numeric(df, "driver_elo_race", 1500.0))
    recent_gain = _numeric(df, "driver_recent_gain_avg", 0.0)
    gain_strength = _rank_percentile(recent_gain)
    team_reliability = _numeric(df, "constructor_reliability_season", 0.88).clip(0.0, 1.0)

    driver_strength = (0.45 * elo_strength + 0.35 * pace_strength + 0.20 * gain_strength).clip(0.0, 1.0)
    track_randomness = (
        0.55 * df["safety_car_probability"]
        + 0.30 * df["weather_randomness"]
        + 0.15 * (1.0 - df["grid_weight"])
    ).clip(0.0, 1.0)

    df["consistency_score"] = (
        0.35
        + 0.85 * (1.0 - driver_strength)
        + 0.45 * track_randomness
    ).clip(0.20, 2.50)
    df["overtaking_skill"] = (
        0.50 * driver_strength
        + 0.30 * gain_strength
        + 0.20 * (1.0 - df["overtaking_difficulty_score"])
    ).clip(0.05, 0.95)
    df["error_rate"] = (
        0.012
        + 0.030 * track_randomness
        + 0.050 * (1.0 - team_reliability)
        + 0.012 * (1.0 - driver_strength)
    ).clip(0.005, 0.14)

    # Keep the scale stable for short grids used in tests.
    scale_factor = np.sqrt(max(field_size, 1) / 20.0)
    df["consistency_score"] = df["consistency_score"] * scale_factor
    return df


def simulate_race_once(
    simulation_frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    enable_overtaking: bool = True,
    enable_strategy: bool = True,
) -> np.ndarray:
    """Run one race simulation and return finishing positions by input row."""
    frame = _ensure_simulation_ready(simulation_frame)
    n_drivers = len(frame)
    safety_car = rng.random() < float(frame["safety_car_probability"].mean())

    pace = frame["adjusted_pace"].to_numpy(dtype=float).copy()
    pace += rng.normal(0.0, frame["consistency_score"].to_numpy(dtype=float))

    if safety_car:
        chaos_scale = 0.35 + 0.75 * float(frame["safety_car_probability"].mean())
        pace += rng.normal(0.0, chaos_scale, size=n_drivers)

    if enable_strategy:
        pace += _sample_strategy_effects(frame, rng, 1, np.array([safety_car]))[0]

    dnf_mask = rng.random(n_drivers) < frame["error_rate"].to_numpy(dtype=float)
    if enable_overtaking:
        order = _simulate_overtaking_order(frame, pace, dnf_mask, rng)
    else:
        order = np.lexsort((frame["grid_position"].to_numpy(dtype=float), pace))
        if dnf_mask.any():
            finishers = [idx for idx in order if not dnf_mask[idx]]
            dnfs = [idx for idx in order if dnf_mask[idx]]
            rng.shuffle(dnfs)
            order = np.array(finishers + dnfs, dtype=int)

    positions = np.empty(n_drivers, dtype=int)
    positions[order] = np.arange(1, n_drivers + 1)
    return positions


def run_monte_carlo(
    prediction_frame: pd.DataFrame,
    *,
    n_simulations: int = 1000,
    random_state: int = RANDOM_STATE,
    enable_overtaking: bool = True,
    enable_strategy: bool = True,
) -> pd.DataFrame:
    """Run many race simulations and aggregate position probabilities."""
    config = SimulationConfig(
        n_simulations=n_simulations,
        random_state=random_state,
        enable_overtaking=enable_overtaking,
        enable_strategy=enable_strategy,
    )
    frame = _ensure_simulation_ready(prediction_frame)
    rng = np.random.default_rng(config.random_state)
    n_drivers = len(frame)

    base_pace = frame["adjusted_pace"].to_numpy(dtype=float)
    noise = rng.normal(
        0.0,
        frame["consistency_score"].to_numpy(dtype=float),
        size=(config.n_simulations, n_drivers),
    )
    simulated_pace = base_pace.reshape(1, -1) + noise

    safety_car_events = rng.random(config.n_simulations) < float(frame["safety_car_probability"].mean())
    if safety_car_events.any():
        chaos_scale = 0.35 + 0.75 * float(frame["safety_car_probability"].mean())
        simulated_pace[safety_car_events] += rng.normal(
            0.0,
            chaos_scale,
            size=(int(safety_car_events.sum()), n_drivers),
        )

    if config.enable_strategy:
        simulated_pace += _sample_strategy_effects(frame, rng, config.n_simulations, safety_car_events)

    dnf_matrix = rng.random((config.n_simulations, n_drivers)) < frame["error_rate"].to_numpy(dtype=float)
    position_matrix = np.empty((config.n_simulations, n_drivers), dtype=np.int16)

    if config.enable_overtaking:
        for sim_idx in range(config.n_simulations):
            order = _simulate_overtaking_order(frame, simulated_pace[sim_idx], dnf_matrix[sim_idx], rng)
            position_matrix[sim_idx, order] = np.arange(1, n_drivers + 1)
    else:
        for sim_idx in range(config.n_simulations):
            order = np.lexsort((frame["grid_position"].to_numpy(dtype=float), simulated_pace[sim_idx]))
            if dnf_matrix[sim_idx].any():
                finishers = [idx for idx in order if not dnf_matrix[sim_idx, idx]]
                dnfs = [idx for idx in order if dnf_matrix[sim_idx, idx]]
                rng.shuffle(dnfs)
                order = np.array(finishers + dnfs, dtype=int)
            position_matrix[sim_idx, order] = np.arange(1, n_drivers + 1)

    return aggregate_results(frame, position_matrix, dnf_matrix)


def aggregate_results(
    simulation_frame: pd.DataFrame,
    position_matrix: np.ndarray,
    dnf_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate simulation outcomes into probabilities and intervals."""
    frame = simulation_frame.reset_index(drop=True).copy()
    n_drivers = len(frame)
    n_simulations = position_matrix.shape[0]

    output = frame[["driver", "team", "grid_position"]].copy()
    output["expected_finish"] = np.round(position_matrix.mean(axis=0), 3)
    output["predicted_rank"] = output["expected_finish"].rank(method="first").astype(int)
    output["finish_p05"] = np.percentile(position_matrix, 5, axis=0).round(1)
    output["finish_p95"] = np.percentile(position_matrix, 95, axis=0).round(1)
    output["win_probability"] = np.round((position_matrix == 1).mean(axis=0), 4)
    output["podium_probability"] = np.round((position_matrix <= 3).mean(axis=0), 4)
    output["top5_probability"] = np.round((position_matrix <= 5).mean(axis=0), 4)
    if dnf_matrix is not None:
        output["dnf_probability"] = np.round(dnf_matrix.mean(axis=0), 4)

    for position in range(1, n_drivers + 1):
        output[f"{POSITION_PROB_PREFIX}{position:02d}"] = np.round(
            (position_matrix == position).mean(axis=0),
            4,
        )

    diagnostics = [
        "pace_score",
        "adjusted_pace",
        "grid_weight",
        "overtaking_difficulty_score",
        "safety_car_probability",
        "consistency_score",
        "overtaking_skill",
        "error_rate",
        "track_type",
    ]
    for column in diagnostics:
        output[column] = frame[column].to_numpy()

    return output.sort_values(
        ["predicted_rank", "expected_finish", "grid_position"],
        ignore_index=True,
    )


def _ensure_simulation_ready(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure all simulation layers have been applied."""
    frame = prediction_frame.copy().reset_index(drop=True)
    if "adjusted_pace" not in frame.columns:
        frame = apply_track_adjustment(frame)
    if "consistency_score" not in frame.columns:
        frame = apply_driver_variability(frame)
    return frame


def _simulate_overtaking_order(
    frame: pd.DataFrame,
    pace: np.ndarray,
    dnf_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate adjacent overtaking opportunities from grid order."""
    grid = frame["grid_position"].to_numpy(dtype=float)
    order = np.argsort(grid, kind="stable").astype(int)
    difficulty = float(frame["overtaking_difficulty_score"].mean())
    base_passes = int(np.ceil(len(order) * (0.25 + 0.95 * (1.0 - difficulty))))
    n_passes = max(3, min(len(order), base_passes))
    skill = frame["overtaking_skill"].to_numpy(dtype=float)

    for _ in range(n_passes):
        for position in range(len(order) - 2, -1, -1):
            ahead = order[position]
            behind = order[position + 1]
            if dnf_mask[ahead] or dnf_mask[behind]:
                continue
            pace_advantage = pace[ahead] - pace[behind]
            if pace_advantage <= 0:
                continue
            probability = overtake_probability(
                pace_advantage,
                difficulty,
                skill[behind],
            )
            if rng.random() < probability:
                order[position], order[position + 1] = order[position + 1], order[position]

    if dnf_mask.any():
        finishers = [idx for idx in order if not dnf_mask[idx]]
        dnfs = [idx for idx in order if dnf_mask[idx]]
        dnfs.sort(key=lambda idx: pace[idx], reverse=True)
        order = np.array(finishers + dnfs, dtype=int)

    return order


def overtake_probability(
    pace_difference: float,
    overtaking_difficulty_score: float,
    overtaking_skill: float,
) -> float:
    """Probability that the trailing car passes an adjacent slower car."""
    raw = (
        1.30 * pace_difference
        + 1.15 * overtaking_skill
        - 2.25 * overtaking_difficulty_score
        - 0.35
    )
    probability = 1.0 / (1.0 + np.exp(-raw))
    track_gate = 0.10 + 0.90 * (1.0 - overtaking_difficulty_score)
    return float(np.clip(probability * track_gate, 0.01, 0.90))


def _sample_strategy_effects(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    n_simulations: int,
    safety_car_events: np.ndarray,
) -> np.ndarray:
    """Sample simplified one-stop/two-stop strategy effects in position units."""
    tire_deg = frame["tire_degradation_level"].to_numpy(dtype=float)
    pit_delta = frame["pit_stop_delta_level"].to_numpy(dtype=float)
    grid_weight = frame["grid_weight"].to_numpy(dtype=float)

    two_stop_probability = np.clip(0.15 + 0.55 * tire_deg - 0.25 * pit_delta, 0.05, 0.75)
    two_stop = rng.random((n_simulations, len(frame))) < two_stop_probability.reshape(1, -1)

    extra_pit_loss = pit_delta * (0.45 + 0.85 * grid_weight)
    fresh_tyre_gain = tire_deg * (0.35 + 0.30 * (1.0 - grid_weight))
    strategy_effect = two_stop * (extra_pit_loss - fresh_tyre_gain).reshape(1, -1)

    if safety_car_events.any():
        safety_car_discount = (0.45 * pit_delta).reshape(1, -1)
        strategy_effect[safety_car_events] -= two_stop[safety_car_events] * safety_car_discount

    strategy_std = 0.10 + 0.35 * tire_deg + 0.15 * (1.0 - grid_weight)
    strategy_effect += rng.normal(0.0, strategy_std, size=(n_simulations, len(frame)))
    return strategy_effect


def _resolve_grid_weight(df: pd.DataFrame) -> pd.Series:
    """Resolve or derive grid position importance on a 0-1 scale."""
    position_importance = _scale_metadata(df, "track_position_importance", default=0.50)
    overtaking_difficulty = _scale_metadata(df, "overtaking_difficulty", default=0.50)
    is_street = _numeric(df, "is_street", 0.0).clip(0.0, 1.0)
    pit_delta = _scale_metadata(df, "pit_delta_index", default=0.50)
    tire_deg = _scale_metadata(df, "tire_deg_index", default=0.50)

    grid_weight = (
        0.15
        + 0.35 * position_importance
        + 0.25 * overtaking_difficulty
        + 0.15 * is_street
        + 0.10 * pit_delta
        - 0.25 * tire_deg
    )

    archetype = df.get("track_archetype", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    benchmark = df.get("benchmark_group", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    grid_weight = grid_weight + np.where((archetype == "high_grid") | (benchmark == "high_grid"), 0.15, 0.0)
    grid_weight = grid_weight - np.where((archetype == "power") | (benchmark == "power"), 0.12, 0.0)
    grid_weight = grid_weight - np.where((archetype == "chaos") | (benchmark == "chaos"), 0.25, 0.0)
    return pd.Series(grid_weight, index=df.index).clip(0.15, 0.90)


def _infer_track_type(row: pd.Series) -> str:
    """Map metadata into simulator track categories."""
    archetype = str(row.get("track_archetype", "")).lower()
    is_street = float(row.get("is_street", 0) or 0)
    high_downforce = _scale_scalar(row.get("high_downforce_bias", 3), default=0.50)

    if is_street >= 0.5:
        return "street"
    if archetype == "power":
        return "power"
    if archetype == "high_grid" or high_downforce >= 0.75:
        return "high_downforce"
    return "balanced"


def _scale_metadata(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Scale metadata that may be supplied as 1-5 or already as 0-1."""
    values = _numeric(df, column, default)
    if values.max() > 1.0:
        values = (values - 1.0) / 4.0
    return values.clip(0.0, 1.0)


def _scale_scalar(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric > 1.0:
        numeric = (numeric - 1.0) / 4.0
    return float(np.clip(numeric, 0.0, 1.0))


def _numeric(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Read a numeric column with a stable fallback."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _rank_percentile(values: pd.Series) -> pd.Series:
    """Convert values to 0-1 percentile ranks where higher is better."""
    if len(values) <= 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    ranked = values.rank(method="average", pct=True)
    return ranked.fillna(0.5).clip(0.0, 1.0)
