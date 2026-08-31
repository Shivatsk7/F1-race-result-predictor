"""What became of every v0 feature (FR-202, T-202).

T-202 asks that each v0 feature be "ported or explicitly dropped with reason".
A prose paragraph claiming that would be unfalsifiable, so the claim lives here
as data and is tested: every v0 name must appear exactly once, every port must
name a feature that is actually registered, and every registered feature must
either descend from a v0 one or be listed as new.

`V0_FEATURES` is copied verbatim from `src/config.py` at tag `v0-vibe-coded`.
Do not tidy it. It is a historical record, and its order is the order the model
was actually trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Verbatim from v0's `config.FEATURES`.
V0_FEATURES: Final[tuple[str, ...]] = (
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
    "team_performance_index",
    "constructor_points_per_race_season",
    "constructor_reliability_season",
    "constructor_recent_finish_avg",
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
    "weather_dry",
    "weather_wet",
    "weather_mixed",
)


@dataclass(frozen=True, slots=True)
class Disposition:
    """Either `ported_as` names a registered feature, or `reason` explains why not."""

    ported_as: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if bool(self.ported_as) == bool(self.reason):
            raise ValueError(
                "a disposition is either a port or a documented drop, not both "
                "and not neither"
            )


_UNSOURCED_TRACK_INDEX = (
    "Dropped. Read from v0's hand-authored data/reference/track_metadata.csv, "
    "which carries no source, no method, and no date — the numbers were "
    "invented. FR-109 declares circuit type and overtaking tier as the circuit "
    "attribute set, and T-101b enumerated both from the corpus. Adding twelve "
    "further indices would put author guesses into the model wearing the "
    "clothes of data."
)

_OBSERVED_WEATHER = (
    "Dropped from the primary set. Computed in v0 from the weather that was "
    "recorded during the race, which Constitution III.2 and FR-206 make "
    "inadmissible — this is the leak that sat in the v0 model for four months "
    "and inflated every number it produced. FR-206 admits a pre-race forecast "
    "or a climatological prior instead; T-107 established that forecasts exist "
    "from 2017 and ERA5 priors cover the whole corpus, but neither is collected "
    "yet, so no weather feature is registered."
)

DISPOSITIONS: Final[dict[str, Disposition]] = {
    # -- starting position -------------------------------------------------
    "grid_position": Disposition(ported_as="grid_slot"),
    "grid_position_sq": Disposition(ported_as="grid_position_sq"),
    "quali_vs_teammate": Disposition(ported_as="grid_rank_vs_teammate"),
    "street_grid_pressure": Disposition(ported_as="grid_x_street"),
    "overqualified_by_grid": Disposition(ported_as="overqualified_by_grid"),
    "strength_rank_prior": Disposition(ported_as="strength_rank_prior"),
    "qualifying_gap_to_pole": Disposition(
        reason=(
            "Dropped. Qualifying session times are not in the canonical row "
            "schema (T-101), so the gap cannot be computed from the collected "
            "corpus. Adding it is a collection change, not a feature one. "
            "grid_rank_vs_teammate carries the same signal at lower resolution."
        )
    ),
    "weighted_quali_gap": Disposition(
        reason=(
            "Dropped with its input, qualifying_gap_to_pole. The product of a "
            "missing quantity and a dropped track weight has nothing left to "
            "compute."
        )
    ),
    "underqualified_by_grid": Disposition(
        reason=(
            "Dropped as the exact negation of overqualified_by_grid. Two "
            "features that are x and -x carry one signal between them; the "
            "second only splits the importance attributed to the first."
        )
    ),
    "track_grid_weight": Disposition(
        reason=(
            "Dropped. A logistic of five undeclared track indices with six "
            "hand-chosen coefficients, none of them fitted or justified. "
            "grid_x_overtaking expresses the same idea using the one circuit "
            "attribute the project actually declares, and leaves the shape of "
            "the interaction to the model."
        )
    ),
    "weighted_grid_position": Disposition(ported_as="grid_x_overtaking"),
    "grid_track_interaction": Disposition(
        reason=(
            "Dropped as a duplicate: grid position multiplied by a track "
            "difficulty index is what grid_x_overtaking now computes."
        )
    ),
    # -- driver form --------------------------------------------------------
    "driver_elo_race": Disposition(ported_as="driver_elo_race_pct"),
    "driver_elo_quali": Disposition(ported_as="driver_elo_quali_pct"),
    "driver_win_rate_season": Disposition(ported_as="driver_win_rate_season_pct"),
    "driver_podium_rate_season": Disposition(ported_as="driver_podium_rate_season_pct"),
    "driver_points_per_race_season": Disposition(
        ported_as="driver_points_per_race_season_pct"
    ),
    "driver_recent_finish_avg": Disposition(ported_as="driver_finish_last3_pct"),
    "driver_recent_finish_last5": Disposition(ported_as="driver_finish_last5_pct"),
    "driver_recent_grid_avg": Disposition(ported_as="driver_grid_last3_pct"),
    "driver_recent_points_last3": Disposition(ported_as="driver_points_last3_pct"),
    "driver_recent_gain_avg": Disposition(ported_as="driver_gain_last5_pct"),
    # -- constructor form ---------------------------------------------------
    "constructor_points_per_race_season": Disposition(
        ported_as="constructor_points_per_race_season_pct"
    ),
    "constructor_reliability_season": Disposition(
        ported_as="constructor_reliability_season_pct"
    ),
    "constructor_recent_finish_avg": Disposition(ported_as="constructor_finish_last3_pct"),
    "team_performance_index": Disposition(
        reason=(
            "Dropped. A blend of three quantities with weights 0.45/0.30/0.25 "
            "and two magic constants (21 - mean finish, 20 x reliability) that "
            "nobody chose for a reason. All three components are registered "
            "separately, so the model can weight them itself and be disagreed "
            "with when it does."
        )
    ),
    # -- circuit history ----------------------------------------------------
    "driver_avg_finish_at_circuit": Disposition(ported_as="driver_circuit_finish_mean_pct"),
    "driver_avg_grid_at_circuit": Disposition(ported_as="driver_circuit_grid_mean_pct"),
    "driver_best_finish_at_circuit": Disposition(ported_as="driver_circuit_best_finish_pct"),
    "constructor_avg_finish_at_circuit": Disposition(
        ported_as="constructor_circuit_finish_mean_pct"
    ),
    # -- circuit attributes -------------------------------------------------
    "is_street": Disposition(ported_as="is_street_circuit"),
    "is_permanent": Disposition(ported_as="is_permanent_circuit"),
    "is_hybrid": Disposition(ported_as="is_hybrid_circuit"),
    "overtaking_difficulty": Disposition(ported_as="overtaking_difficulty"),
    "is_temporary": Disposition(
        reason=(
            "Dropped. v0 carried four circuit categories where the declared "
            "table has three, and 'temporary' overlapped both street and "
            "hybrid. A category with no boundary is not a category."
        )
    ),
    "track_archetype_code": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "benchmark_group_code": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "track_position_importance": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "safety_car_risk": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "pit_delta_index": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "tire_deg_index": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "avg_speed_index": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "high_downforce_bias": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "traction_bias": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "braking_severity": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "surface_evolution_rate": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    "weather_volatility": Disposition(reason=_UNSOURCED_TRACK_INDEX),
    # -- weather ------------------------------------------------------------
    "weather_dry": Disposition(reason=_OBSERVED_WEATHER),
    "weather_wet": Disposition(reason=_OBSERVED_WEATHER),
    "weather_mixed": Disposition(reason=_OBSERVED_WEATHER),
}


#: Features with no v0 ancestor, and why they exist. Listed so the disposition
#: test can insist that every registered feature is accounted for in one
#: direction or the other — a feature that appeared from nowhere is as much a
#: gap in the record as a v0 one that vanished.
NEW_FEATURES: Final[dict[str, str]] = {
    "field_size": "How many cars started; the corpus ranges from 20 to 26.",
    "is_sprint_weekend": "FR-211: the weekend's format is declared in advance.",
    "regime_id": "FR-210: the regulation regime in force, from the declared table.",
    "seasons_since_regime_start": "FR-210: seasons elapsed since the regime began.",
    "regime_is_major": "FR-210: whether that regime began with a major reset.",
    "driver_prior_races": "FR-212: zero identifies a cold start exactly.",
    "constructor_prior_races": "FR-212: zero identifies a new team exactly.",
    "sprint_finish_pct": "FR-211: the sprint result, null where no sprint ran.",
    "sprint_places_gained_pct": "FR-211: sprint race-day progress, null likewise.",
    "practice_long_run_pace_pct": "FR-213: telemetry group, separate arm.",
    "practice_stint_trend_pct": "FR-213: telemetry group, separate arm.",
    "practice_best_lap_pct": "FR-213: telemetry group, separate arm.",
    "practice_long_run_laps": "FR-213: sample size behind the practice pace.",
    "weather_prior_wet": "FR-206: climatological prior, admissible pre-race.",
    "weather_prior_temp": "FR-206: climatological prior, admissible pre-race.",
    "weather_prior_wind": "FR-206: climatological prior, admissible pre-race.",
    "weather_actual_precip": "FR-206/FR-207: observed, oracle set only.",
    "weather_actual_temp": "FR-206/FR-207: observed, oracle set only.",
    "finished_race": "FR-207: post-race, oracle set only.",
    "race_distance_share": "FR-207: post-race, oracle set only.",
}


def ported_names() -> tuple[str, ...]:
    return tuple(d.ported_as for d in DISPOSITIONS.values() if d.ported_as)


def dropped_names() -> tuple[str, ...]:
    return tuple(name for name, d in DISPOSITIONS.items() if not d.ported_as)


__all__ = [
    "DISPOSITIONS",
    "Disposition",
    "NEW_FEATURES",
    "V0_FEATURES",
    "dropped_names",
    "ported_names",
]
