"""The three baselines (FR-301, FR-302, FR-303, FR-305).

These exist to make the project's question answerable. "The model gets a
Spearman of 0.7" means nothing until you know that starting-grid order alone
gets 0.6 — and in this sport it very nearly does, which is exactly why the
baselines are evaluated in P3, *before* the model exists and before there is
any temptation to move the bar.

All three implement `Predictor`, so the evaluation harness cannot tell them
apart from the model (FR-304).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from f1pred.models.base import rank_to_positions


class BaselineError(ValueError):
    """Raised when a baseline cannot read what it needs from a race frame."""


def _require(race: pd.DataFrame, column: str, who: str) -> pd.Series:
    if column not in race.columns:
        raise BaselineError(
            f"{who} needs column {column!r}, which the race frame does not "
            f"carry. Baselines read the same frame the model does (FR-304), so "
            f"a missing column is a harness bug, not a reason to substitute."
        )
    return race[column]


@dataclass(frozen=True, slots=True)
class GridOrderBaseline:
    """FR-301: finish where you started.

    The one that matters. Constitution I names beating this as the project's
    whole point, and FR-504 makes it the baseline every delta is measured
    against.
    """

    name: str = "grid_order"

    def predict(self, race: pd.DataFrame) -> pd.Series:
        grid = _require(race, "grid_position", self.name)
        return rank_to_positions(pd.to_numeric(grid, errors="coerce"), ascending=True)


@dataclass(frozen=True, slots=True)
class StandingsOrderBaseline:
    """FR-302: finish in championship order.

    Standings are read strictly before the race, from the same prior-points
    column the features use, so this baseline is subject to the same leakage
    rules as everything else. Ties — every driver on zero points at round one —
    fall back to grid order rather than to frame order, because an arbitrary
    tie-break would make the baseline's score depend on row ordering.
    """

    name: str = "standings_order"
    #: Cumulative season points before this race.
    points_column: str = "championship_points_prior"

    def predict(self, race: pd.DataFrame) -> pd.Series:
        points = pd.to_numeric(
            _require(race, self.points_column, self.name), errors="coerce"
        ).fillna(0.0)
        grid = pd.to_numeric(
            _require(race, "grid_position", self.name), errors="coerce"
        )
        # Sort by points descending, then grid ascending. Encoded as one key so
        # the ranking stays a single deterministic operation.
        key = -points * 1000.0 + grid.fillna(grid.max() + 1)
        return rank_to_positions(key, ascending=True)


@dataclass(frozen=True, slots=True)
class RandomOrderBaseline:
    """One seeded random ordering. FR-303 reports the *mean over many*.

    A single draw is deliberately not the deliverable: its variance across
    seeds is wider than the gap between a good model and a bad one, so quoting
    one would make the random floor meaningless. `RandomOrderControl` below is
    what the harness evaluates.
    """

    seed: int
    name: str = "random_order"

    def predict(self, race: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self.seed + int(race["season"].iloc[0]) * 1000
                                    + int(race["round"].iloc[0]))
        draw = pd.Series(rng.permutation(len(race)), index=race.index)
        return rank_to_positions(draw, ascending=True)


#: FR-303's floor: "at least 1000 seeded random orderings".
DEFAULT_RANDOM_DRAWS: Final[int] = 1000


@dataclass(frozen=True, slots=True)
class RandomOrderControl:
    """FR-303: the random-chance floor, as a mean over many seeded draws.

    Exposes the draws rather than an aggregate so the harness scores each one
    through the identical path as every other predictor (FR-304) and does its
    own averaging and interval.
    """

    draws: int = DEFAULT_RANDOM_DRAWS
    base_seed: int = 20260831
    name: str = "random_order"

    def __post_init__(self) -> None:
        if self.draws < DEFAULT_RANDOM_DRAWS:
            raise BaselineError(
                f"FR-303 requires at least {DEFAULT_RANDOM_DRAWS} orderings; "
                f"{self.draws} were asked for. A thinner control has variance "
                f"wide enough to make the floor meaningless."
            )

    def predictors(self) -> tuple[RandomOrderBaseline, ...]:
        return tuple(
            RandomOrderBaseline(seed=self.base_seed + i) for i in range(self.draws)
        )


def championship_points_prior(df: pd.DataFrame) -> pd.Series:
    """Cumulative points each driver had scored this season *before* this race.

    FR-302's input, and computed here rather than borrowed from the feature
    registry on purpose: a baseline that depended on the feature pipeline would
    stop being an independent comparison the moment that pipeline changed.
    """
    for column in ("season", "round", "driver_id", "points"):
        if column not in df.columns:
            raise BaselineError(f"championship standings need column {column!r}")
    ordered = df.sort_values(["season", "round", "driver_id"], kind="stable")
    cumulative = (
        ordered.groupby(["season", "driver_id"], sort=False)["points"]
        .apply(lambda s: s.shift(1).fillna(0.0).cumsum())
        .reset_index(level=[0, 1], drop=True)
    )
    return cumulative.reindex(df.index).astype("float64")


__all__ = [
    "BaselineError",
    "DEFAULT_RANDOM_DRAWS",
    "GridOrderBaseline",
    "RandomOrderBaseline",
    "RandomOrderControl",
    "StandingsOrderBaseline",
    "championship_points_prior",
]
