"""The chronological split (FR-403, FR-501, FR-501a, plan D-4).

The split is an object, not an argument. Nothing here accepts a `test_size`
float, a shuffle flag, or a random state, so random splitting is not
discouraged — it is unexpressible. That matters more than a warning would: a
random split over race data leaks the future into training through every
rolling feature, and it produces a number that looks wonderful.

The object also does not receive a season list. It receives a `CorpusWindow`
(D-9), which resolves itself from what is on disk, so the boundary advances at
a season rollover with no code or config edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd

from f1pred.corpus import CorpusWindow, default_window

#: Constitution IV.2, restated by FR-501. These are floors, not defaults, and
#: FR-501a forbids lowering them to fit a thin corpus.
MIN_TRAINING_SEASONS: Final[int] = 5
MIN_HOLDOUT_SEASONS: Final[int] = 3
MIN_TOTAL_SEASONS: Final[int] = 8


class SplitError(ValueError):
    """Raised when a split would violate the constitutional minima."""


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Train on the earlier seasons, evaluate on the later ones. Never mixed.

    `holdout_seasons` is how many of the window's *most recent* seasons are
    held out. It is a count rather than a fraction because a fraction would
    have to be rounded, and a rounding rule is one refactor away from becoming
    a `test_size`.
    """

    window: CorpusWindow
    holdout_seasons: int = MIN_HOLDOUT_SEASONS
    season_end_dates: dict[int, date] | None = None

    def __post_init__(self) -> None:
        if self.holdout_seasons < MIN_HOLDOUT_SEASONS:
            raise SplitError(
                f"holdout of {self.holdout_seasons} seasons is below the "
                f"Constitution IV.2 floor of {MIN_HOLDOUT_SEASONS}. FR-501a is "
                f"explicit that the minima are not lowered to fit."
            )
        seasons = self.window.resolve(self.season_end_dates)
        if len(seasons) < MIN_TOTAL_SEASONS:
            raise SplitError(
                f"{len(seasons)} seasons resolved; FR-501 requires at least "
                f"{MIN_TOTAL_SEASONS}."
            )
        if len(seasons) - self.holdout_seasons < MIN_TRAINING_SEASONS:
            raise SplitError(
                f"a {self.holdout_seasons}-season holdout leaves "
                f"{len(seasons) - self.holdout_seasons} training seasons, below "
                f"the floor of {MIN_TRAINING_SEASONS}."
            )

    # -- the boundary -------------------------------------------------------

    @property
    def seasons(self) -> tuple[int, ...]:
        return self.window.resolve(self.season_end_dates)

    @property
    def training_seasons(self) -> tuple[int, ...]:
        return self.seasons[: -self.holdout_seasons]

    @property
    def holdout(self) -> tuple[int, ...]:
        return self.seasons[-self.holdout_seasons :]

    @property
    def boundary_season(self) -> int:
        """The first held-out season. Nothing at or after it may train."""
        return self.holdout[0]

    # -- applying it --------------------------------------------------------

    def _seasons_of(self, df: pd.DataFrame) -> pd.Series:
        if "season" not in df.columns:
            raise SplitError("frame has no `season` column to split on")
        return pd.to_numeric(df["season"], errors="coerce")

    def train_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.training_seasons)

    def holdout_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.holdout)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Training rows and holdout rows, in that order.

        Rows from seasons outside the window belong to neither and are dropped
        — silently including them in training would widen the corpus past what
        the results file says it used.
        """
        return df[self.train_mask(df)].copy(), df[self.holdout_mask(df)].copy()

    def assert_no_leakage(self, training: pd.DataFrame) -> None:
        """FR-403: nothing at or after the boundary may appear in training.

        Called by the trainer rather than trusted, because the expensive
        version of this mistake is one that trains happily and reports a
        number nobody can reproduce.
        """
        seasons = set(self._seasons_of(training).dropna().astype(int))
        offending = sorted(s for s in seasons if s >= self.boundary_season)
        if offending:
            raise SplitError(
                f"training data contains season(s) {offending} at or after the "
                f"holdout boundary {self.boundary_season} (FR-403)"
            )

    # -- reporting ----------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """FR-506/FR-510: what every results file records about the split."""
        seasons = self.seasons
        shortfall = len(seasons) < MIN_TOTAL_SEASONS
        return {
            "window": self.window.describe(self.season_end_dates),
            "training_seasons": list(self.training_seasons),
            "holdout_seasons": list(self.holdout),
            "boundary_season": self.boundary_season,
            "n_training_seasons": len(self.training_seasons),
            "n_holdout_seasons": len(self.holdout),
            # FR-501a: recorded in every results file, never absorbed.
            "meets_constitutional_minimum": not shortfall,
            "minimum_required": {
                "total": MIN_TOTAL_SEASONS,
                "training": MIN_TRAINING_SEASONS,
                "holdout": MIN_HOLDOUT_SEASONS,
            },
        }


def default_split(
    *, holdout_seasons: int = MIN_HOLDOUT_SEASONS, as_of: date | None = None
) -> ChronologicalSplit:
    """The project's standard split: the default window, most recent N held out."""
    return ChronologicalSplit(window=default_window(as_of), holdout_seasons=holdout_seasons)


__all__ = [
    "MIN_HOLDOUT_SEASONS",
    "MIN_TOTAL_SEASONS",
    "MIN_TRAINING_SEASONS",
    "ChronologicalSplit",
    "SplitError",
    "default_split",
]
