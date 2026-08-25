"""The corpus window (FR-508, FR-509, FR-510, plan D-9).

The window resolves itself at run time from a clock rather than naming a season
list. A literal `SEASONS = [2010, ..., 2025]` would be correct for exactly one
year and silently wrong afterwards, with no way to distinguish "deliberately
narrow corpus" from "nobody updated the list".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from f1pred.config import (
    CORPUS_START_SEASON,
    DEFAULT_WINDOW_WIDTH,
    MIN_WINDOW_WIDTH,
)


class WindowTooNarrowError(ValueError):
    """Raised when a configuration would produce a sub-constitutional corpus."""


@dataclass(frozen=True, slots=True)
class CorpusWindow:
    """A fixed-width rolling window over completed seasons.

    `width` is configurable per FR-510 but may not resolve below
    `MIN_WINDOW_WIDTH` for any reported result. `as_of` is injected rather than
    read from the clock inside the class, so the season-boundary behaviour is
    testable without freezing global time.
    """

    width: int = DEFAULT_WINDOW_WIDTH
    as_of: date | None = None
    earliest_available: int = CORPUS_START_SEASON

    def __post_init__(self) -> None:
        if self.width < MIN_WINDOW_WIDTH:
            raise WindowTooNarrowError(
                f"width={self.width} resolves below the Constitution IV.2 floor "
                f"of {MIN_WINDOW_WIDTH} seasons. Failing loudly rather than "
                f"producing a sub-constitutional corpus (FR-510)."
            )

    # -- resolution ---------------------------------------------------------

    def _today(self) -> date:
        return self.as_of if self.as_of is not None else date.today()

    def latest_completed_season(
        self, season_end_dates: dict[int, date] | None = None
    ) -> int:
        """The most recent season whose final round has already been raced.

        FR-509: an in-progress season is inadmissible. When `season_end_dates`
        is supplied (from the collected schedule) completion is decided by the
        final round's race date. Without it we fall back to the calendar year,
        which is deliberately conservative: a season is treated as complete only
        once the following year has begun.
        """
        today = self._today()
        if season_end_dates:
            completed = [s for s, end in season_end_dates.items() if end < today]
            if not completed:
                raise ValueError(
                    "no season in season_end_dates has completed as of "
                    f"{today.isoformat()}"
                )
            return max(completed)
        return today.year - 1

    def resolve(
        self, season_end_dates: dict[int, date] | None = None
    ) -> tuple[int, ...]:
        """The concrete season list this window denotes."""
        newest = self.latest_completed_season(season_end_dates)
        oldest = newest - self.width + 1
        if oldest < self.earliest_available:
            raise WindowTooNarrowError(
                f"a {self.width}-season window ending {newest} would start in "
                f"{oldest}, before the earliest available season "
                f"{self.earliest_available}. Widen availability or narrow the "
                f"window deliberately — do not silently truncate."
            )
        return tuple(range(oldest, newest + 1))

    # -- reporting ----------------------------------------------------------

    def describe(
        self, season_end_dates: dict[int, date] | None = None
    ) -> dict[str, object]:
        """The record written into every results file (FR-510).

        The width alone does not identify a run — `W=16` means different seasons
        in different years — so the resolved list travels with it.
        """
        seasons = self.resolve(season_end_dates)
        return {
            "width": self.width,
            "as_of": self._today().isoformat(),
            "seasons": list(seasons),
            "first_season": seasons[0],
            "last_season": seasons[-1],
        }


def admissible(
    seasons: Sequence[int],
    window: CorpusWindow,
    season_end_dates: dict[int, date] | None = None,
) -> tuple[int, ...]:
    """Filter `seasons` to those the window admits, preserving order."""
    allowed = set(window.resolve(season_end_dates))
    return tuple(s for s in seasons if s in allowed)


def archive_seasons(
    latest_completed: int, earliest: int = CORPUS_START_SEASON
) -> tuple[int, ...]:
    """Every collected season, ignoring the rolling window.

    FR-511 runs over this rather than over `CorpusWindow`: the regulation-
    transfer experiment is a diagnostic about regulatory history, and binding it
    to a window that advances yearly would silently destroy observations as they
    age out.
    """
    return tuple(range(earliest, latest_completed + 1))


#: Convenience for callers that only need the default policy.
def default_window(as_of: date | None = None) -> CorpusWindow:
    return CorpusWindow(width=DEFAULT_WINDOW_WIDTH, as_of=as_of)


__all__: list[str] = [
    "CorpusWindow",
    "WindowTooNarrowError",
    "admissible",
    "archive_seasons",
    "default_window",
]
