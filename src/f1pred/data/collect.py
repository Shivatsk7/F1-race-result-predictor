"""Resumable, checkpointed collection (FR-101..111).

Design notes that are not obvious from the requirements:

* **Completeness is asserted, never inferred** (FR-110). The upstream API caps
  `limit` at 100 server-side and truncates without error, so a request for a
  whole season returns five races and looks entirely successful. The T-099
  spike's own first pass under-counted 2023's sprints 6 -> 2 this way. Every
  paginated read therefore checks against the total the API reports.

* **A failed race is a gap, not a deletion** (FR-104a..c). It is discarded
  whole, recorded in a manifest, and counted toward an abort threshold.

* **Transient failure is not absence** (FR-111). The 5-second upstream default
  tripped three times in ninety requests during the spike, then succeeded on
  immediate retry.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from f1pred.config import MAX_FAILURE_RATE, MAX_RETRIES, ROW_COUNT_DEVIATION_TOLERANCE

#: The API caps `limit` at 100 regardless of what is requested.
PAGE_SIZE: int = 100


class CollectionError(RuntimeError):
    """Raised when collection cannot honestly continue."""


class IncompletePageError(CollectionError):
    """Raised when a paginated read did not reach the reported total (FR-110)."""


class FailureThresholdExceeded(CollectionError):
    """Raised when too many races failed to trust the corpus (FR-104c)."""


# --------------------------------------------------------------------------
# Retry (FR-111)
# --------------------------------------------------------------------------


def with_retry(
    fn: Callable[[], Any],
    *,
    what: str,
    attempts: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry a transient failure with exponential backoff.

    Exhausting retries raises. The caller records that as a *failure* under
    FR-104, never as a session that does not exist — the distinction matters
    because the latter would silently shrink the corpus.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any transport error is fair game
            last = exc
            if attempt < attempts - 1:
                sleep(2.0**attempt)
    raise CollectionError(
        f"{what}: {attempts} attempts exhausted; last error "
        f"{type(last).__name__}: {last}"
    ) from last


# --------------------------------------------------------------------------
# Pagination (FR-110)
# --------------------------------------------------------------------------


class PagedResponse(Protocol):
    """The shape this module needs from a paginated API response."""

    @property
    def total_results(self) -> int: ...


def paginate(
    fetch: Callable[[int, int], Any],
    *,
    what: str,
    page_size: int = PAGE_SIZE,
    rows_in: Callable[[Any], int],
    sleep: Callable[[float], None] = time.sleep,
    pause: float = 0.0,
) -> list[Any]:
    """Page to exhaustion, asserting against the total the API reports.

    `fetch(limit, offset)` returns a response exposing `total_results`.
    """
    pages: list[Any] = []
    offset = 0
    seen = 0
    total: int | None = None

    while True:
        if pause:
            sleep(pause)
        page = with_retry(
            lambda: fetch(page_size, offset), what=f"{what} offset={offset}"
        )
        reported = int(page.total_results)
        if total is None:
            total = reported
        elif reported != total:
            raise IncompletePageError(
                f"{what}: reported total changed mid-pagination "
                f"({total} -> {reported}); the corpus underneath is moving"
            )

        got = rows_in(page)
        pages.append(page)
        seen += got
        offset += page_size

        if seen >= total:
            break
        if got == 0:
            raise IncompletePageError(
                f"{what}: pagination stalled at {seen}/{total} rows. A page "
                f"returning nothing while rows remain means the read is "
                f"truncated, which is indistinguishable from a small season "
                f"unless asserted (FR-110)."
            )

    if seen != total:
        raise IncompletePageError(
            f"{what}: collected {seen} rows but the API reports {total}. "
            f"Completeness is asserted from the reported total, never inferred "
            f"from a successful response (FR-110)."
        )
    return pages


# --------------------------------------------------------------------------
# Failure manifest (FR-104a..c)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RaceFailure:
    season: int
    round: int
    reason: str
    race_date: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "season": self.season,
            "round": self.round,
            "race_date": self.race_date,
            "reason": self.reason,
        }


@dataclass
class FailureManifest:
    """FR-104b: failures persist as gaps rather than vanishing.

    Rolling features must be able to tell "this race is missing" from "this race
    did not happen", so the manifest travels with the data and is reproduced in
    every results file.
    """

    failures: list[RaceFailure] = field(default_factory=list)
    races_attempted: int = 0
    max_failure_rate: float = MAX_FAILURE_RATE

    def record_success(self) -> None:
        self.races_attempted += 1

    def record_failure(
        self, season: int, rnd: int, reason: str, race_date: str | None = None
    ) -> None:
        self.races_attempted += 1
        self.failures.append(RaceFailure(season, rnd, reason, race_date))

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def failure_rate(self) -> float:
        if self.races_attempted == 0:
            return 0.0
        return self.failure_count / self.races_attempted

    def missing_rounds(self, season: int) -> set[int]:
        return {f.round for f in self.failures if f.season == season}

    def check_threshold(self) -> None:
        """FR-104c: abort rather than produce a corpus that looks complete."""
        if self.failure_rate > self.max_failure_rate:
            raise FailureThresholdExceeded(
                f"{self.failure_count} of {self.races_attempted} races failed "
                f"({self.failure_rate:.1%}), above the {self.max_failure_rate:.1%} "
                f"threshold. Continuing would yield a corpus that looks complete "
                f"and is not (FR-104c)."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "races_attempted": self.races_attempted,
            "failure_count": self.failure_count,
            "failure_rate": round(self.failure_rate, 4),
            "max_failure_rate": self.max_failure_rate,
            "failures": [f.as_dict() for f in self.failures],
        }

    def summary(self) -> str:
        if not self.failures:
            return f"All {self.races_attempted} races collected; no failures."
        head = ", ".join(f"{f.season} r{f.round}" for f in self.failures[:5])
        more = "" if self.failure_count <= 5 else f" (+{self.failure_count - 5} more)"
        return (
            f"{self.failure_count} of {self.races_attempted} races failed "
            f"({self.failure_rate:.1%}): {head}{more}"
        )


# --------------------------------------------------------------------------
# Row-count sanity (FR-106)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeasonCount:
    season: int
    rows: int
    races: int
    expected_rows: int

    @property
    def deviation(self) -> float:
        if self.expected_rows == 0:
            return 0.0
        return abs(self.rows - self.expected_rows) / self.expected_rows

    @property
    def is_anomalous(self) -> bool:
        return self.deviation > ROW_COUNT_DEVIATION_TOLERANCE

    def warning(self) -> str | None:
        if not self.is_anomalous:
            return None
        return (
            f"season {self.season}: {self.rows} rows across {self.races} races "
            f"deviates {self.deviation:.1%} from the expected "
            f"{self.expected_rows} (>{ROW_COUNT_DEVIATION_TOLERANCE:.0%})"
        )


def check_season_counts(
    df: pd.DataFrame, *, expected_entries_per_race: int | None = None
) -> list[SeasonCount]:
    """FR-106: per-season row counts, with a deviation warning.

    The expected entry count is the season's own **modal** field size unless
    supplied. A fixed constant would fire every season, because field size
    genuinely varies across the corpus (24 cars in 2010, 20 today). The median
    is not good enough either: it moves toward an anomalous race and can absorb
    the very deviation this check exists to catch. The mode does not, since most
    races run the full field.

    Note the scope: this is a *season*-level check, as FR-106 specifies. A single
    truncated race inside a full season shifts the season total by only a few
    percent and will not trip it.
    """
    counts: list[SeasonCount] = []
    for season, season_df in df.groupby("season", sort=True):
        races = int(season_df["round"].nunique())
        per_race = season_df.groupby("round").size()
        if expected_entries_per_race is not None:
            entries = expected_entries_per_race
        else:
            modes = per_race.mode()
            entries = int(modes.max()) if not modes.empty else 0
        counts.append(
            SeasonCount(
                season=int(season),
                rows=len(season_df),
                races=races,
                expected_rows=races * entries,
            )
        )
    return counts


# --------------------------------------------------------------------------
# Checkpointing (FR-102, FR-103)
# --------------------------------------------------------------------------


def checkpoint_path(root: "Any", season: int) -> "Any":
    from pathlib import Path

    return Path(root) / f"season_{season}.parquet"


def write_checkpoint(df: pd.DataFrame, root: Any, season: int) -> Any:
    """FR-102: persist a season atomically.

    Written to a temporary name and moved into place, so an interrupted write
    cannot leave a half-season that the next run mistakes for a checkpoint.
    """
    from pathlib import Path

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = checkpoint_path(root, season)
    tmp = final.with_suffix(".parquet.partial")
    df.to_parquet(tmp, index=False)
    tmp.replace(final)
    return final


def read_checkpoint(root: Any, season: int) -> pd.DataFrame | None:
    path = checkpoint_path(root, season)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def completed_seasons(root: Any, seasons: "Any") -> set[int]:
    """FR-102/FR-103: which seasons already have a valid checkpoint."""
    return {s for s in seasons if checkpoint_path(root, s).exists()}


def seasons_to_collect(root: Any, seasons: "Any") -> list[int]:
    """FR-103: resume from the first incomplete season, refetching nothing."""
    done = completed_seasons(root, seasons)
    return [s for s in seasons if s not in done]


def iter_pending(root: Any, seasons: "Any") -> Iterator[int]:
    yield from seasons_to_collect(root, seasons)


__all__ = [
    "PAGE_SIZE",
    "CollectionError",
    "FailureManifest",
    "FailureThresholdExceeded",
    "IncompletePageError",
    "RaceFailure",
    "SeasonCount",
    "check_season_counts",
    "checkpoint_path",
    "completed_seasons",
    "iter_pending",
    "paginate",
    "read_checkpoint",
    "seasons_to_collect",
    "with_retry",
    "write_checkpoint",
]
