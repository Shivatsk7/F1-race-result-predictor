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

* **A race is reassembled before it is stored** (FR-104a). The upstream API
  paginates by *result row*, not by race, so the last race on a page is
  routinely truncated mid-field — a 2016 season read returns round 5 with 12 of
  its 22 entries on page 1 and the remaining 10 on page 2. Storing a page as it
  arrives would silently shrink a field, which is precisely the partial race
  FR-104a forbids.

This module is the only part of the project that touches the network (NFR-05).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import pandas as pd

from f1pred.config import (
    FASTF1_CACHE_DIR,
    MAX_FAILURE_RATE,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    ROW_COUNT_DEVIATION_TOLERANCE,
)
from f1pred.data.schema import (
    RACE_ROW_SCHEMA,
    ClassificationOutcome,
    SchemaError,
    classify,
    empty_frame,
    WeekendFormat,
    map_weekend_format,
    validate_frame,
)

# target.py imports only schema, so this direction carries no cycle.
from f1pred.data.target import FINAL_LAP_POSITION_COLUMN, drop_non_starters

#: The API caps `limit` at 100 regardless of what is requested.
PAGE_SIZE: int = 100

#: Politeness pause between requests. jolpica is a free community service run by
#: volunteers; the whole corpus is only ~100 requests, so there is no reason to
#: go faster than one per second.
REQUEST_PAUSE_SECONDS: float = 1.0

#: Extra sleep after a 429 before the FR-111 retry ladder resumes. Backing off by
#: the ordinary 1/2/4/8 schedule would keep knocking on a door that has just
#: asked us to stop.
RATE_LIMIT_BACKOFF_SECONDS: float = 60.0


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


# --------------------------------------------------------------------------
# The API client (FR-101, FR-108, FR-110, FR-111)
# --------------------------------------------------------------------------


class RateLimited(CollectionError):
    """The API asked us to slow down. Transient, and retried under FR-111."""


def _looks_rate_limited(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


class ErgastClient:
    """The project's only networked object.

    fastf1 3.8.3 reaches an Ergast-compatible backend at
    ``https://api.jolpi.ca/ergast/f1``; the dependency is transitive through
    fastf1 rather than direct, per plan D-8. Responses are requested **uncast**
    so that every conversion into the canonical schema happens in this
    repository, where it is tested, rather than inside a library whose casting
    rules can change under us.
    """

    def __init__(
        self,
        *,
        pause: float = REQUEST_PAUSE_SECONDS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        cache_dir: Path | None = FASTF1_CACHE_DIR,
        rate_limit_backoff: float = RATE_LIMIT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.pause = pause
        self.rate_limit_backoff = rate_limit_backoff
        self._sleep = sleep

        import fastf1
        import fastf1.ergast.interface as ergast_interface
        from fastf1.ergast import Ergast

        # The upstream default is 5 seconds, which the T-099 spike measured
        # tripping three times in ninety requests. It is a module constant read
        # at call time, and is the only timeout knob fastf1 exposes.
        ergast_interface.TIMEOUT = timeout

        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(str(cache_dir))

        self._fastf1 = fastf1
        self._api = Ergast(result_type="raw", auto_cast=False)

    # -- transport ----------------------------------------------------------

    def _call(self, fn: Callable[[], Any]) -> Any:
        """One request, with the politeness that a free service deserves."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transport errors are opaque
            if _looks_rate_limited(exc):
                self._sleep(self.rate_limit_backoff)
                raise RateLimited(str(exc)) from exc
            raise

    # -- endpoints ----------------------------------------------------------

    def race_result_pages(self, season: int) -> list[Any]:
        """Every page of a season's race results, asserted complete (FR-110)."""

        def fetch(limit: int, offset: int) -> Any:
            return self._call(
                lambda: self._api.get_race_results(
                    season=season, limit=limit, offset=offset
                )
            )

        return paginate(
            fetch,
            what=f"race results {season}",
            rows_in=rows_in_page,
            pause=self.pause,
        )

    def lap_positions(
        self, season: int, round_: int, lap_number: int
    ) -> dict[str, int]:
        """Driver -> track position at the end of one lap (FR-001b).

        One request per tied lap rather than a whole race's lap history. The
        endpoint is asked for a single `lap_number`, which is the only lap that
        can separate the drivers tied on it, so the corpus-wide cost is one
        request per tie group rather than one per race.

        A driver absent from the response did not complete this lap. That is
        information, not an error: it is why zero-lap retirements have no entry
        anywhere and cannot be separated by this rule at all.
        """
        self._sleep(self.pause)
        page = with_retry(
            lambda: self._call(
                lambda: self._api.get_lap_times(
                    season=season, round=round_, lap_number=lap_number, limit=PAGE_SIZE
                )
            ),
            what=f"lap {lap_number} positions {season} round {round_}",
        )
        return positions_from_lap_page(page)

    def event_formats(self, season: int) -> dict[int, str]:
        """Round -> raw weekend-format string for one season (FR-108).

        Read from fastf1's event schedule, which is the source that carries the
        format vocabulary (`conventional`, `sprint`, `sprint_shootout`,
        `sprint_qualifying`). The results endpoint does not expose it.
        """
        self._sleep(self.pause)
        schedule = with_retry(
            lambda: self._call(
                lambda: self._fastf1.get_event_schedule(
                    season, include_testing=False
                )
            ),
            what=f"event schedule {season}",
        )
        return {
            int(row.RoundNumber): str(row.EventFormat)
            for row in schedule.itertuples()
        }

    def sprint_result_pages(self, season: int) -> list[Any]:
        """Every page of a season's sprint results, asserted complete (FR-110).

        Paged for the same reason race results are. 2023 is the case that
        proves it matters: a single unpaginated request returns two of its six
        sprints and reports no error at all, which is the exact shape of the
        defect FR-110 exists to catch.

        A season with no sprints returns no pages. That is the answer for every
        season before 2021 and is not a failure.
        """

        def fetch(limit: int, offset: int) -> Any:
            return self._call(
                lambda: self._api.get_sprint_results(
                    season=season, limit=limit, offset=offset
                )
            )

        return paginate(
            fetch,
            what=f"sprint results {season}",
            rows_in=sprint_rows_in_page,
            pause=self.pause,
        )


def rows_in_page(page: Any) -> int:
    """Result rows on one page — the unit the API paginates by (FR-110).

    Counting *races* here would be wrong: the reported total counts result rows,
    and a race straddles the page boundary.
    """
    return sum(len(race.get("Results", ())) for race in page)


def sprint_rows_in_page(page: Any) -> int:
    """Sprint result rows on one page. The payload key differs from the race one."""
    return sum(len(race.get("SprintResults", ())) for race in page)


# --------------------------------------------------------------------------
# Payload -> canonical rows (FR-001..003a, FR-105, FR-108)
# --------------------------------------------------------------------------


def merge_paged_races(pages: Sequence[Any]) -> dict[int, dict[str, Any]]:
    """Reassemble whole races from row-paginated pages (FR-104a).

    Returns round number -> race payload whose `Results` hold the entire field.
    A race split across two pages appears in both, each time with part of its
    field; concatenating is the only way to get an intact race, and an intact
    race is what FR-104a requires before anything is stored.
    """
    races: dict[int, dict[str, Any]] = {}
    for page in pages:
        for race in page:
            rnd = int(race["round"])
            if rnd in races:
                races[rnd]["Results"].extend(race.get("Results", ()))
            else:
                merged = dict(race)
                merged["Results"] = list(race.get("Results", ()))
                races[rnd] = merged
    return races


def rows_from_race(
    race: Mapping[str, Any], *, weekend_format_raw: str
) -> list[dict[str, Any]]:
    """One canonical row per entry in one race.

    Nothing is inferred and nothing is defaulted: an unrecognised weekend format
    (FR-108) or position sentinel (FR-003a) raises, and the caller records the
    race as a failure rather than storing a guess.
    """
    weekend_format = map_weekend_format(weekend_format_raw)
    season = int(race["season"])
    rnd = int(race["round"])
    race_date = pd.Timestamp(str(race["date"]))
    circuit_id = str(race["Circuit"]["circuitId"])

    rows: list[dict[str, Any]] = []
    for entry in race.get("Results", ()):
        driver = entry["Driver"]
        code = driver.get("code")
        position_text = str(entry["positionText"])
        rows.append(
            {
                "season": season,
                "round": rnd,
                "race_date": race_date,
                "circuit_id": circuit_id,
                "weekend_format": weekend_format.value,
                "weekend_format_raw": str(weekend_format_raw),
                "driver_id": str(driver["driverId"]),
                "driver_code": None if code is None else str(code),
                "constructor_id": str(entry["Constructor"]["constructorId"]),
                "grid_position": int(entry["grid"]),
                "classified_position": int(entry["position"]),
                "position_text": position_text,
                # FR-003a: derived from the position sentinel, never the status
                # string. Bahrain 2022 classified two cars that had stopped.
                "classification_outcome": classify(position_text).value,
                # FR-003: the raw status verbatim. ~30 distinct values a season,
                # and FR-752's retirement-cause analysis needs all of them.
                "status_raw": str(entry["status"]),
                "laps_completed": int(entry["laps"]),
                "points": float(entry["points"]),
                # FR-001b's tie-break is fetched separately and only where a tie
                # exists (T-101e); it is deliberately absent here.
                "final_lap_position": None,
                # FR-211's sprint result likewise: a separate endpoint, fetched
                # only for the seasons that have sprints (T-212). Null here
                # means "not yet fetched"; null after that pass means "no
                # sprint happened", and only the second is a fact.
                "sprint_position": None,
                "sprint_grid_position": None,
            }
        )
    if not rows:
        raise CollectionError(
            f"{season} r{rnd}: no result rows in the payload. An empty race is "
            f"not a small race (FR-104a)."
        )
    return rows


def frame_from_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Cast canonical rows to the schema's dtypes and validate them."""
    if not rows:
        return empty_frame()
    df = pd.DataFrame(list(rows))
    df["race_date"] = pd.to_datetime(df["race_date"])
    for col, dtype in RACE_ROW_SCHEMA.items():
        if col != "race_date":
            df[col] = df[col].astype(dtype)
    return validate_frame(df)


def positions_from_lap_page(page: Any) -> dict[str, int]:
    """Parse the lap-times payload into driver -> position for that lap.

    Shape: a list of races, each carrying `Laps`, each carrying `Timings`. Only
    one lap was requested, so anything beyond it is upstream disagreeing with
    the query and is read rather than assumed.
    """
    out: dict[str, int] = {}
    for race in page or ():
        for lap in race.get("Laps", ()) or ():
            for timing in lap.get("Timings", ()) or ():
                driver = timing.get("driverId")
                position = timing.get("position")
                if driver is None or position is None:
                    continue
                out[str(driver)] = int(position)
    return out


#: Retirements tied on this many completed laps cannot be separated by FR-001b:
#: a driver who completed no lap appears in no lap's timing data. See the
#: FR-001c note in spec §3.0.
ZERO_LAPS: int = 0


def tie_groups_needing_lap_position(
    df: pd.DataFrame,
) -> list[tuple[int, int, int]]:
    """`(season, round, lap)` for every tie FR-001b can actually resolve.

    Zero-lap ties are excluded deliberately rather than attempted and failed:
    there is no lap to ask about, so a request would be spent to learn nothing.
    """
    starters, _ = drop_non_starters(df)
    retired = starters[
        starters["classification_outcome"]
        == ClassificationOutcome.NOT_CLASSIFIED.value
    ]
    if retired.empty:
        return []

    groups: list[tuple[int, int, int]] = []
    for (season, rnd), race in retired.groupby(["season", "round"], sort=True):
        counts = race["laps_completed"].value_counts()
        for lap, n in counts.items():
            if n > 1 and int(lap) > ZERO_LAPS:
                groups.append((int(season), int(rnd), int(lap)))
    return sorted(groups)


@dataclass(frozen=True, slots=True)
class TieBreakReport:
    """What the FR-001b pass could and could not resolve."""

    groups_attempted: int
    rows_resolved: int
    rows_unresolved_zero_lap: int
    rows_unresolved_absent: int
    failed_groups: tuple[tuple[int, int, int], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "groups_attempted": self.groups_attempted,
            "rows_resolved": self.rows_resolved,
            "rows_unresolved_zero_lap": self.rows_unresolved_zero_lap,
            "rows_unresolved_absent": self.rows_unresolved_absent,
            "failed_groups": [list(g) for g in self.failed_groups],
        }


def add_final_lap_positions(
    df: pd.DataFrame, *, client: Any, log: Callable[[str], None] = print
) -> tuple[pd.DataFrame, TieBreakReport]:
    """Fill `final_lap_position` for retirements FR-001b can separate.

    Only tied rows are touched. An untied retirement is already ordered by laps
    completed, so a position for it would be data the target never reads.

    A group whose fetch fails leaves its rows null rather than half-filled: the
    grid fallback is a declared, flagged state, and a partially resolved tie
    would be an undeclared one.
    """
    out = df.copy()
    if FINAL_LAP_POSITION_COLUMN not in out.columns:
        out[FINAL_LAP_POSITION_COLUMN] = pd.NA
    out[FINAL_LAP_POSITION_COLUMN] = out[FINAL_LAP_POSITION_COLUMN].astype("Int64")

    groups = tie_groups_needing_lap_position(out)
    resolved = absent = 0
    failed: list[tuple[int, int, int]] = []

    for season, rnd, lap in groups:
        try:
            positions = client.lap_positions(season, rnd, lap)
        except Exception as exc:  # noqa: BLE001 - a tie is not worth aborting a run
            failed.append((season, rnd, lap))
            log(f"  tie {season} r{rnd} lap {lap}: fetch failed ({exc}); left to grid fallback")
            continue

        mask = (
            (out["season"] == season)
            & (out["round"] == rnd)
            & (out["laps_completed"] == lap)
        )
        for idx in out.index[mask]:
            driver = out.at[idx, "driver_id"]
            if driver in positions:
                out.at[idx, FINAL_LAP_POSITION_COLUMN] = positions[driver]
                resolved += 1
            else:
                absent += 1

    starters, _ = drop_non_starters(out)
    retired = starters[
        starters["classification_outcome"]
        == ClassificationOutcome.NOT_CLASSIFIED.value
    ]
    zero_lap_tied = 0
    for _, race in retired.groupby(["season", "round"], sort=True):
        counts = race["laps_completed"].value_counts()
        zero_lap_tied += int(counts.get(ZERO_LAPS, 0)) if counts.get(ZERO_LAPS, 0) > 1 else 0

    return out, TieBreakReport(
        groups_attempted=len(groups),
        rows_resolved=resolved,
        rows_unresolved_zero_lap=zero_lap_tied,
        rows_unresolved_absent=absent,
        failed_groups=tuple(failed),
    )


# --------------------------------------------------------------------------
# Sprint results (FR-211, T-212)
# --------------------------------------------------------------------------

SPRINT_POSITION_COLUMN: Final[str] = "sprint_position"
SPRINT_GRID_COLUMN: Final[str] = "sprint_grid_position"


@dataclass(frozen=True, slots=True)
class SprintReport:
    """What the sprint pass found. FR-211's nulls are counted, not assumed."""

    seasons_attempted: tuple[int, ...] = ()
    sprints_fetched: int = 0
    rows_filled: int = 0
    #: Rows on a sprint weekend that the sprint classification does not cover —
    #: a car that started the Grand Prix but not the sprint, or the reverse.
    rows_absent_from_sprint: int = 0
    #: Rounds the corpus marks as sprint weekends but the API has no sprint for,
    #: and the reverse. Either direction is a disagreement between FR-108's
    #: format mapping and the results endpoint, and is reported rather than
    #: reconciled.
    rounds_missing_results: tuple[tuple[int, int], ...] = ()
    rounds_unexpected: tuple[tuple[int, int], ...] = ()
    failed_seasons: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "seasons_attempted": list(self.seasons_attempted),
            "sprints_fetched": self.sprints_fetched,
            "rows_filled": self.rows_filled,
            "rows_absent_from_sprint": self.rows_absent_from_sprint,
            "rounds_missing_results": [list(r) for r in self.rounds_missing_results],
            "rounds_unexpected": [list(r) for r in self.rounds_unexpected],
            "failed_seasons": list(self.failed_seasons),
        }

    def summary(self) -> str:
        return (
            f"sprint results: {self.sprints_fetched} sprints, "
            f"{self.rows_filled} rows filled, "
            f"{self.rows_absent_from_sprint} rows on a sprint weekend with no "
            f"sprint classification"
        )


def sprint_rounds_expected(df: pd.DataFrame) -> set[tuple[int, int]]:
    """(season, round) the corpus marks as a sprint weekend, from FR-108's format.

    Read from the declared format rather than from the results endpoint, so the
    two can be compared. Agreement is worth knowing; disagreement is worth
    knowing more.
    """
    sprint_formats = {
        WeekendFormat.SPRINT_SETS_GRID.value,
        WeekendFormat.SPRINT_STANDALONE.value,
    }
    rows = df[df["weekend_format"].isin(sprint_formats)]
    return {(int(s), int(r)) for s, r in zip(rows["season"], rows["round"])}


def merge_paged_sprints(pages: Sequence[Any]) -> dict[int, list[Mapping[str, Any]]]:
    """Reassemble whole sprints from row-paginated pages (FR-104a, FR-110).

    Same hazard as the race endpoint: the API paginates by result row, so a
    sprint whose field straddles a page boundary arrives in two halves.
    """
    sprints: dict[int, list[Mapping[str, Any]]] = {}
    for page in pages:
        for race in page:
            sprints.setdefault(int(race["round"]), []).extend(
                race.get("SprintResults", ())
            )
    return sprints


def add_sprint_results(
    df: pd.DataFrame, *, client: Any, log: Callable[[str], None] = print
) -> tuple[pd.DataFrame, SprintReport]:
    """Fill the sprint columns for weekends that had a sprint (FR-211).

    Only seasons the corpus already marks as containing a sprint are fetched,
    so the pass costs a handful of requests rather than one per season. Every
    other row keeps its null, and the null is the point: FR-211 forbids
    imputing a neutral value for a sprint that never happened, because a
    neutral value is a claim and an absent sprint is not.

    A season whose fetch fails leaves its rows null and is recorded. Half a
    season of sprint results is worse than none, since nothing downstream could
    tell the two apart.
    """
    out = df.copy()
    for column in (SPRINT_POSITION_COLUMN, SPRINT_GRID_COLUMN):
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = out[column].astype("Int64")

    expected = sprint_rounds_expected(out)
    seasons = sorted({season for season, _ in expected})
    if not seasons:
        return out, SprintReport()

    index = {
        (int(s), int(r), str(d)): idx
        for idx, (s, r, d) in enumerate(
            zip(out["season"], out["round"], out["driver_id"])
        )
    }

    fetched = filled = 0
    seen: set[tuple[int, int]] = set()
    failed: list[int] = []

    for season in seasons:
        try:
            pages = client.sprint_result_pages(season)
        except Exception as exc:  # noqa: BLE001 - one season is not the corpus
            failed.append(season)
            log(f"  sprint results {season}: fetch failed ({exc}); rows left null")
            continue

        sprints = merge_paged_sprints(pages)
        fetched += len(sprints)
        for rnd, results in sprints.items():
            seen.add((season, rnd))
            for entry in results:
                driver = str(entry["Driver"]["driverId"])
                idx = index.get((season, rnd, driver))
                if idx is None:
                    # Contested the sprint but not the Grand Prix. Nothing in
                    # the corpus to attach it to.
                    continue
                out.iat[idx, out.columns.get_loc(SPRINT_POSITION_COLUMN)] = int(
                    entry["position"]
                )
                out.iat[idx, out.columns.get_loc(SPRINT_GRID_COLUMN)] = int(
                    entry["grid"]
                )
                filled += 1
        log(f"  sprint results {season}: {len(sprints)} sprint(s)")

    on_sprint_weekend = out.apply(
        lambda row: (int(row["season"]), int(row["round"])) in expected, axis=1
    )
    absent = int(
        (on_sprint_weekend & out[SPRINT_POSITION_COLUMN].isna()).sum()
    )

    return out, SprintReport(
        seasons_attempted=tuple(seasons),
        sprints_fetched=fetched,
        rows_filled=filled,
        rows_absent_from_sprint=absent,
        rounds_missing_results=tuple(sorted(expected - seen)),
        rounds_unexpected=tuple(sorted(seen - expected)),
        failed_seasons=tuple(failed),
    )


# --------------------------------------------------------------------------
# Season and corpus collection (FR-101..106)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeasonReport:
    season: int
    rows: int
    races: int
    #: Rounds the schedule listed. `None` for a season served from a checkpoint,
    #: where the schedule was deliberately not refetched (FR-102).
    scheduled_races: int | None
    failed_rounds: tuple[int, ...]
    from_checkpoint: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "season": self.season,
            "rows": self.rows,
            "races": self.races,
            "scheduled_races": self.scheduled_races,
            "failed_rounds": list(self.failed_rounds),
            "from_checkpoint": self.from_checkpoint,
        }


def collect_season(
    season: int, *, client: Any, manifest: FailureManifest
) -> tuple[pd.DataFrame, SeasonReport]:
    """Collect one whole season into canonical rows.

    A race that cannot be built is excluded whole and recorded as a gap
    (FR-104a, FR-104b); the threshold is checked after every race so that a run
    which has stopped being trustworthy stops (FR-104c).
    """
    formats = client.event_formats(season)
    races = merge_paged_races(client.race_result_pages(season))

    frames: list[pd.DataFrame] = []
    failed: list[int] = []

    for rnd in sorted(races):
        race = races[rnd]
        race_date = str(race.get("date")) or None
        raw_format = formats.get(rnd)
        try:
            if raw_format is None:
                raise SchemaError(
                    f"round {rnd} is absent from the {season} event schedule, so "
                    f"its weekend format is unknown. FR-108 makes that an error "
                    f"rather than a silent CONVENTIONAL."
                )
            frames.append(
                frame_from_rows(rows_from_race(race, weekend_format_raw=raw_format))
            )
        except Exception as exc:  # noqa: BLE001 - any defect fails the race whole
            failed.append(rnd)
            manifest.record_failure(
                season, rnd, f"{type(exc).__name__}: {exc}", race_date
            )
        else:
            manifest.record_success()

    # A scheduled round that returned no results at all is a gap too — it never
    # reached the loop above, so nothing else would have noticed it.
    for rnd in sorted(set(formats) - set(races)):
        failed.append(rnd)
        manifest.record_failure(
            season, rnd, "scheduled round returned no race results"
        )

    # FR-104c is checked per season, not per race: after one race the rate is
    # either 0% or 100%, so a per-race check would abort on the first failure
    # whatever the threshold said, and the threshold would mean nothing. The
    # manifest accumulates across seasons, so the across-the-run rate is
    # checked here too.
    manifest.check_threshold()

    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else empty_frame()
    )
    df = validate_frame(df)
    return df, SeasonReport(
        season=season,
        rows=len(df),
        races=int(df["round"].nunique()) if len(df) else 0,
        scheduled_races=len(formats),
        failed_rounds=tuple(sorted(failed)),
    )


def collect_corpus(
    seasons: Sequence[int],
    *,
    root: Any,
    client: Any,
    manifest: FailureManifest | None = None,
    log: Callable[[str], None] = print,
) -> tuple[list[SeasonReport], FailureManifest]:
    """Collect every season that has no checkpoint yet (FR-101..103).

    Resumption is the default rather than a flag: an interrupted run is
    restarted by running the same command again.
    """
    manifest = manifest if manifest is not None else FailureManifest()
    pending = set(seasons_to_collect(root, seasons))
    reports: list[SeasonReport] = []

    for season in seasons:
        if season not in pending:
            existing = read_checkpoint(root, season)
            assert existing is not None
            reports.append(
                SeasonReport(
                    season=season,
                    rows=len(existing),
                    races=int(existing["round"].nunique()) if len(existing) else 0,
                    scheduled_races=None,
                    failed_rounds=(),
                    from_checkpoint=True,
                )
            )
            log(f"{season}: checkpoint present, not refetched (FR-102)")
            continue

        df, report = collect_season(season, client=client, manifest=manifest)
        write_checkpoint(df, root, season)
        reports.append(report)
        log(
            f"{season}: {report.rows} rows across {report.races} races "
            f"(schedule says {report.scheduled_races})"
            + (f" — failed rounds {list(report.failed_rounds)}" if report.failed_rounds else "")
        )

    return reports, manifest


def write_failure_manifest(manifest: FailureManifest, root: Any) -> Path:
    """FR-104b: the manifest travels with the data."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "failure_manifest.json"
    path.write_text(json.dumps(manifest.as_dict(), indent=2) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Run collection over the corpus window (T-106).

    Deliberately minimal: the full command surface is FR-801, built in P6. This
    exists so that the collection procedure Constitution I.3 requires to be
    documented is a command in the repository rather than a scratch script.
    """
    from f1pred.config import RAW_DIR
    from f1pred.corpus import default_window

    window = default_window()
    seasons = window.resolve()
    print(f"corpus window: W={window.width} -> {seasons[0]}..{seasons[-1]}")

    client = ErgastClient()
    reports, manifest = collect_corpus(seasons, root=RAW_DIR, client=client)

    frames = [
        df for df in (read_checkpoint(RAW_DIR, s) for s in seasons) if df is not None
    ]
    corpus = pd.concat(frames, ignore_index=True) if frames else empty_frame()
    for count in check_season_counts(corpus):
        warning = count.warning()
        if warning:
            print(f"FR-106 warning: {warning}")

    print(manifest.summary())
    print(f"manifest written to {write_failure_manifest(manifest, RAW_DIR)}")
    print(f"{len(corpus)} rows across {len(reports)} seasons")
    return 0


__all__ = [
    "PAGE_SIZE",
    "RATE_LIMIT_BACKOFF_SECONDS",
    "REQUEST_PAUSE_SECONDS",
    "CollectionError",
    "ErgastClient",
    "FailureManifest",
    "FailureThresholdExceeded",
    "IncompletePageError",
    "RaceFailure",
    "RateLimited",
    "SeasonCount",
    "SeasonReport",
    "TieBreakReport",
    "ZERO_LAPS",
    "add_final_lap_positions",
    "check_season_counts",
    "checkpoint_path",
    "collect_corpus",
    "collect_season",
    "completed_seasons",
    "frame_from_rows",
    "iter_pending",
    "main",
    "merge_paged_races",
    "paginate",
    "positions_from_lap_page",
    "read_checkpoint",
    "rows_from_race",
    "rows_in_page",
    "seasons_to_collect",
    "tie_groups_needing_lap_position",
    "with_retry",
    "write_checkpoint",
    "write_failure_manifest",
]


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
