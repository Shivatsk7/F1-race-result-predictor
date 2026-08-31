"""Practice-session pace and tyre-stint data (FR-213, FR-214, T-214).

This is the project's second networked module, and the only other one (NFR-05).

**Why practice at all.** Qualifying tells you a car's peak over one lap with no
fuel; the race is decided by pace over a stint. Friday practice is the only
place in the weekend where long runs are visible *before* the race, which is
what makes this admissible under Constitution III.1 at all.

**Why it is a separate arm and not a feature.** The source begins in 2018 and
the corpus begins in 2010, so these columns are null for 3,453 of the corpus's
6,911 rows — very nearly half. FR-214 forbids imputing that away and FR-215
therefore keeps the whole group out of the primary set. A feature that is
absent for half your data is not a weak feature, it is two different datasets
wearing one name.

**What is extracted, and what is thrown away.** Only green-flag, non-deleted,
non-in/out laps inside a stint long enough to be a long run. A practice session
is a mixture of qualifying simulations, aero rakes, tyre scrubbing, and traffic;
taking a mean over all of it would measure the team's programme rather than the
car's pace.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from f1pred.config import FASTF1_CACHE_DIR, RAW_DIR

logger = logging.getLogger(__name__)

#: FR-214: the telemetry source's first season. Earlier rows are null, and the
#: null is a fact about the source, never a gap to be filled.
TELEMETRY_START_SEASON: Final[int] = 2018

#: The upstream timing API allows 500 calls an hour and a single session load
#: costs about six of them, so roughly 83 sessions an hour is the ceiling. This
#: is a wall, not a throttle: exceeding it returns errors rather than slow
#: responses, and no amount of parallelism moves it. Measured 2026-08-31 by
#: hitting it — 77 sessions succeeded, then 83 consecutive failures.
RATE_LIMIT_CALLS_PER_HOUR: Final[int] = 500
API_CALLS_PER_SESSION: Final[int] = 6

#: Sleep this long when the limit is hit, then resume. Long because the limit is
#: hourly and a shorter wait simply fails again.
RATE_LIMIT_SLEEP_SECONDS: Final[float] = 600.0

#: Preference order for the one session read per weekend. FP2 first, and not
#: only to save requests: on a conventional weekend FP2 is *the* long-run
#: session, run at the race's time of day on representative fuel. FP1 is aero
#: rakes and rookie runs, FP3 is qualifying preparation. Reading all three
#: triples the cost to add mostly the laps this module already filters out.
#: A sprint weekend holds only FP1, so the fallback is not optional.
PREFERRED_PRACTICE_ORDER: Final[tuple[str, ...]] = (
    "Practice 2",
    "Practice 1",
    "Practice 3",
)

#: A stint must reach this many usable laps before it counts as a long run.
#: Below five, a "stint" is an out-lap, a push lap, and an in-lap — which is a
#: qualifying simulation, the opposite of what this measures. Declared rather
#: than tuned: tuning it against race results would be fitting the feature to
#: the outcome.
LONG_RUN_MIN_LAPS: Final[int] = 5

#: A lap slower than this multiple of the driver's own best is traffic, a lift,
#: or a cool-down, not pace. Generous on purpose — the filter exists to remove
#: obvious non-laps, not to sculpt the distribution.
MAX_LAP_RATIO: Final[float] = 1.10

#: fastf1 concatenates a track-status code per marshalling sector, so a lap's
#: value is a string like "12" meaning it saw both all-clear and yellow. These
#: are the codes that actually stop a driver setting a representative time:
#: safety car, red flag, and virtual safety car. A yellow is deliberately NOT
#: here — it is usually in a sector the driver is nowhere near, and excluding
#: every lap that saw one costs roughly half the usable data at a busy session.
#: A yellow that did cost the driver time shows up as a slow lap, which
#: MAX_LAP_RATIO already removes.
DISRUPTIVE_TRACK_STATUS: Final[tuple[str, ...]] = ("4", "5", "6", "7")

PRACTICE_ROW_SCHEMA: Final[dict[str, str]] = {
    "season": "int64",
    "round": "int64",
    "driver_code": "string",
    #: How many of the weekend's practice sessions this driver appeared in.
    "practice_sessions": "int64",
    #: Usable green laps, after every filter above.
    "practice_laps": "int64",
    #: Median long-run lap time in seconds. The headline FR-213 quantity.
    "long_run_pace_s": "float64",
    "long_run_laps": "int64",
    "best_lap_s": "float64",
    #: Seconds per lap of tyre life inside long runs — FR-213's "tyre-stint
    #: characteristics". Positive means the car slowed as the tyre aged.
    #:
    #: Deliberately NOT called degradation. Within a stint three things move a
    #: lap time: the tyre wearing out (slower), the fuel load burning off
    #: (faster), and the track rubbering in (faster). This slope is their sum,
    #: and on a green Friday the last two usually win, which is why the value is
    #: frequently negative. Naming it degradation would invite a reader to
    #: conclude that a car with a negative number has tyres that improve with
    #: age. Its use is comparative: fuel burn and track evolution hit every car
    #: on the same day about equally, so the *ranking* of this number within a
    #: weekend does carry tyre information even though its level does not.
    #: FR-209 makes the feature relative anyway, which is what makes it usable.
    "stint_pace_trend_s_per_lap": "float64",
    "longest_stint_laps": "int64",
    "stints": "int64",
}

#: Columns that carry a measurement and may legitimately be missing when a
#: session gave no usable long run — a wet Friday, or a red flag.
NULLABLE_MEASUREMENTS: Final[frozenset[str]] = frozenset(
    {"long_run_pace_s", "best_lap_s", "stint_pace_trend_s_per_lap"}
)


class TelemetryError(RuntimeError):
    """Raised when practice data cannot be read at all."""


def _is_rate_limit(exc: BaseException) -> bool:
    """Recognise the hourly cap. Matched on text: the type is not exported."""
    text = f"{type(exc).__name__} {exc}".lower()
    return "ratelimit" in text or "rate limit" in text or "429" in text


# --------------------------------------------------------------------------
# The client — the only part that touches the network
# --------------------------------------------------------------------------


class PracticeClient:
    """Reads practice sessions through fastf1, cached on disk.

    Separated behind a small surface for the same reason `ErgastClient` is:
    everything downstream is then testable offline against a recorded frame,
    and the network lives in one object that tests do not construct.
    """

    def __init__(
        self,
        cache_dir: Path | None = FASTF1_CACHE_DIR,
        *,
        rate_limit_sleep: float = RATE_LIMIT_SLEEP_SECONDS,
        rate_limit_retries: int = 6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import fastf1

        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(str(cache_dir))
        self._fastf1 = fastf1
        self.rate_limit_sleep = rate_limit_sleep
        self.rate_limit_retries = rate_limit_retries
        self._sleep = sleep

    def practice_names(
        self, season: int, *, all_sessions: bool = False
    ) -> dict[int, tuple[str, ...]]:
        """Round -> the practice sessions to read that weekend.

        Read from the schedule rather than assumed, because the number of
        practice sessions is not constant: a sprint weekend has one, a
        conventional weekend three, and the rules changed twice inside the
        covered seasons.

        By default this returns the single best available session per weekend,
        per `PREFERRED_PRACTICE_ORDER`. `all_sessions=True` returns every one,
        which is a three-times more expensive read against a hard hourly rate
        limit — worth it only to check what the cheaper policy gives up.
        """
        schedule = self._fastf1.get_event_schedule(season, include_testing=False)
        out: dict[int, tuple[str, ...]] = {}
        for row in schedule.itertuples():
            names = [getattr(row, f"Session{i}", None) for i in range(1, 6)]
            held = [n for n in names if isinstance(n, str) and n.startswith("Practice")]
            if not held:
                continue
            if all_sessions:
                out[int(row.RoundNumber)] = tuple(held)
                continue
            chosen = next((n for n in PREFERRED_PRACTICE_ORDER if n in held), held[0])
            out[int(row.RoundNumber)] = (chosen,)
        return out

    def session_laps(self, season: int, round_: int, name: str) -> pd.DataFrame:
        """Lap records for one practice session. Car telemetry is not loaded.

        The lap table already carries compound, tyre life, stint number, and an
        accuracy flag, so pulling per-sample car data would multiply the
        download by two orders of magnitude to learn nothing more.

        A rate-limit refusal is waited out rather than recorded as a failure.
        The distinction matters: a session skipped because the hour's quota ran
        out is not a session with no data, and treating the two alike would put
        a hole in the coverage report that no amount of re-running would close.
        """
        for attempt in range(self.rate_limit_retries + 1):
            try:
                session = self._fastf1.get_session(season, round_, name)
                session.load(
                    laps=True, telemetry=False, weather=False, messages=False
                )
                return session.laps
            except Exception as exc:  # noqa: BLE001 - upstream types are opaque
                if not _is_rate_limit(exc) or attempt == self.rate_limit_retries:
                    raise
                logger.warning(
                    "rate limited at %s r%s %s; sleeping %.0fs",
                    season, round_, name, self.rate_limit_sleep,
                )
                self._sleep(self.rate_limit_sleep)
        raise TelemetryError("unreachable")


# --------------------------------------------------------------------------
# Lap filtering and per-driver aggregation
# --------------------------------------------------------------------------


def usable_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag laps that represent pace rather than a team's programme.

    Drops, in order: laps fastf1 itself flags inaccurate, deleted laps, in- and
    out-laps, anything run under a flag, and laps far off the driver's own best.
    Each of these is a lap the car was not trying to go quickly on, and leaving
    them in would make a team that ran more aero tests look slower.
    """
    if laps.empty:
        return laps.iloc[0:0]

    work = laps.copy()
    keep = pd.Series(True, index=work.index)

    if "IsAccurate" in work:
        keep &= work["IsAccurate"].fillna(False).astype(bool)
    if "Deleted" in work:
        keep &= ~work["Deleted"].fillna(False).astype(bool)
    for column in ("PitInTime", "PitOutTime"):
        if column in work:
            keep &= work[column].isna()
    if "TrackStatus" in work:
        status = work["TrackStatus"].astype("string").fillna("")
        disrupted = pd.Series(False, index=work.index)
        for code in DISRUPTIVE_TRACK_STATUS:
            disrupted |= status.str.contains(code, regex=False)
        keep &= ~disrupted
    if "LapTime" in work:
        keep &= work["LapTime"].notna()

    work = work[keep]
    if work.empty:
        return work

    work["lap_seconds"] = work["LapTime"].dt.total_seconds()
    best = work.groupby("Driver", sort=False)["lap_seconds"].transform("min")
    return work[work["lap_seconds"] <= best * MAX_LAP_RATIO]


def _stint_pace_slope(stint_laps: Iterable[tuple[np.ndarray, np.ndarray]]) -> float:
    """Seconds per lap of tyre life, pooled across stints. Not degradation.

    Each stint is centred on its own mean before pooling, so a slow stint on
    heavy fuel and a fast one on low fuel contribute their *shape* and not their
    level. Without the centring this would mostly measure fuel burn.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for life, seconds in stint_laps:
        if len(life) < 2:
            continue
        xs.append(life - life.mean())
        ys.append(seconds - seconds.mean())
    if not xs:
        return float("nan")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    variance = float((x * x).sum())
    if variance == 0.0:
        return float("nan")
    return float((x * y).sum() / variance)


def aggregate_driver(laps: pd.DataFrame) -> dict[str, Any]:
    """Collapse one driver's usable laps across a weekend into FR-213's metrics."""
    result: dict[str, Any] = {
        "practice_laps": int(len(laps)),
        "best_lap_s": float(laps["lap_seconds"].min()) if len(laps) else float("nan"),
        "long_run_pace_s": float("nan"),
        "long_run_laps": 0,
        "stint_pace_trend_s_per_lap": float("nan"),
        "longest_stint_laps": 0,
        "stints": 0,
    }
    if laps.empty or "Stint" not in laps:
        return result

    # A stint is identified within its own session: stint 1 of FP1 and stint 1
    # of FP2 are different runs on different tyres.
    grouped = laps.groupby(["session_name", "Stint"], dropna=False, sort=True)
    result["stints"] = int(grouped.ngroups)
    result["longest_stint_laps"] = int(grouped.size().max())

    long_runs = [g for _, g in grouped if len(g) >= LONG_RUN_MIN_LAPS]
    if not long_runs:
        return result

    pooled = pd.concat(long_runs)
    result["long_run_pace_s"] = float(pooled["lap_seconds"].median())
    result["long_run_laps"] = int(len(pooled))
    if "TyreLife" in pooled:
        result["stint_pace_trend_s_per_lap"] = _stint_pace_slope(
            (
                (
                    g["TyreLife"].to_numpy(dtype="float64"),
                    g["lap_seconds"].to_numpy(dtype="float64"),
                )
                for g in long_runs
                if g["TyreLife"].notna().all()
            )
        )
    return result


def weekend_rows(
    season: int, round_: int, sessions: dict[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    """One row per driver who ran in any of a weekend's practice sessions."""
    frames = []
    for name, laps in sessions.items():
        cleaned = usable_laps(laps)
        if cleaned.empty:
            continue
        cleaned = cleaned.copy()
        cleaned["session_name"] = name
        frames.append(cleaned)

    appearances: dict[str, set[str]] = {}
    for name, laps in sessions.items():
        if "Driver" in laps:
            for driver in laps["Driver"].dropna().unique():
                appearances.setdefault(str(driver), set()).add(name)

    if not frames:
        # The weekend happened but produced no usable lap — a washed-out Friday.
        # The drivers are still recorded, with null measurements, because
        # "ran and set no representative lap" is not the same as "no data".
        return [
            {
                "season": season,
                "round": round_,
                "driver_code": driver,
                "practice_sessions": len(names),
                **aggregate_driver(pd.DataFrame(columns=["lap_seconds"])),
            }
            for driver, names in sorted(appearances.items())
        ]

    pooled = pd.concat(frames, ignore_index=True)
    rows = []
    for driver, laps in pooled.groupby("Driver", sort=True):
        rows.append(
            {
                "season": season,
                "round": round_,
                "driver_code": str(driver),
                "practice_sessions": len(appearances.get(str(driver), ())),
                **aggregate_driver(laps),
            }
        )
    return rows


def frame_from_rows(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Rows -> a frame with the declared practice schema and dtypes."""
    if not rows:
        return pd.DataFrame(
            {c: pd.Series(dtype=d) for c, d in PRACTICE_ROW_SCHEMA.items()}
        )
    df = pd.DataFrame(list(rows))
    for column, dtype in PRACTICE_ROW_SCHEMA.items():
        if column not in df:
            df[column] = pd.NA
        df[column] = df[column].astype(dtype)
    return df[list(PRACTICE_ROW_SCHEMA)].sort_values(
        ["season", "round", "driver_code"], kind="stable"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Season collection and coverage (FR-214)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PracticeSeasonReport:
    season: int
    rounds_scheduled: int
    rounds_collected: int
    sessions_read: int
    sessions_failed: tuple[tuple[int, str], ...]
    rows: int

    def as_dict(self) -> dict[str, object]:
        return {
            "season": self.season,
            "rounds_scheduled": self.rounds_scheduled,
            "rounds_collected": self.rounds_collected,
            "sessions_read": self.sessions_read,
            "sessions_failed": [list(f) for f in self.sessions_failed],
            "rows": self.rows,
        }


def practice_checkpoint_path(root: Path, season: int) -> Path:
    return Path(root) / f"practice_{season}.parquet"


def write_practice_checkpoint(df: pd.DataFrame, root: Path, season: int) -> Path:
    """Atomic per-season checkpoint, mirroring FR-102's race collection."""
    path = practice_checkpoint_path(root, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    df.to_parquet(temp, index=False)
    temp.replace(path)
    return path


def read_practice_checkpoint(root: Path, season: int) -> pd.DataFrame | None:
    path = practice_checkpoint_path(root, season)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def collect_practice_season(
    season: int,
    *,
    client: Any,
    log: Callable[[str], None] = logger.info,
) -> tuple[pd.DataFrame, PracticeSeasonReport]:
    """Every practice session of one season, tolerant of individual failures.

    A session that will not load is recorded and skipped; the weekend keeps
    whatever its other sessions gave. That is deliberately weaker than
    FR-104a's all-or-nothing rule for race results, and it is safe here for a
    reason: these features never enter the primary set, so a thinner weekend
    degrades a side experiment rather than corrupting the headline answer.
    """
    names_by_round = client.practice_names(season)
    rows: list[dict[str, Any]] = []
    read = 0
    failed: list[tuple[int, str]] = []

    for round_ in sorted(names_by_round):
        sessions: dict[str, pd.DataFrame] = {}
        for name in names_by_round[round_]:
            try:
                sessions[name] = client.session_laps(season, round_, name)
                read += 1
            except Exception as exc:  # noqa: BLE001 - one session is not a season
                failed.append((round_, f"{name}: {type(exc).__name__}: {exc}"[:160]))
        if sessions:
            rows.extend(weekend_rows(season, round_, sessions))

    frame = frame_from_rows(rows)
    report = PracticeSeasonReport(
        season=season,
        rounds_scheduled=len(names_by_round),
        rounds_collected=int(frame[["season", "round"]].drop_duplicates().shape[0]),
        sessions_read=read,
        sessions_failed=tuple(failed),
        rows=len(frame),
    )
    log(f"practice {season}: {report.rows} rows over {report.rounds_collected} rounds")
    return frame, report


def collect_practice_corpus(
    seasons: Sequence[int],
    *,
    client: Any,
    root: Path = RAW_DIR,
    log: Callable[[str], None] = logger.info,
) -> tuple[pd.DataFrame, list[PracticeSeasonReport]]:
    """Collect the covered seasons, checkpointing each, resuming from disk."""
    frames: list[pd.DataFrame] = []
    reports: list[PracticeSeasonReport] = []

    for season in seasons:
        if season < TELEMETRY_START_SEASON:
            raise TelemetryError(
                f"season {season} precedes the telemetry source's first season "
                f"({TELEMETRY_START_SEASON}). FR-214 makes such a season null, "
                f"not collected — asking for it is a bug in the caller."
            )
        existing = read_practice_checkpoint(root, season)
        if existing is not None:
            frames.append(existing)
            reports.append(
                PracticeSeasonReport(
                    season, -1, int(existing[["season", "round"]].drop_duplicates().shape[0]),
                    0, (), len(existing),
                )
            )
            log(f"practice {season}: served from checkpoint")
            continue
        frame, report = collect_practice_season(season, client=client, log=log)
        write_practice_checkpoint(frame, root, season)
        frames.append(frame)
        reports.append(report)

    combined = (
        pd.concat(frames, ignore_index=True) if frames else frame_from_rows([])
    )
    return combined, reports


def coverage(corpus: pd.DataFrame, practice: pd.DataFrame) -> pd.DataFrame:
    """Per-season telemetry coverage, as FR-214 requires it be reported.

    Coverage is stated as a share of *corpus rows*, not of practice rows: the
    question FR-215 needs answered is how much of the modelled data this arm can
    speak about, and a season with abundant practice data the corpus does not
    contain is worth nothing.
    """
    keys = ["season", "round", "driver_code"]
    marked = corpus[keys].merge(
        practice[keys].assign(_covered=True), on=keys, how="left"
    )
    # `notna` rather than `fillna(False)`: the merge leaves NaN for an
    # uncovered row, and filling an object column triggers a downcast warning.
    marked["_covered"] = marked["_covered"].notna()
    out = (
        marked.groupby("season")["_covered"]
        .agg(corpus_rows="size", covered_rows="sum")
        .reset_index()
    )
    out["covered_rows"] = out["covered_rows"].astype("int64")
    out["coverage"] = (out["covered_rows"] / out["corpus_rows"]).round(4)
    out["source_covers_season"] = out["season"] >= TELEMETRY_START_SEASON
    return out


def attach(corpus: pd.DataFrame, practice: pd.DataFrame) -> pd.DataFrame:
    """Left-join practice measurements onto the corpus, filling nothing.

    A left join is the whole of FR-214's compliance: every row the practice
    frame does not reach keeps a null, and no code path anywhere converts that
    null into a number.
    """
    keys = ["season", "round", "driver_code"]
    measurements = [c for c in PRACTICE_ROW_SCHEMA if c not in keys]
    merged = corpus.merge(practice[keys + measurements], on=keys, how="left")
    if len(merged) != len(corpus):
        raise TelemetryError(
            f"attaching practice data changed the row count "
            f"({len(corpus)} -> {len(merged)}); the practice frame has "
            f"duplicate {keys} and the join fanned out"
        )
    return merged


__all__ = [
    "API_CALLS_PER_SESSION",
    "LONG_RUN_MIN_LAPS",
    "MAX_LAP_RATIO",
    "NULLABLE_MEASUREMENTS",
    "PRACTICE_ROW_SCHEMA",
    "PracticeClient",
    "PREFERRED_PRACTICE_ORDER",
    "PracticeSeasonReport",
    "RATE_LIMIT_CALLS_PER_HOUR",
    "TELEMETRY_START_SEASON",
    "TelemetryError",
    "aggregate_driver",
    "attach",
    "collect_practice_corpus",
    "collect_practice_season",
    "coverage",
    "frame_from_rows",
    "practice_checkpoint_path",
    "read_practice_checkpoint",
    "usable_laps",
    "weekend_rows",
    "write_practice_checkpoint",
]
