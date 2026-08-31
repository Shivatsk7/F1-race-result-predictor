"""Weather: a pre-race prior and a post-race observation (FR-206, FR-751a, T-218).

**The hazard this module is built around.** One endpoint serves both roles.
Ask the ERA5 archive for the week before a race and you get a climatological
prior, which Constitution III.1 admits. Ask it for the race day and you get
what actually happened, which Constitution III.2 forbids in the primary set and
which is exactly the defect that inflated every v0 number for four months. The
two requests differ only in a date range. Nothing in the response says which
one you made.

So the separation is made structural rather than remembered:

* `climatological_prior` filters to seasons **strictly before** the race's own
  season before computing anything. A prior for the 2015 Bahrain GP is built
  from Bahrain mid-Marches in 2000-2014. It cannot see 2015 because the years
  it may read are chosen by a comparison the caller cannot pass through.
* `race_observation` returns the race day itself and its columns are registered
  `POST_RACE`, so the static leakage check keeps them out of the primary matrix
  and the behavioural check catches any pre-race feature that reads them.

Both come from one cached download per circuit, so the corpus costs 35 requests
rather than one per race.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from f1pred.config import RAW_DIR
from f1pred.reference import circuits

logger = logging.getLogger(__name__)

ARCHIVE_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"

#: ERA5 reanalysis, served by Open-Meteo. Free for non-commercial use (FR-904),
#: attribution recorded in the README under FR-905.
SOURCE_NAME: Final[str] = "Open-Meteo ERA5 archive"

#: Daily variables pulled. Deliberately few: the prior only has to say how
#: likely rain is and roughly how hot and windy it gets, and every extra
#: variable is another column to keep on the right side of the leakage line.
DAILY_VARIABLES: Final[tuple[str, ...]] = (
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_mean",
    "wind_speed_10m_max",
)

#: First year requested per circuit. ERA5 reaches further back, but a prior
#: built from the 1980s describes a different climate than the race is run in.
ARCHIVE_START_YEAR: Final[int] = 1995

#: Half-width, in days, of the calendar window a prior is built from. A race
#: moves around the calendar between seasons, so asking for the exact date
#: across years would compare mid-March with late-March. Fifteen days either
#: side keeps the season while absorbing the drift.
PRIOR_WINDOW_DAYS: Final[int] = 15

#: A prior needs this many prior-year observations to be reported at all.
#: Below it the value is null — which for the corpus's first seasons is the
#: honest answer, not a number to be manufactured.
MIN_PRIOR_YEARS: Final[int] = 5

#: Daily total in millimetres above which a day counts as wet. A racing circuit
#: dries quickly; a trace of drizzle is not a wet race.
WET_DAY_MM: Final[float] = 1.0


class WeatherError(RuntimeError):
    """Raised when weather data cannot be obtained or is internally inconsistent."""


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class ArchiveClient:
    """Reads the ERA5 daily archive. The project's third and last networked object."""

    def __init__(
        self,
        *,
        pause: float = 2.0,
        timeout: float = 60.0,
        rate_limit_sleep: float = 65.0,
        max_retries: int = 8,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.pause = pause
        self.timeout = timeout
        #: The archive limits by the minute, not the hour, and counts a
        #: thirty-year daily series as many calls. Measured 2026-08-31: five
        #: circuits through, then 429 on every one after. Waiting just over a
        #: minute clears it, so a refusal is a pause rather than a failure —
        #: recording it as missing data would put permanent holes in the prior
        #: for whichever circuits happened to come late in the alphabet.
        self.rate_limit_sleep = rate_limit_sleep
        self.max_retries = max_retries
        self._sleep = sleep

    def circuit_daily(
        self, circuit_id: str, *, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Every day in the range at one circuit, one request.

        Whole years are fetched rather than per-race windows because the archive
        charges the same for a decade as for a week, and a single cached frame
        per circuit removes any chance of two races disagreeing about the same
        day's weather.
        """
        import requests

        circuit = circuits.get(circuit_id)
        params = {
            "latitude": circuit.latitude,
            "longitude": circuit.longitude,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "UTC",
        }
        for attempt in range(self.max_retries + 1):
            self._sleep(self.pause)
            response = requests.get(ARCHIVE_URL, params=params, timeout=self.timeout)
            if response.status_code == 200:
                break
            if response.status_code == 429 and attempt < self.max_retries:
                logger.warning(
                    "%s: archive rate limited; waiting %.0fs",
                    circuit_id, self.rate_limit_sleep,
                )
                self._sleep(self.rate_limit_sleep)
                continue
            raise WeatherError(
                f"{circuit_id}: archive returned {response.status_code}: "
                f"{response.text[:200]}"
            )
        daily = response.json().get("daily")
        if not daily or not daily.get("time"):
            raise WeatherError(f"{circuit_id}: archive returned no daily series")

        frame = pd.DataFrame(daily)
        frame["date"] = pd.to_datetime(frame.pop("time"))
        frame.insert(0, "circuit_id", circuit_id)
        return frame


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def archive_path(root: Path = RAW_DIR) -> Path:
    return Path(root) / "weather_archive.parquet"


def write_archive(df: pd.DataFrame, root: Path = RAW_DIR) -> Path:
    path = archive_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    df.to_parquet(temp, index=False)
    temp.replace(path)
    return path


def read_archive(root: Path = RAW_DIR) -> pd.DataFrame | None:
    path = archive_path(root)
    return pd.read_parquet(path) if path.exists() else None


def collect_archive(
    circuit_ids: Sequence[str],
    *,
    client: Any,
    start_year: int = ARCHIVE_START_YEAR,
    end_year: int | None = None,
    log: Callable[[str], None] = logger.info,
) -> tuple[pd.DataFrame, list[str]]:
    """One request per circuit. Returns the archive and any circuits that failed."""
    end_year = end_year or pd.Timestamp.today().year
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for circuit_id in circuit_ids:
        try:
            frames.append(
                client.circuit_daily(
                    circuit_id, start_year=start_year, end_year=end_year
                )
            )
            log(f"weather {circuit_id}: ok")
        except Exception as exc:  # noqa: BLE001
            failed.append(circuit_id)
            log(f"weather {circuit_id}: FAILED ({exc})")
    combined = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    return combined, failed


# --------------------------------------------------------------------------
# The prior — admissible pre-race (FR-206)
# --------------------------------------------------------------------------


def _window_mask(dates: pd.Series, target: pd.Timestamp, days: int) -> pd.Series:
    """Days within `days` of the target's day-of-year, wrapping at New Year."""
    delta = (dates.dt.dayofyear - target.dayofyear).abs()
    return np.minimum(delta, 365 - delta) <= days


def climatological_prior(
    archive: pd.DataFrame,
    circuit_id: str,
    race_date: pd.Timestamp,
    *,
    window_days: int = PRIOR_WINDOW_DAYS,
    min_years: int = MIN_PRIOR_YEARS,
) -> dict[str, float]:
    """What this circuit's weather is usually like at this time of year.

    Built **only** from calendar years strictly before the race's own. That
    filter is the whole compliance argument for FR-206: no caller can widen it,
    because the cut-off is derived from `race_date` itself rather than passed
    in.

    Returns nulls, not defaults, when fewer than `min_years` years are
    available. The corpus's earliest seasons genuinely have a thin archive
    behind them and FR-214's principle applies here too — an absent measurement
    is not a zero.
    """
    null = {
        "weather_prior_wet_probability": float("nan"),
        "weather_prior_precip_mm": float("nan"),
        "weather_prior_temp_max_c": float("nan"),
        "weather_prior_wind_kmh": float("nan"),
        "weather_prior_years": 0.0,
    }
    at_circuit = archive[archive["circuit_id"] == circuit_id]
    if at_circuit.empty:
        return null

    # The cut-off. Strictly earlier years only.
    earlier = at_circuit[at_circuit["date"].dt.year < race_date.year]
    if earlier.empty:
        return null

    window = earlier[_window_mask(earlier["date"], race_date, window_days)]
    years = int(window["date"].dt.year.nunique())
    if years < min_years:
        return null | {"weather_prior_years": float(years)}

    precip = pd.to_numeric(window["precipitation_sum"], errors="coerce")
    return {
        "weather_prior_wet_probability": float((precip >= WET_DAY_MM).mean()),
        "weather_prior_precip_mm": float(precip.mean()),
        "weather_prior_temp_max_c": float(
            pd.to_numeric(window["temperature_2m_max"], errors="coerce").mean()
        ),
        "weather_prior_wind_kmh": float(
            pd.to_numeric(window["wind_speed_10m_max"], errors="coerce").mean()
        ),
        "weather_prior_years": float(years),
    }


def race_observation(
    archive: pd.DataFrame, circuit_id: str, race_date: pd.Timestamp
) -> dict[str, float]:
    """What actually happened on race day. **Post-race. Oracle set only.**

    Registered `POST_RACE` so the static check keeps it out of the primary
    matrix. It exists to quantify how much a weather leak is worth — which is
    the question v0 answered by accident.
    """
    day = archive[
        (archive["circuit_id"] == circuit_id)
        & (archive["date"] == pd.Timestamp(race_date).normalize())
    ]
    if day.empty:
        return {
            "weather_actual_precip_mm": float("nan"),
            "weather_actual_temp_max_c": float("nan"),
            "weather_actual_wind_kmh": float("nan"),
        }
    row = day.iloc[0]
    return {
        "weather_actual_precip_mm": float(row["precipitation_sum"]),
        "weather_actual_temp_max_c": float(row["temperature_2m_max"]),
        "weather_actual_wind_kmh": float(row["wind_speed_10m_max"]),
    }


PRIOR_COLUMNS: Final[tuple[str, ...]] = (
    "weather_prior_wet_probability",
    "weather_prior_precip_mm",
    "weather_prior_temp_max_c",
    "weather_prior_wind_kmh",
    "weather_prior_years",
)

ACTUAL_COLUMNS: Final[tuple[str, ...]] = (
    "weather_actual_precip_mm",
    "weather_actual_temp_max_c",
    "weather_actual_wind_kmh",
)


def race_weather_table(archive: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per race carrying both the prior and the observation."""
    races = (
        corpus[["season", "round", "race_date", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["season", "round"], kind="stable")
        .reset_index(drop=True)
    )
    rows = []
    for r in races.itertuples():
        date = pd.Timestamp(r.race_date)
        rows.append(
            {
                "season": int(r.season),
                "round": int(r.round),
                "circuit_id": str(r.circuit_id),
                **climatological_prior(archive, str(r.circuit_id), date),
                **race_observation(archive, str(r.circuit_id), date),
            }
        )
    return pd.DataFrame(rows)


def attach(corpus: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join the per-race weather onto the corpus, filling nothing."""
    keys = ["season", "round"]
    columns = [c for c in weather.columns if c not in (*keys, "circuit_id")]
    merged = corpus.merge(weather[keys + columns], on=keys, how="left")
    if len(merged) != len(corpus):
        raise WeatherError(
            f"attaching weather changed the row count ({len(corpus)} -> "
            f"{len(merged)}); the weather table has duplicate {keys}"
        )
    return merged


def provenance(retrieved: str) -> dict[str, str]:
    """FR-905's record for the README. Written by the collector, not by hand."""
    return {
        "source": SOURCE_NAME,
        "url": ARCHIVE_URL,
        "dataset": "ERA5 reanalysis",
        "retrieved": retrieved,
        "licence": "Open-Meteo: free for non-commercial use; ERA5 via Copernicus (CC-BY-4.0)",
        "role_split": (
            "Days before the race year form the FR-206 climatological prior "
            "(PRE_RACE). The race day itself is POST_RACE and reachable only "
            "from the oracle set."
        ),
    }


__all__ = [
    "ACTUAL_COLUMNS",
    "ARCHIVE_START_YEAR",
    "ARCHIVE_URL",
    "ArchiveClient",
    "DAILY_VARIABLES",
    "MIN_PRIOR_YEARS",
    "PRIOR_COLUMNS",
    "PRIOR_WINDOW_DAYS",
    "SOURCE_NAME",
    "WET_DAY_MM",
    "WeatherError",
    "archive_path",
    "attach",
    "climatological_prior",
    "collect_archive",
    "provenance",
    "race_observation",
    "race_weather_table",
    "read_archive",
    "write_archive",
]
