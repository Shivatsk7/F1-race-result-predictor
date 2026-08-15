"""Data collector for historical F1 data using FastF1."""

import logging
from pathlib import Path

import pandas as pd
import fastf1
from fastf1 import Cache

from src.config import (
    HISTORICAL_YEARS_START,
    HISTORICAL_YEARS_END,
    RAW_DIR,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)

# Enable FastF1 caching
CACHE_DIR.mkdir(parents=True, exist_ok=True)
Cache.enable_cache(str(CACHE_DIR))


def collect_season_data(year: int) -> pd.DataFrame:
    """Collect all race data for a given season year."""
    logger.info(f"Collecting data for {year} season...")
    schedule = fastf1.get_event_schedule(year)
    race_records = []

    for _, event in schedule.iterrows():
        try:
            session = fastf1.get_session(year, event["RoundNumber"], "R")
            session.load(telemetry=False, weather=True)

            results = session.results
            weather_data = session.weather_data

            # Get qualifying data
            try:
                quali = fastf1.get_session(year, event["RoundNumber"], "Q")
                quali.load(telemetry=False, laps=False, weather=False)
                quali_results = quali.results
            except Exception:
                quali_results = None

            for _, driver in results.iterrows():
                record = _build_driver_record(
                    driver, session, weather_data, quali_results, year, event
                )
                race_records.append(record)

        except Exception as e:
            logger.warning(f"Skipping round {event['RoundNumber']} of {year}: {e}")
            continue

    df = pd.DataFrame(race_records)
    logger.info(f"Collected {len(df)} records for {year}")
    return df


def _build_driver_record(driver, session, weather_data, quali_results, year, event):
    """Build a single driver record from race data."""
    driver_number = driver["DriverNumber"]
    driver_name = driver["Abbreviation"]

    # Determine weather conditions
    weather_summary = _summarize_weather(weather_data)

    # Qualifying info
    grid_pos = driver.get("GridPosition", None)
    quali_gap = None
    quali_sector_ranks = {"sector1": None, "sector2": None, "sector3": None}

    if quali_results is not None:
        quali_row = quali_results[quali_results["DriverNumber"] == driver_number]
        if not quali_row.empty:
            quali_row = quali_row.iloc[0]
            pole_time = quali_results["Q3"].min() if quali_results["Q3"].notna().any() else None
            driver_quali_time = quali_row.get("Q3", None) or quali_row.get("Q2", None) or quali_row.get("Q1", None)
            if pole_time and driver_quali_time:
                quali_gap = (driver_quali_time - pole_time).total_seconds()

    # Race result
    race_position = driver.get("Position", None)
    status = driver.get("Status", None)
    points = driver.get("Points", None)

    # Team info
    team = driver.get("TeamName", None)

    record = {
        "year": year,
        "round": event["RoundNumber"],
        "race_name": event.get("EventName", ""),
        "circuit": event.get("Location", ""),
        "driver": driver_name,
        "driver_number": driver_number,
        "team": team,
        "grid_position": grid_pos,
        "qualifying_gap_to_pole": quali_gap,
        "race_position": race_position,
        "status": status,
        "points": points,
        "weather_dry": weather_summary.get("dry", 1),
        "weather_wet": weather_summary.get("wet", 0),
        "weather_mixed": weather_summary.get("mixed", 0),
    }
    return record


def _summarize_weather(weather_data) -> dict:
    """Summarize weather data into dry/wet/mixed categories."""
    if weather_data is None or weather_data.empty:
        return {"dry": 1, "wet": 0, "mixed": 0}

    rain_vals = weather_data["Rainfall"].dropna()
    if rain_vals.empty:
        return {"dry": 1, "wet": 0, "mixed": 0}

    rain_pct = rain_vals.mean()
    if rain_pct == 0:
        return {"dry": 1, "wet": 0, "mixed": 0}
    elif rain_pct >= 0.7:
        return {"dry": 0, "wet": 1, "mixed": 0}
    else:
        return {"dry": 0, "wet": 0, "mixed": 1}


def collect_all_seasons() -> pd.DataFrame:
    """Collect data for all configured seasons."""
    all_data = []
    for year in range(HISTORICAL_YEARS_START, HISTORICAL_YEARS_END + 1):
        filepath = RAW_DIR / f"season_{year}.parquet"
        if filepath.exists():
            logger.info(f"Loading cached data for {year}")
            df = pd.read_parquet(filepath)
        else:
            df = collect_season_data(year)
            df.to_parquet(filepath, index=False)

        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_parquet(RAW_DIR / "all_seasons.parquet", index=False)
    logger.info(f"Total records collected: {len(combined)}")
    return combined


def load_cached_data() -> pd.DataFrame:
    """Load the best available local raw dataset without fetching new data."""
    combined_path = RAW_DIR / "all_seasons.parquet"
    if combined_path.exists():
        logger.info("Loading cached combined dataset from %s", combined_path)
        return pd.read_parquet(combined_path)

    season_frames = []
    for file_path in sorted(RAW_DIR.glob("season_*.parquet")):
        season_frames.append(pd.read_parquet(file_path))

    if season_frames:
        logger.info("Loading cached season files from %s", RAW_DIR)
        return pd.concat(season_frames, ignore_index=True)

    raise FileNotFoundError(
        "No cached raw dataset found. Run `python -m src.cli collect` first."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all_seasons()
