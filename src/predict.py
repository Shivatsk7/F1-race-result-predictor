"""Prediction module for upcoming F1 races."""

import logging
from datetime import datetime

import fastf1
import pandas as pd

from src.config import CACHE_DIR
from src.feature_engineer import engineer_features
from src.model import RacePredictor

logger = logging.getLogger(__name__)


def predict_race(year: int, round_number: int) -> pd.DataFrame:
    """Predict the outcome of a specific race."""
    predictor = RacePredictor().load()
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    event = _load_event(year, round_number)
    quali_results = _load_qualifying_results(year, round_number)
    race_session = _load_race_session(year, round_number)
    race_results = None if race_session is None else race_session.results

    if quali_results is not None and not quali_results.empty:
        driver_entries = quali_results
    elif race_results is not None and not race_results.empty:
        driver_entries = race_results
    else:
        logger.error("No qualifying or race entry list available for %s round %s", year, round_number)
        return pd.DataFrame()

    weather_summary = {"dry": 1, "wet": 0, "mixed": 0}
    if race_session is not None and getattr(race_session, "weather_data", None) is not None:
        from src.data_collector import _summarize_weather

        weather_summary = _summarize_weather(race_session.weather_data)

    pred_rows = [
        _build_prediction_features(
            driver=row,
            event=event,
            weather_summary=weather_summary,
            quali_results=quali_results,
            year=year,
            round_number=round_number,
        )
        for _, row in driver_entries.iterrows()
    ]

    pred_df = pd.DataFrame(pred_rows)
    pred_df["_is_prediction_row"] = True

    historical_df = _load_historical_data(year, round_number)
    if not historical_df.empty:
        historical_df = historical_df.copy()
        historical_df["_is_prediction_row"] = False

    combined = pd.concat([historical_df, pred_df], ignore_index=True, sort=False)
    combined = engineer_features(combined, keep_unlabeled=True, save_processed=False)

    pred_features = combined[combined["_is_prediction_row"]].copy()
    if pred_features.empty:
        logger.error("No prediction features generated")
        return pd.DataFrame()

    pred_features = pred_features.sort_values(["grid_position", "driver"], na_position="last").reset_index(drop=True)

    prediction_frame = predictor.predict_with_confidence(
        pred_features,
        race_keys=pred_features[["year", "round", "benchmark_group", "benchmark_group_code"]],
    )
    results = pd.concat(
        [
            pred_features[["driver", "team", "grid_position"]].reset_index(drop=True),
            prediction_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    results = results.sort_values(
        ["predicted_position", "predicted_position_raw", "grid_position"],
        na_position="last",
    ).reset_index(drop=True)
    results.index = results.index + 1
    results.index.name = "predicted_rank"
    return results


def predict_next_race() -> pd.DataFrame:
    """Predict the outcome of the next upcoming race."""
    current_year = datetime.now().year
    schedule = fastf1.get_event_schedule(current_year)

    today = pd.Timestamp.now()
    future_races = schedule[schedule["EventDate"] > today]

    if future_races.empty:
        logger.info("No upcoming races found")
        return pd.DataFrame()

    next_race = future_races.iloc[0]
    logger.info("Next race: %s (Round %s)", next_race["EventName"], next_race["RoundNumber"])
    return predict_race(current_year, int(next_race["RoundNumber"]))


def _build_prediction_features(
    driver: pd.Series,
    event: pd.Series,
    weather_summary: dict,
    quali_results: pd.DataFrame | None,
    year: int,
    round_number: int,
) -> dict:
    """Build feature dict for a single driver prediction."""
    driver_number = driver.get("DriverNumber")
    driver_name = driver.get("Abbreviation") or driver.get("BroadcastName") or driver.get("FullName") or str(driver_number)
    team = driver.get("TeamName", "")
    grid_position = driver.get("GridPosition")
    if pd.isna(grid_position):
        grid_position = driver.get("Position")

    qualifying_gap = None
    if quali_results is not None and not quali_results.empty and driver_number is not None:
        quali_row = quali_results[quali_results["DriverNumber"] == driver_number]
        if not quali_row.empty:
            driver_quali_time = _best_quali_time(quali_row.iloc[0])
            pole_time = _pole_quali_time(quali_results)
            if driver_quali_time is not None and pole_time is not None:
                qualifying_gap = (driver_quali_time - pole_time).total_seconds()

    return {
        "year": year,
        "round": round_number,
        "race_name": event.get("EventName", ""),
        "circuit": event.get("Location", ""),
        "driver": driver_name,
        "driver_number": driver_number,
        "team": team,
        "grid_position": grid_position,
        "qualifying_gap_to_pole": qualifying_gap,
        "race_position": None,
        "status": None,
        "points": None,
        "weather_dry": weather_summary.get("dry", 1),
        "weather_wet": weather_summary.get("wet", 0),
        "weather_mixed": weather_summary.get("mixed", 0),
    }


def _load_event(year: int, round_number: int) -> pd.Series:
    """Load a schedule row for the requested event."""
    schedule = fastf1.get_event_schedule(year)
    event = schedule[schedule["RoundNumber"] == round_number]
    if event.empty:
        raise ValueError(f"Round {round_number} not found in {year} schedule.")
    return event.iloc[0]


def _load_qualifying_results(year: int, round_number: int) -> pd.DataFrame | None:
    """Load qualifying results when they are available."""
    try:
        quali_session = fastf1.get_session(year, round_number, "Q")
        quali_session.load(telemetry=False, laps=False, weather=False)
        return quali_session.results
    except Exception as exc:
        logger.warning("Qualifying data unavailable for %s round %s: %s", year, round_number, exc)
        return None


def _load_race_session(year: int, round_number: int):
    """Load race session context when it exists."""
    try:
        race_session = fastf1.get_session(year, round_number, "R")
        race_session.load(telemetry=False, weather=True)
        return race_session
    except Exception as exc:
        logger.warning("Race session context unavailable for %s round %s: %s", year, round_number, exc)
        return None


def _pole_quali_time(quali_results: pd.DataFrame):
    """Return the fastest available qualifying time."""
    candidate_times = [time for time in (_best_quali_time(row) for _, row in quali_results.iterrows()) if time is not None]
    return min(candidate_times) if candidate_times else None


def _best_quali_time(row: pd.Series):
    """Return a driver's best available qualifying session time."""
    for column in ("Q3", "Q2", "Q1"):
        value = row.get(column)
        if pd.notna(value):
            return value
    return None


def _load_historical_data(year: int, round_number: int) -> pd.DataFrame:
    """Load historical data strictly before the requested race."""
    from src.config import RAW_DIR

    filepath = RAW_DIR / "all_seasons.parquet"
    all_data = []

    if filepath.exists():
        all_data.append(pd.read_parquet(filepath))
    else:
        for file_path in RAW_DIR.glob("season_*.parquet"):
            all_data.append(pd.read_parquet(file_path))

    if not all_data:
        return pd.DataFrame()

    historical = pd.concat(all_data, ignore_index=True)
    return historical[
        (historical["year"] < year)
        | ((historical["year"] == year) & (historical["round"] < round_number))
    ].copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = predict_next_race()
    if not results.empty:
        print("\n" + "=" * 60)
        print("RACE PREDICTION")
        print("=" * 60)
        print(results.to_string())
