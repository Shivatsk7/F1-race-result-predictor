"""Predicting one unraced race from a starting grid (FR-801, US-4).

The awkward part of predicting a race that has not happened is that almost
every feature this project has is a *history* feature — a driver's recent form,
a constructor's rolling pace, seasons since the regulation reset. None of them
can be computed from a grid sheet alone. They are computed from the corpus,
with the new race appended to the end of it.

Appending is the whole trick, and it is safe for one specific reason: the new
race is chronologically last, and every primary feature is shifted so it reads
only earlier races. A fake outcome on the final race can therefore reach no
feature of any row, because there is no row after it. That is asserted here
(`_assert_is_last`) rather than assumed, because it stops being true the moment
somebody predicts a race that sits in the middle of the corpus — and the
failure would be silent, which is how the last leak survived.

The placeholder outcome exists only so `build_target` can run. It is discarded
before anything is returned, and the primary feature set cannot read it in any
case: post-race columns live in the oracle set (FR-207).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from f1pred.config import RAW_DIR, SEED
from f1pred.data import weather
from f1pred.data.schema import (
    RACE_ROW_SCHEMA,
    ClassificationOutcome,
    WeekendFormat,
)
from f1pred.dataset import enrich, load_corpus
from f1pred.features.build import build_primary
from f1pred.models.ranker import TrainedRanker
from f1pred.simulate.montecarlo import (
    fit_simulation_parameters,
    pace_noise_std_from_interval,
    simulate_race,
)

logger = logging.getLogger(__name__)

__all__ = ["PredictionError", "append_unraced", "predict_race"]

#: Sentinel outcome for the appended race. Every value here is discarded; the
#: columns exist only because `build_target` computes an order per race and
#: cannot run against a race with no classification at all.
_PLACEHOLDER_STATUS = "Finished"
_PLACEHOLDER_OUTCOME = ClassificationOutcome.CLASSIFIED.value


class PredictionError(ValueError):
    """Raised when a grid cannot be predicted from the corpus on disk."""


def _assert_is_last(corpus: pd.DataFrame, grid: pd.DataFrame) -> None:
    """The appended race must be after every race in the corpus.

    Not a formality. Features are shifted within driver and constructor, so a
    race inserted mid-corpus would have its placeholder outcome read by every
    later race's form features — a leak with the same shape as v0's, arriving
    through the prediction path rather than the training one.
    """
    season = int(grid["season"].iloc[0])
    round_number = int(grid["round"].iloc[0])
    race_date = pd.Timestamp(grid["race_date"].iloc[0])

    already = corpus[
        (corpus["season"] == season) & (corpus["round"] == round_number)
    ]
    if not already.empty:
        raise PredictionError(
            f"season {season} round {round_number} is already in the corpus "
            f"({len(already)} rows). This command predicts a race that has not "
            f"run; to score a race that has, use `f1pred evaluate`, which holds "
            f"it out properly instead of predicting a race it was trained on."
        )

    latest = pd.Timestamp(corpus["race_date"].max())
    if race_date <= latest:
        raise PredictionError(
            f"the grid is dated {race_date.date()}, which is not after the "
            f"corpus's last race ({latest.date()}). Every feature this model "
            f"uses is shifted to read only earlier races, so a race inserted "
            f"before the end of the corpus would leak its placeholder outcome "
            f"into the form features of every race after it."
        )


def append_unraced(corpus: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Return the corpus with the grid appended as one more race.

    The appended rows carry placeholder outcome columns so the target builder
    can run over them. They are meaningless and never read: the caller drops
    the appended race's target, and the primary feature set cannot see a
    race's own outcome by construction.
    """
    _assert_is_last(corpus, grid)

    n = len(grid)
    rows = pd.DataFrame(
        {
            "season": grid["season"].to_numpy(),
            "round": grid["round"].to_numpy(),
            "race_date": grid["race_date"].to_numpy(),
            "circuit_id": grid["circuit_id"].astype("string").to_numpy(),
            "weekend_format": pd.array([WeekendFormat.CONVENTIONAL.value] * n, dtype="string"),
            "weekend_format_raw": pd.array(["unraced"] * n, dtype="string"),
            "driver_id": grid["driver_id"].astype("string").to_numpy(),
            "driver_code": pd.array([pd.NA] * n, dtype="string"),
            "constructor_id": grid["constructor_id"].astype("string").to_numpy(),
            "grid_position": grid["grid_position"].to_numpy(),
            # Placeholders, discarded. Set from the grid so the synthetic order
            # is at least internally consistent; nothing downstream reads it.
            "classified_position": grid["grid_position"].to_numpy(),
            "position_text": grid["grid_position"].astype("string").to_numpy(),
            "classification_outcome": pd.array([_PLACEHOLDER_OUTCOME] * n, dtype="string"),
            "status_raw": pd.array([_PLACEHOLDER_STATUS] * n, dtype="string"),
            "laps_completed": np.zeros(n, dtype="int64"),
            "points": np.zeros(n, dtype="float64"),
            "final_lap_position": pd.array([pd.NA] * n, dtype="Int64"),
            "sprint_position": pd.array([pd.NA] * n, dtype="Int64"),
            "sprint_grid_position": pd.array([pd.NA] * n, dtype="Int64"),
        }
    )
    for column, dtype in RACE_ROW_SCHEMA.items():
        if column in rows.columns:
            rows[column] = rows[column].astype(dtype)

    # Enrichment columns (weather, telemetry) that the corpus carries and a
    # grid sheet cannot. Created with the corpus's own dtype rather than as
    # bare `NA`: an all-object null column changes the dtype of the column it
    # is concatenated onto, and a float feature silently becoming object is
    # the kind of thing that surfaces as a model error three calls later.
    for column in corpus.columns:
        if column not in rows.columns:
            # An empty series of the target dtype fills with that dtype's own
            # null — NaN for float64, pd.NA for the extension types — which is
            # what keeps the concatenated column's dtype unchanged.
            rows[column] = pd.Series(index=range(n), dtype=corpus[column].dtype)

    return pd.concat([corpus, rows[list(corpus.columns)]], ignore_index=True)


def _attach_grid_weather_prior(frame: pd.DataFrame, grid: pd.DataFrame, root) -> pd.DataFrame:
    """Give the appended race the circuit's climatological prior for its date.

    This is the admissible half of FR-751a: what this track's weather is
    usually like in this part of the year, computed from earlier years and
    knowable months in advance. The observed weather of the race being
    predicted is the inadmissible half, and there is deliberately no path by
    which it can arrive — `grid_csv` refuses the column outright.
    """
    archive = weather.read_archive(root)
    if archive is None or archive.empty:
        logger.info("no weather archive; the new race's weather features stay null")
        return frame

    circuit_id = str(grid["circuit_id"].iloc[0])
    race_date = pd.Timestamp(grid["race_date"].iloc[0])
    prior = weather.climatological_prior(archive, circuit_id, race_date)

    mask = (frame["season"] == int(grid["season"].iloc[0])) & (
        frame["round"] == int(grid["round"].iloc[0])
    )
    for column, value in prior.items():
        if column not in frame.columns:
            frame[column] = pd.NA
        frame.loc[mask, column] = value
    return frame


def _flag_text(row: pd.Series) -> str:
    flags = []
    if bool(row.get("low_information", False)):
        flags.append("LOW_INFORMATION")
    if bool(row.get("post_transfer", False)):
        flags.append("POST_TRANSFER")
    return ",".join(flags)


def predict_race(
    grid: pd.DataFrame,
    model: TrainedRanker,
    *,
    draws: int = 1000,
    seed: int = SEED,
    root=RAW_DIR,
) -> tuple[pd.DataFrame, list[str]]:
    """Rank one unraced grid. Returns the table and any notes for the reader.

    The table carries a point prediction, an interval, and the simulated win,
    podium and points probabilities. It never carries a bare point (FR-752).
    """
    corpus = enrich(load_corpus(root=root), root=root)
    combined = append_unraced(corpus, grid)
    combined = _attach_grid_weather_prior(combined, grid, root)

    matrix = build_primary(combined)
    keys = matrix.keys
    mask = (keys["season"] == int(grid["season"].iloc[0])) & (
        keys["round"] == int(grid["round"].iloc[0])
    )
    if not mask.any():
        raise PredictionError(
            "the appended race did not survive feature building; the most "
            "likely cause is that every entry was dropped as a non-starter"
        )

    race = matrix.X.loc[mask].copy()
    race_keys = keys.loc[mask]
    flags = matrix.flags.loc[mask]

    predicted = model.predict_with_interval(race)

    frame = pd.DataFrame(
        {
            "driver_id": race_keys["driver_id"].to_numpy(),
            "constructor_id": race_keys["constructor_id"].to_numpy(),
            "circuit_id": race_keys["circuit_id"].to_numpy(),
        },
        index=race.index,
    )
    frame["grid_position"] = (
        grid.set_index("driver_id")["grid_position"]
        .reindex(frame["driver_id"])
        .to_numpy()
    )
    frame["predicted_position"] = predicted["predicted_position"]
    # The simulator reads the *feature*, not the raw column: FR-751a admits the
    # climatological prior into the simulation and nothing else, and the
    # feature is the only form of it that has been through the firewall.
    if "weather_prior_wet" in race.columns:
        frame["weather_prior_wet"] = race["weather_prior_wet"]

    training = corpus  # the whole corpus precedes this race, so all of it is training
    pace_std = pace_noise_std_from_interval(
        half_width=(model.interval.upper_offset - model.interval.lower_offset) / 2,
        coverage=model.interval.coverage,
    )
    params = fit_simulation_parameters(training, pace_noise_std=pace_std)
    simulated = simulate_race(frame, params, n_simulations=draws, seed=seed)

    merged = simulated.merge(
        pd.DataFrame(
            {
                "driver_id": frame["driver_id"].to_numpy(),
                "predicted_position": predicted["predicted_position"].to_numpy(),
                "lower": predicted["predicted_position_lower"].to_numpy(),
                "upper": predicted["predicted_position_upper"].to_numpy(),
                "interval_coverage": predicted["interval_coverage"].to_numpy(),
                "low_information": flags["low_information"].to_numpy(),
                "post_transfer": flags["post_transfer"].to_numpy(),
            }
        ),
        on="driver_id",
        how="left",
        validate="one_to_one",
    )

    merged = merged.sort_values("predicted_position", kind="stable").reset_index(drop=True)
    merged["predicted_rank"] = range(1, len(merged) + 1)
    merged["interval"] = [
        f"P{low:.0f}-P{high:.0f}"
        for low, high in zip(merged["lower"], merged["upper"])
    ]
    merged["flags"] = merged.apply(_flag_text, axis=1)

    notes: list[str] = []
    flagged = merged.loc[merged["low_information"], "driver_id"].tolist()
    if flagged:
        notes.append(
            f"\n{len(flagged)} entr{'y' if len(flagged) == 1 else 'ies'} flagged "
            f"LOW_INFORMATION — {', '.join(flagged)} — served a declared "
            f"cold-start fallback rather than their own history (FR-212). Their "
            f"placement rests on an assumption, not evidence."
        )
    transferred = merged.loc[merged["post_transfer"], "driver_id"].tolist()
    if transferred:
        notes.append(
            f"{len(transferred)} entr{'y' if len(transferred) == 1 else 'ies'} "
            f"changed constructor since last season — {', '.join(transferred)} — "
            f"so their constructor-form history was reset rather than carried "
            f"across the move (FR-216)."
        )
    if "weather_prior_wet" not in frame.columns or frame["weather_prior_wet"].isna().all():
        notes.append(
            "This race was predicted with its weather features null — either no "
            "weather archive is on disk, or the archive holds too few earlier "
            "years at this circuit to form a prior. Run `f1pred collect "
            "--weather` if the archive is what is missing. A null here is "
            "carried through rather than filled (FR-109, FR-212)."
        )
    return merged, notes
