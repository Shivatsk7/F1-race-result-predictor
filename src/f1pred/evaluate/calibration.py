"""Calibration of simulated probabilities against held-out outcomes.

Implements FR-701..706: T-505 (reliability curves), T-506 (Brier scores
against two baselines), T-507 (interval coverage), T-508 (the OQ-3 threshold),
and T-509 (the `UNCALIBRATED` label that no call site can omit).

**Two baselines, both fitted on training seasons only.** FR-703 asks for a
climatological baseline (the base rate) and the grid-order baseline. Neither
is invented: the climatological rate and the grid-conditional rate are both
read off `ChronologicalSplit`'s training rows, the same discipline every other
prior-only quantity in this project follows (`features.priors`, T-510's
simulation parameters). A grid slot absent from training falls back to the
climatological rate rather than an unseen key error — a very large field is
rare but not impossible.

**Race-level bootstrap, again.** T-307 established that a metric here is
resampled by race, not by driver-row, because twenty cars in one race are not
twenty independent draws. The Brier scores and the coverage rate get the same
treatment; `_bootstrap_race_mean` is the same percentile bootstrap as
`evaluate.bootstrap`, kept local because it resamples a plain per-race series
rather than the ranking-`Metric` battery that module is built around.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from f1pred.config import RESULTS_DIR, SEED
from f1pred.dataset import Experiment, enrich, load_corpus, prepare
from f1pred.evaluate.bootstrap import DEFAULT_CONFIDENCE, DEFAULT_REPLICATES, Interval
from f1pred.evaluate.runner import train_model
from f1pred.evaluate.split import ChronologicalSplit, default_split
from f1pred.models.ranker import TrainedRanker
from f1pred.simulate.montecarlo import (
    POINTS_POSITIONS,
    REQUIRED_RACE_COLUMNS,
    SimulationParameters,
    fit_simulation_parameters,
    pace_noise_std_from_interval,
    simulate_race,
)

logger = logging.getLogger(__name__)

#: The three event types FR-702 names.
EVENT_TYPES: Final[tuple[str, ...]] = ("win", "podium", "points")

#: OQ-3's answer (T-508). A reliability-curve bin, or an interval's observed
#: coverage, may differ from what it claims by at most this much before the
#: probability is rendered `UNCALIBRATED` (FR-705). Ten percentage points is a
#: round, conservative bound chosen to be fixed *before* looking at whether
#: this run passes it — not tuned to make it pass — because OQ-3 asks for a
#: threshold set once real reliability curves exist, and a threshold picked
#: after seeing the number it is about to gate is not a threshold.
CALIBRATION_ERROR_THRESHOLD: Final[float] = 0.10

#: A reliability-curve bin below this many observations is excluded from
#: `max_reliability_gap` rather than trusted — see that function's docstring.
MIN_RELIABILITY_BIN_COUNT: Final[int] = 10


class CalibrationError(ValueError):
    """Raised when a calibration quantity cannot be computed as specified."""


# --------------------------------------------------------------------------
# Baselines, fitted on training rows only (FR-703)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventBaselines:
    """Climatological and grid-conditional base rates for one event type."""

    climatological_rate: float
    grid_rate: dict[int, float]

    def grid_probability(self, grid_position: int) -> float:
        return self.grid_rate.get(int(grid_position), self.climatological_rate)


def fit_event_baselines(
    training_frame: pd.DataFrame, *, event: str
) -> EventBaselines:
    """FR-703's two baselines for one event, read off training rows only.

    `training_frame` is `split.split(built.frame)[0]` — the same frame the
    model itself trained on, carrying `target_position` and `grid_position`.
    """
    if event not in EVENT_TYPES:
        raise CalibrationError(f"unknown event type {event!r}; must be one of {EVENT_TYPES}")
    if training_frame.empty:
        raise CalibrationError("cannot fit baselines from an empty training frame")
    outcome = _event_outcome(training_frame["target_position"], event)
    climatological_rate = float(outcome.mean())
    grid_rate = (
        pd.Series(outcome.to_numpy(), index=training_frame["grid_position"])
        .groupby(level=0)
        .mean()
        .to_dict()
    )
    return EventBaselines(climatological_rate=climatological_rate, grid_rate=grid_rate)


def _event_outcome(target_position: pd.Series, event: str) -> pd.Series:
    positions = pd.to_numeric(target_position, errors="coerce")
    if event == "win":
        return (positions == 1).astype(float)
    if event == "podium":
        return (positions <= 3).astype(float)
    return (positions <= POINTS_POSITIONS).astype(float)  # "points"


# --------------------------------------------------------------------------
# Simulating the holdout (T-505..T-507's input)
# --------------------------------------------------------------------------


def simulate_holdout(
    built: Experiment,
    model: TrainedRanker,
    params: SimulationParameters,
    split: ChronologicalSplit,
    *,
    n_simulations: int = 1000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Simulate every holdout race and attach the actual outcome to each row.

    One row per (season, round, driver_id), carrying the simulated
    probabilities, the model's own interval (FR-402), and what actually
    happened — the frame T-505..T-507 all read.

    `predict_with_interval` ranks whatever frame it is handed as *one* race —
    exactly what `TrainedRanker.predict` documents and what
    `evaluate.harness.predict_all_races` calls out race-by-race for. Calling
    it once on the whole holdout would rank 1..len(holdout) globally instead
    of 1..field_size per race, which is not a subtle difference: every
    downstream interval and probability here would be checked against a
    number that does not mean "finishing position".
    """
    frame = built.frame
    holdout = frame[split.holdout_mask(frame)]
    if holdout.empty:
        raise CalibrationError("no holdout races to simulate")

    race_columns = [c for c in REQUIRED_RACE_COLUMNS if c != "predicted_position"]
    optional_columns = [c for c in ("weather_prior_wet",) if c in holdout.columns]

    pieces = []
    for (season, rnd), race in holdout.groupby(["season", "round"], sort=True):
        predicted = model.predict_with_interval(race)
        race = race.join(
            predicted[
                ["predicted_position", "predicted_position_lower", "predicted_position_upper"]
            ]
        )
        race_frame = race[[*race_columns, "predicted_position", *optional_columns]]
        result = simulate_race(
            race_frame, params, n_simulations=n_simulations, seed=seed
        )
        result = result.merge(
            race[
                [
                    "driver_id",
                    "target_position",
                    "predicted_position_lower",
                    "predicted_position_upper",
                ]
            ],
            on="driver_id",
            how="left",
        )
        result["season"] = season
        result["round"] = rnd
        pieces.append(result)

    simulated = pd.concat(pieces, ignore_index=True)
    simulated["actual_win"] = _event_outcome(simulated["target_position"], "win")
    simulated["actual_podium"] = _event_outcome(simulated["target_position"], "podium")
    simulated["actual_points"] = _event_outcome(simulated["target_position"], "points")
    simulated["interval_contains_actual"] = simulated["target_position"].between(
        simulated["predicted_position_lower"], simulated["predicted_position_upper"]
    )
    return simulated


# --------------------------------------------------------------------------
# T-505: reliability curves
# --------------------------------------------------------------------------


def reliability_curve(
    probabilities: pd.Series, outcomes: pd.Series, *, n_bins: int = 10
) -> pd.DataFrame:
    """FR-702: binned predicted-vs-observed frequency for one event type."""
    if len(probabilities) != len(outcomes):
        raise CalibrationError("probabilities and outcomes must be the same length")
    if len(probabilities) == 0:
        raise CalibrationError("cannot build a reliability curve from no rows")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = pd.cut(probabilities, bins=edges, include_lowest=True)
    work = pd.DataFrame(
        {"probability": probabilities.to_numpy(), "outcome": outcomes.to_numpy(), "bin": bins}
    )
    curve = (
        work.groupby("bin", observed=True)
        .agg(
            mean_predicted=("probability", "mean"),
            observed_frequency=("outcome", "mean"),
            count=("outcome", "size"),
        )
        .reset_index()
    )
    curve["bin"] = curve["bin"].astype(str)
    return curve


def max_reliability_gap(curve: pd.DataFrame, *, min_count: int = MIN_RELIABILITY_BIN_COUNT) -> float:
    """The largest |predicted - observed| gap across the curve's occupied bins.

    A bin with `count < min_count` is excluded, not just an empty one. The
    real corpus's win event is rare enough (roughly one winner per 20-car
    field) that its top probability bins routinely hold a single row, and that
    row's observed frequency is then either exactly 0.0 or exactly 1.0 by
    construction — a coin-flip outcome standing in for "the largest gap in the
    whole curve" the first time this ran, at 0.499 against a same-corpus
    median bin gap under 0.05. This is the same judgement `EventBaselines`,
    `shrink_towards`, and FR-212's cold-start flag all make elsewhere: a
    single observation is not evidence, thin or otherwise, and treating it as
    equal to a 384-row bin is not a stricter calibration check, it is a
    noisier one.
    """
    occupied = curve[curve["count"] >= min_count]
    if occupied.empty:
        return float("nan")
    return float((occupied["mean_predicted"] - occupied["observed_frequency"]).abs().max())


# --------------------------------------------------------------------------
# T-506: Brier scores, and T-511's proper scoring rule
# --------------------------------------------------------------------------


def brier_score(probabilities: pd.Series, outcomes: pd.Series) -> float:
    """FR-703, FR-753: the proper scoring rule this project reports."""
    if len(probabilities) != len(outcomes):
        raise CalibrationError("probabilities and outcomes must be the same length")
    p = probabilities.to_numpy(dtype="float64")
    o = outcomes.to_numpy(dtype="float64")
    return float(np.mean((p - o) ** 2))


def _bootstrap_race_mean(
    per_race: pd.Series,
    *,
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = SEED,
) -> Interval:
    """The same percentile-over-races bootstrap as `evaluate.bootstrap`.

    Not imported from there because that module's public functions resample
    the ranking-`Metric` battery specifically; this resamples one already
    per-race series (a Brier score or a coverage rate) at a time.
    """
    values = per_race.to_numpy(dtype="float64")
    n = len(values)
    if n == 0:
        raise CalibrationError("no races to bootstrap")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    means = values[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [100 * tail, 100 * (1.0 - tail)])
    return Interval(point=float(values.mean()), low=float(low), high=float(high), confidence=confidence)


def brier_score_by_race(simulated: pd.DataFrame, *, event: str, probability_column: str) -> pd.Series:
    """Per-race Brier score, the unit T-307's bootstrap discipline resamples."""
    outcome_column = f"actual_{event}"
    work = simulated.assign(
        _sq_error=(simulated[probability_column] - simulated[outcome_column]) ** 2
    )
    return work.groupby(["season", "round"])["_sq_error"].mean()


# --------------------------------------------------------------------------
# T-507: interval coverage
# --------------------------------------------------------------------------


def interval_coverage(simulated: pd.DataFrame) -> pd.Series:
    """Per-race observed containment rate of FR-402's stated interval."""
    return simulated.groupby(["season", "round"])["interval_contains_actual"].mean()


# --------------------------------------------------------------------------
# T-508, T-509: the threshold and the label that cannot be skipped
# --------------------------------------------------------------------------


def calibration_status(measured_error: float, *, threshold: float = CALIBRATION_ERROR_THRESHOLD) -> str:
    """FR-705: `CALIBRATED` only when the measured error is within threshold.

    A NaN measurement — no data to measure from — is treated as failing,
    never as passing: FR-705 requires the label for calibration that is
    *unmeasured*, and NaN is exactly that state.
    """
    if not np.isfinite(measured_error):
        return "UNCALIBRATED"
    return "CALIBRATED" if measured_error <= threshold else "UNCALIBRATED"


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventCalibration:
    event: str
    reliability_curve: pd.DataFrame
    max_reliability_gap: float
    brier_model: Interval
    brier_climatological: Interval
    brier_grid: Interval
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "reliability_curve": self.reliability_curve.to_dict(orient="records"),
            "max_reliability_gap": self.max_reliability_gap,
            "brier_score": {
                "model": self.brier_model.as_dict(),
                "climatological_baseline": self.brier_climatological.as_dict(),
                "grid_baseline": self.brier_grid.as_dict(),
            },
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    events: dict[str, EventCalibration]
    interval_coverage: Interval
    stated_interval_coverage: float
    coverage_status: str
    n_holdout_races: int
    n_holdout_rows: int
    threshold: float = CALIBRATION_ERROR_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_holdout_races": self.n_holdout_races,
            "n_holdout_rows": self.n_holdout_rows,
            "events": {name: ev.as_dict() for name, ev in self.events.items()},
            "interval_coverage": {
                "stated": self.stated_interval_coverage,
                "observed": self.interval_coverage.as_dict(),
                "status": self.coverage_status,
            },
            # FR-706: no call site may lift a claim out of here without also
            # carrying whether *anything* in this report passed.
            "any_uncalibrated": self.coverage_status == "UNCALIBRATED"
            or any(ev.status == "UNCALIBRATED" for ev in self.events.values()),
        }


def build_calibration_report(
    simulated: pd.DataFrame,
    *,
    training_frame: pd.DataFrame,
    stated_interval_coverage: float,
    threshold: float = CALIBRATION_ERROR_THRESHOLD,
) -> CalibrationReport:
    events: dict[str, EventCalibration] = {}
    for event in EVENT_TYPES:
        probability_column = f"{event}_probability"
        outcome_column = f"actual_{event}"
        baselines = fit_event_baselines(training_frame, event=event)

        curve = reliability_curve(simulated[probability_column], simulated[outcome_column])
        gap = max_reliability_gap(curve)

        climatological_probs = pd.Series(
            baselines.climatological_rate, index=simulated.index
        )
        grid_probs = simulated["grid_position"].map(baselines.grid_probability)

        model_brier = _bootstrap_race_mean(
            brier_score_by_race(simulated, event=event, probability_column=probability_column)
        )
        clim_brier = _bootstrap_race_mean(
            brier_score_by_race(
                simulated.assign(**{probability_column: climatological_probs}),
                event=event,
                probability_column=probability_column,
            )
        )
        grid_brier = _bootstrap_race_mean(
            brier_score_by_race(
                simulated.assign(**{probability_column: grid_probs}),
                event=event,
                probability_column=probability_column,
            )
        )

        events[event] = EventCalibration(
            event=event,
            reliability_curve=curve,
            max_reliability_gap=gap,
            brier_model=model_brier,
            brier_climatological=clim_brier,
            brier_grid=grid_brier,
            status=calibration_status(gap, threshold=threshold),
        )

    coverage_by_race = interval_coverage(simulated)
    coverage_interval = _bootstrap_race_mean(coverage_by_race)
    coverage_gap = abs(coverage_interval.point - stated_interval_coverage)

    return CalibrationReport(
        events=events,
        interval_coverage=coverage_interval,
        stated_interval_coverage=stated_interval_coverage,
        coverage_status=calibration_status(coverage_gap, threshold=threshold),
        n_holdout_races=int(simulated[["season", "round"]].drop_duplicates().shape[0]),
        n_holdout_rows=int(len(simulated)),
        threshold=threshold,
    )


def run_calibration(
    *,
    split: ChronologicalSplit | None = None,
    experiment: Experiment | None = None,
    model: TrainedRanker | None = None,
    n_simulations: int = 1000,
    seed: int = SEED,
    path: Path | None = None,
) -> dict[str, Any]:
    """Disk to a committed calibration record in one call (FR-506's discipline).

    Trains the model if one was not supplied — on the *same* split the
    calibration is then measured against, so the number this writes cannot
    describe a model calibration never actually saw. Fits `SimulationParameters`
    from the raw enriched corpus's training rows only, per FR-751.
    """
    built = experiment if experiment is not None else prepare()
    split = split or default_split()
    fitted_model = model if model is not None else train_model(built, split=split)

    raw = enrich(load_corpus(split.seasons))
    training_raw, _ = split.split(raw)
    pace_std = pace_noise_std_from_interval(
        half_width=(fitted_model.interval.upper_offset - fitted_model.interval.lower_offset)
        / 2,
        coverage=fitted_model.interval.coverage,
    )
    params = fit_simulation_parameters(training_raw, pace_noise_std=pace_std)

    simulated = simulate_holdout(
        built, fitted_model, params, split, n_simulations=n_simulations, seed=seed
    )
    training_frame, _ = split.split(built.frame)
    report = build_calibration_report(
        simulated,
        training_frame=training_frame,
        stated_interval_coverage=fitted_model.interval.coverage,
    )

    record: dict[str, Any] = {
        "split": split.describe(),
        "model_code_commit": fitted_model.code_commit,
        "n_simulations": n_simulations,
        "seed": seed,
        "safety_car_proxy": {
            "corpus_rate": params.corpus_safety_car_rate,
            "note": (
                "measured as the rate of >=2 retirements within the first "
                "5 laps; not a confirmed safety-car deployment count (T-510)"
            ),
        },
        "calibration": report.as_dict(),
    }

    holdout = split.holdout
    target = path or (RESULTS_DIR / f"calibration_{holdout[0]}_{holdout[-1]}.json")
    written = write_calibration_results(record, target)
    logger.info("wrote %s", written)
    return record


def write_calibration_results(record: dict[str, Any], path: Path | None = None) -> Path:
    """FR-506-style machine-readable record, alongside the human-read one."""
    target = path or (RESULTS_DIR / "calibration.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return target


__all__ = [
    "CALIBRATION_ERROR_THRESHOLD",
    "EVENT_TYPES",
    "MIN_RELIABILITY_BIN_COUNT",
    "CalibrationError",
    "CalibrationReport",
    "EventBaselines",
    "EventCalibration",
    "brier_score",
    "brier_score_by_race",
    "build_calibration_report",
    "calibration_status",
    "fit_event_baselines",
    "interval_coverage",
    "max_reliability_gap",
    "run_calibration",
    "reliability_curve",
    "simulate_holdout",
    "write_calibration_results",
]
