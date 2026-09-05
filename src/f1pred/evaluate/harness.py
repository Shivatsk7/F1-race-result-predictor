"""Running an evaluation (FR-304, FR-305, FR-505, FR-506, FR-507).

One function scores every predictor. The model, the grid baseline and a
thousand random orderings all enter through `evaluate`, are handed the same
holdout frame, and come out with the same nine metrics and the same intervals.
FR-304 asks for that, and doing it structurally means it cannot be forgotten:
there is no second path to score the model on.

The verdict is computed, not written. FR-507 requires the evaluation to say so
explicitly when the model fails to beat the grid, and a human writing that
sentence is a human who might not.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from f1pred import render
from f1pred.provenance import stamp
from f1pred.config import RESULTS_DIR, SEED
from f1pred.evaluate.bootstrap import (
    DEFAULT_REPLICATES,
    Interval,
    bootstrap_delta,
    bootstrap_metrics,
    mean_of_intervals,
)
from f1pred.evaluate.metrics import (
    METRICS,
    PRIMARY_METRIC,
    RACE_KEYS,
    Metric,
    per_race_scores,
)
from f1pred.evaluate.split import ChronologicalSplit
from f1pred.models.base import Predictor
from f1pred.reference import circuits

#: FR-504 measures every delta against this one.
REFERENCE_BASELINE: Final[str] = "grid_order"

ACTUAL_COLUMN: Final[str] = "target_position"


class EvaluationError(ValueError):
    """Raised when an evaluation cannot be run as specified."""


@dataclass(slots=True)
class PredictorResult:
    name: str
    metrics: dict[str, Interval]
    per_race: pd.DataFrame
    #: FR-303: set when this line summarises many seeded draws rather than one run.
    draws: int = 1
    delta_vs_reference: dict[str, Interval] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "races_scored": int(len(self.per_race)),
            "draws": self.draws,
            "metrics": {k: v.as_dict() for k, v in self.metrics.items()},
        }
        if self.delta_vs_reference:
            out["delta_vs_" + REFERENCE_BASELINE] = {
                k: v.as_dict() for k, v in self.delta_vs_reference.items()
            }
        return out


def predict_all_races(
    predictor: Predictor, frame: pd.DataFrame
) -> pd.Series:
    """Run a predictor race by race and reassemble, index preserved.

    Race by race because a finishing order is only defined within a race:
    a predictor handed the whole holdout at once could return positions
    running past the field size, and every metric would quietly accept them.
    """
    pieces = [
        predictor.predict(race)
        for _, race in frame.groupby(list(RACE_KEYS), sort=True)
    ]
    return pd.concat(pieces).reindex(frame.index)


def score_predictor(
    predictor: Predictor,
    frame: pd.DataFrame,
    *,
    metrics: Sequence[Metric] = METRICS,
) -> pd.DataFrame:
    """The single scoring path. Every predictor goes through here (FR-304)."""
    if ACTUAL_COLUMN not in frame.columns:
        raise EvaluationError(
            f"holdout frame has no {ACTUAL_COLUMN!r}; the FR-001 target must be "
            f"built before evaluation"
        )
    predicted = predict_all_races(predictor, frame)
    return per_race_scores(
        frame[list(RACE_KEYS)], predicted, frame[ACTUAL_COLUMN], metrics=metrics
    )


def _segment(
    per_race: pd.DataFrame, frame: pd.DataFrame, metric: Metric
) -> dict[str, dict[str, float]]:
    """FR-505: by season, circuit type, and overtaking tier.

    Segmented on the primary metric only. Nine metrics times three segmentations
    is a wall of numbers nobody reads; the requirement's purpose is that
    aggregate performance cannot conceal systematic failure, and one metric
    broken three ways serves that.
    """
    races = (
        frame[[*RACE_KEYS, "circuit_id"]]
        .drop_duplicates()
        .merge(per_race, on=list(RACE_KEYS), how="right")
    )
    races["circuit_type"] = races["circuit_id"].map(
        lambda c: circuits.get(c).circuit_type.value
    )
    races["overtaking_tier"] = races["circuit_id"].map(
        lambda c: circuits.get(c).overtaking_tier.value
    )
    return {
        by: {
            str(key): float(group[metric.name].mean(skipna=True))
            for key, group in races.groupby(by, sort=True)
        }
        for by in ("season", "circuit_type", "overtaking_tier")
    }


def evaluate(
    frame: pd.DataFrame,
    predictors: Iterable[Predictor],
    *,
    split: ChronologicalSplit,
    random_control: Iterable[Predictor] | None = None,
    metrics: Sequence[Metric] = METRICS,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Score everything on the holdout and assemble the results record.

    `frame` is the *whole* corpus; the holdout is taken here rather than by the
    caller, so no call site can evaluate on a slice of its own choosing.
    """
    _, holdout = split.split(frame)
    if holdout.empty:
        raise EvaluationError(
            f"holdout seasons {split.holdout} select no rows from this frame"
        )

    results: dict[str, PredictorResult] = {}
    for predictor in predictors:
        per_race = score_predictor(predictor, holdout, metrics=metrics)
        results[predictor.name] = PredictorResult(
            name=predictor.name,
            metrics=bootstrap_metrics(
                per_race, metrics=metrics, replicates=replicates, seed=seed
            ),
            per_race=per_race,
        )

    # FR-303: the random floor is the mean over many seeded draws, each scored
    # through the identical path above.
    if random_control is not None:
        draws = list(random_control)
        if draws:
            per_draw = [
                bootstrap_metrics(
                    score_predictor(d, holdout, metrics=metrics),
                    metrics=metrics,
                    replicates=64,
                    seed=seed + i,
                )
                for i, d in enumerate(draws)
            ]
            reference_per_race = score_predictor(draws[0], holdout, metrics=metrics)
            results[draws[0].name] = PredictorResult(
                name=draws[0].name,
                metrics=mean_of_intervals(per_draw),
                per_race=reference_per_race,
                draws=len(draws),
            )

    if REFERENCE_BASELINE not in results:
        raise EvaluationError(
            f"the {REFERENCE_BASELINE!r} baseline was not among the predictors; "
            f"FR-504 measures every delta against it, so it is not optional"
        )

    reference = results[REFERENCE_BASELINE]
    for name, result in results.items():
        if name == REFERENCE_BASELINE or result.draws > 1:
            continue
        result.delta_vs_reference = bootstrap_delta(
            result.per_race,
            reference.per_race,
            metrics=metrics,
            replicates=replicates,
            seed=seed,
        )

    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "primary_metric": PRIMARY_METRIC.name,
        "split": split.describe(),
        "holdout_races": int(holdout[list(RACE_KEYS)].drop_duplicates().shape[0]),
        "holdout_rows": int(len(holdout)),
        # FR-305: baselines sit in the same structure as everything else.
        "predictors": {name: r.as_dict() for name, r in results.items()},
        "segmentation": {
            name: _segment(r.per_race, holdout, PRIMARY_METRIC)
            for name, r in results.items()
        },
        "verdict": verdict(results),
    }


def verdict(results: dict[str, PredictorResult]) -> dict[str, Any]:
    """FR-507: state plainly whether anything beat the grid, in both directions."""
    reference = results[REFERENCE_BASELINE]
    metric = PRIMARY_METRIC.name
    lines: dict[str, Any] = {
        "metric": metric,
        "reference": REFERENCE_BASELINE,
        "reference_score": reference.metrics[metric].point,
        "beaten_by": [],
        "not_beaten_by": [],
    }
    for name, result in results.items():
        if name == REFERENCE_BASELINE or not result.delta_vs_reference:
            continue
        delta = result.delta_vs_reference[metric]
        entry = {"name": name, **delta.as_dict()}
        beats = delta.excludes_zero and delta.point > 0
        (lines["beaten_by"] if beats else lines["not_beaten_by"]).append(entry)
    lines["any_predictor_beats_the_grid"] = bool(lines["beaten_by"])
    return lines


def _delta(entry: dict[str, Any]) -> str:
    """A delta and its interval, at the precision the interval supports.

    FR-804 rule 2 applies here and not only to tables: this sentence is the
    project's headline claim, and it is the last place a spare decimal should
    be allowed to imply a resolution the bootstrap did not find. Signed, so a
    delta that rounds to zero still shows which side of zero it fell.
    """
    places = render.decimals_supported_by(
        max(entry["point"] - entry["ci_low"], entry["ci_high"] - entry["point"])
    )
    return (
        f"{entry['point']:+.{places}f} "
        f"(95% CI {entry['ci_low']:+.{places}f} to {entry['ci_high']:+.{places}f}"
        f"{'' if entry['excludes_zero'] else ', includes zero'})"
    )


def render_verdict(record: dict[str, Any]) -> str:
    """The sentence FR-507 requires, computed rather than written."""
    v = record["verdict"]
    metric, ref = v["metric"], v["reference"]
    reference_metric = record["predictors"][ref]["metrics"][metric]
    head = (
        f"Primary metric: {metric}. "
        f"{ref} scores "
        + render.format_measurement(
            reference_metric["point"],
            low=reference_metric["ci_low"],
            high=reference_metric["ci_high"],
        )
        + f" on {record['holdout_races']} holdout races "
        f"({record['split']['holdout_seasons'][0]}-"
        f"{record['split']['holdout_seasons'][-1]})."
    )
    if not v["beaten_by"] and not v["not_beaten_by"]:
        return head + "\n\nNo predictor was compared against it."
    if v["any_predictor_beats_the_grid"]:
        best = max(v["beaten_by"], key=lambda e: e["point"])
        body = f"\n\nBEATS THE GRID: {best['name']} by {_delta(best)}."
    else:
        body = "\n\nDOES NOT BEAT THE GRID BASELINE."
    for entry in v["not_beaten_by"]:
        body += f"\n  {entry['name']}: {_delta(entry)}."
    return head + body


def write_results(record: dict[str, Any], path: Path | None = None) -> Path:
    """FR-506: the machine-readable file, alongside whatever a human reads.

    Stamped on the way out (AC-11, AC-21). Doing it here rather than at each
    caller is the point: a results file whose provenance depends on its author
    remembering is a results file that will eventually lack it.
    """
    stamp(record)
    target = path or (RESULTS_DIR / "baselines.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return target


__all__ = [
    "ACTUAL_COLUMN",
    "EvaluationError",
    "PredictorResult",
    "REFERENCE_BASELINE",
    "evaluate",
    "predict_all_races",
    "render_verdict",
    "score_predictor",
    "verdict",
    "write_results",
]
