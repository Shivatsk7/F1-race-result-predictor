"""What the model scores on races it has already seen (FR-405, FR-902).

The evaluation harness reports holdout performance and nothing else, which is
correct — holdout performance is the only number that answers the project's
question. But it means a model that has memorised its training races and a
model that has learned something look identical from the outside, and both
report a plausible holdout score.

P4 hit exactly that. The first trained ranker scored **0.9625 in-sample against
0.6302 on the holdout**: four hundred trees at depth six fit two hundred and
seventy-four races almost exactly. Every test was green, the pipeline was
correct end to end, and the number was simply an overfitted model's number.
Nothing in the harness asks the question that exposes it, so nothing caught it;
it was found by hand.

This module asks that question, and commits the answer. It exists for two
reasons beyond the historical one:

* **FR-902 and AC-7.** Those two numbers are cited on the task board as the
  justification for early stopping. A performance figure in project
  documentation must resolve to a committed file in `results/`, and until this
  module existed they resolved to a session nobody else could rerun — which is
  the specific failure the README says this rebuild was for.
* **A regression guard with a number attached.** The gap between in-sample and
  holdout is a property of the configuration, not of a run. If a future change
  to the feature set or the hyperparameters reopens it, the reopening is
  visible here rather than as a holdout score that drifts for unclear reasons.

The gap is diagnostic, never a target. Driving it to zero is not the goal —
some gap is expected and healthy — and nothing here should be optimised
against. It is reported so that a *large* one cannot pass unnoticed again.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from f1pred.config import RESULTS_DIR
from f1pred.dataset import Experiment, prepare
from f1pred.evaluate.harness import score_predictor, write_results
from f1pred.evaluate.metrics import PRIMARY_METRIC
from f1pred.evaluate.split import ChronologicalSplit, default_split
from f1pred.models.baselines import GridOrderBaseline
from f1pred.models.ranker import Hyperparameters, TrainedRanker, train

logger = logging.getLogger(__name__)

#: Above this, the fit is reproducing its training races rather than learning
#: from them. Not a pass/fail threshold — it is the line above which the
#: results file says so in words, so that a reader who skims the numbers still
#: sees the finding. The unstopped P4 configuration sat at 0.33.
WIDE_GAP: Final[float] = 0.15

STOPPED_ARM: Final[str] = "early_stopping"
UNSTOPPED_ARM: Final[str] = "no_early_stopping"


def in_sample_and_holdout(
    model: TrainedRanker, frame: pd.DataFrame, *, split: ChronologicalSplit
) -> dict[str, float]:
    """Score one model on both sides of the boundary.

    The training side is scored through `score_predictor`, the same function
    the harness uses for the holdout. A separate in-sample scorer could drift
    from it, and then the gap would be partly an artefact of two code paths
    rather than a property of the model.
    """
    training, holdout = split.split(frame)
    metric = PRIMARY_METRIC.name
    fitted = float(score_predictor(model, training)[metric].mean())
    unseen = float(score_predictor(model, holdout)[metric].mean())
    return {
        "in_sample": fitted,
        "holdout": unseen,
        "gap": fitted - unseen,
    }


def _arm(
    built: Experiment,
    split: ChronologicalSplit,
    params: Hyperparameters,
    name: str,
) -> dict[str, Any]:
    model = train(
        built.frame, split=split, features=built.feature_names, params=params, name=name
    )
    scores = in_sample_and_holdout(model, built.frame, split=split)
    return {
        "arm": name,
        "early_stopping_rounds": params.early_stopping_rounds,
        "n_estimators_ceiling": params.n_estimators,
        "trees_kept": model.trees,
        **scores,
        "memorising": scores["gap"] > WIDE_GAP,
    }


def run(
    *,
    seasons: Iterable[int] | None = None,
    split: ChronologicalSplit | None = None,
    params: Hyperparameters | None = None,
    experiment: Experiment | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Both arms — as configured, and with early stopping switched off.

    The unstopped arm is run every time rather than quoted from history. It is
    the control: without it the stopped arm's gap is a number with nothing to
    compare against, and the claim that early stopping is what closed it would
    rest on a measurement nobody repeats.
    """
    built = experiment if experiment is not None else prepare(seasons)
    split = split or default_split()
    configured = params or Hyperparameters()
    unstopped = Hyperparameters(
        **{**configured.as_dict(), "early_stopping_rounds": None}
    )

    arms = {
        STOPPED_ARM: _arm(built, split, configured, STOPPED_ARM),
        UNSTOPPED_ARM: _arm(built, split, unstopped, UNSTOPPED_ARM),
    }

    # The grid baseline on both sides, because a gap means nothing without
    # knowing that the training seasons are not simply easier races. The grid
    # cannot memorise anything, so whatever gap it shows is the corpus's, not
    # the model's.
    training, holdout = split.split(built.frame)
    metric = PRIMARY_METRIC.name
    grid = GridOrderBaseline()
    reference = {
        "in_sample": float(score_predictor(grid, training)[metric].mean()),
        "holdout": float(score_predictor(grid, holdout)[metric].mean()),
    }
    reference["gap"] = reference["in_sample"] - reference["holdout"]

    record = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "experiment": "fit_diagnostic",
        "question": (
            "does the model reproduce its training races rather than learn "
            "from them?"
        ),
        "primary_metric": metric,
        "wide_gap_threshold": WIDE_GAP,
        "split": split.describe(),
        "corpus_rows": len(built),
        "arms": arms,
        "grid_order_reference": reference,
        "finding": _finding(arms, reference),
    }
    written = write_results(record, path or RESULTS_DIR / "fit_diagnostic_2023_2025.json")
    logger.info("wrote %s", written)
    return record


def _finding(arms: dict[str, dict[str, Any]], reference: dict[str, float]) -> str:
    """The sentence, computed from the arms rather than written by hand."""
    stopped, unstopped = arms[STOPPED_ARM], arms[UNSTOPPED_ARM]
    return (
        f"Without early stopping the ranker scores {unstopped['in_sample']:.4f} "
        f"on the races it trained on against {unstopped['holdout']:.4f} on the "
        f"holdout, a gap of {unstopped['gap']:.4f} over "
        f"{unstopped['trees_kept']} trees — it is reproducing its training "
        f"races. With early stopping the gap is {stopped['gap']:.4f} over "
        f"{stopped['trees_kept']} trees and the holdout score rises to "
        f"{stopped['holdout']:.4f}. The grid baseline, which cannot memorise "
        f"anything, shows a gap of {reference['gap']:+.4f} on the same split, "
        f"so the training seasons are not simply the easier races."
    )


def render(record: dict[str, Any]) -> str:
    lines = [
        f"Fit diagnostic ({record['primary_metric']}), "
        f"holdout {record['split']['holdout_seasons'][0]}-"
        f"{record['split']['holdout_seasons'][-1]}.",
        "",
        f"{'arm':<20} {'trees':>6} {'in-sample':>10} {'holdout':>9} {'gap':>8}",
    ]
    for arm in record["arms"].values():
        lines.append(
            f"{arm['arm']:<20} {arm['trees_kept']:>6} {arm['in_sample']:>10.4f} "
            f"{arm['holdout']:>9.4f} {arm['gap']:>8.4f}"
            + ("   MEMORISING" if arm["memorising"] else "")
        )
    ref = record["grid_order_reference"]
    lines.append(
        f"{'grid_order':<20} {'-':>6} {ref['in_sample']:>10.4f} "
        f"{ref['holdout']:>9.4f} {ref['gap']:>8.4f}"
    )
    return "\n".join([*lines, "", record["finding"]])


def main() -> int:  # pragma: no cover - thin entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(render(run()))
    return 0


__all__ = [
    "STOPPED_ARM",
    "UNSTOPPED_ARM",
    "WIDE_GAP",
    "in_sample_and_holdout",
    "main",
    "render",
    "run",
]


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
