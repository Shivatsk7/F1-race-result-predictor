"""The headline run: model against all three baselines, in one call (AC-3, AC-4).

Everything this module needs already existed — the corpus loader, the feature
build, the ranker, the harness, the bootstrap. What did not exist was one
function that puts them in the same order every time. That is the whole point
of it: FR-506 promises a machine-readable record of a run, and a record whose
provenance is "somebody assembled this in a session once" is not one.

The order is fixed here rather than left to the caller because it is the order
that makes the comparison fair: one corpus, built once; one split, applied by
the trainer and by the harness alike; one scoring path for the model and for
the baselines (FR-304). A caller who could reorder those could, without
intending to, train on the rows the model is about to be scored on.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from f1pred.config import RESULTS_DIR
from f1pred.dataset import Experiment, prepare
from f1pred.evaluate.harness import evaluate, render_verdict, write_results
from f1pred.evaluate.split import ChronologicalSplit, default_split
from f1pred.models.base import Predictor
from f1pred.models.baselines import (
    GridOrderBaseline,
    RandomOrderControl,
    StandingsOrderBaseline,
)
from f1pred.models.ranker import Hyperparameters, TrainedRanker, train

logger = logging.getLogger(__name__)


def baselines() -> tuple[Predictor, ...]:
    """FR-301 and FR-302, constructed here so no run can quietly drop one.

    A "headline" that omitted the standings baseline would still render a
    verdict, and the verdict would read exactly the same — the absence would
    show up nowhere except as a missing line nobody was looking for. The random
    control is separate because the harness averages it over many draws.
    """
    return (GridOrderBaseline(), StandingsOrderBaseline())


def train_model(
    built: Experiment,
    *,
    split: ChronologicalSplit,
    params: Hyperparameters | None = None,
    name: str = "xgboost_ranker",
) -> TrainedRanker:
    """Fit the ranker on this experiment's frame and its declared features."""
    return train(
        built.frame,
        split=split,
        features=built.feature_names,
        params=params,
        name=name,
    )


def run(
    *,
    seasons: Iterable[int] | None = None,
    split: ChronologicalSplit | None = None,
    params: Hyperparameters | None = None,
    experiment: Experiment | None = None,
    extra_predictors: Sequence[Predictor] = (),
    random_control: bool = True,
    label: str = "model_vs_baselines",
    path: Path | None = None,
) -> tuple[dict[str, Any], TrainedRanker]:
    """Build, train, score, and write. Returns the record and the model.

    The random control is on by default and costs a thousand scored draws,
    because FR-303's floor is what says the harness itself is sound: a metric
    that gives random orderings a respectable score is measuring something
    other than skill, and that is worth re-establishing on every run rather
    than trusting a number from a previous one.
    """
    built = experiment if experiment is not None else prepare(seasons)
    split = split or default_split()
    model = train_model(built, split=split, params=params)

    record = evaluate(
        built.frame,
        (*baselines(), model, *extra_predictors),
        split=split,
        random_control=RandomOrderControl().predictors() if random_control else None,
    )

    # FR-404's sixth field is the evaluation metrics, so it can only be filled
    # in after the harness has run. Recording the model's own line rather than
    # the whole record keeps the model's provenance about the model.
    scored = model.with_evaluation_metrics(record["predictors"][model.name]["metrics"])
    record["model"] = scored.metadata()
    record["model"]["interval"] = scored.interval.as_dict()
    # A run without weather is a different experiment from one with it, and
    # after the fact only this field distinguishes them (FR-206, FR-214).
    record["absent_sources"] = list(built.absent_sources)
    record["corpus_rows"] = len(built)

    target = path or RESULTS_DIR / _filename(label, split)
    written = write_results(record, target)
    logger.info("wrote %s", written)
    return record, scored


def _filename(label: str, split: ChronologicalSplit) -> str:
    """Name the file after the holdout it reports, as P3's files already are.

    A results file called `model.json` is one rerun away from describing a
    different holdout than the report that cites it.
    """
    holdout = split.holdout
    return f"{label}_{holdout[0]}_{holdout[-1]}.json"


def main() -> int:  # pragma: no cover - thin entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    record, model = run()
    print(render_verdict(record))
    print(f"\ntrained on {model.training_rows} rows, {model.code_commit}")
    return 0


__all__ = [
    "baselines",
    "main",
    "run",
    "train_model",
]


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
