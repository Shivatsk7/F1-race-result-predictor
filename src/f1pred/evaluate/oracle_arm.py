"""T-407: what knowing the answer is worth, measured rather than argued (FR-207).

The oracle arm is not a model anyone would ship. It is a **ruler**. FR-207 keeps
a labelled feature set that includes post-race columns — whether the car was
classified, how much of the distance it covered, what the weather actually did —
and this module trains the same ranker on it, on the same corpus and the same
split as the primary arm, so the two can be subtracted. The number that comes
out is the size of the gap between predicting a race and reading its result.

That gap is the point. v0 carried `weather_mixed`, computed from the race that
had already happened, for four months, and reported the resulting score as
skill. Nothing in v0 could say how much of that score was the leak, because
nothing had measured what a leak is worth. Here it is measured once, on the
project's own corpus, and any future score that lands in that territory is
suspect for a stated reason rather than a vague one.

**Nothing in the oracle arm is admissible for any honest claim about
performance.** It cannot be run at prediction time — its inputs do not exist
until the race is over — so its score is not a forecast of anything. The record
this module writes says so in a field of its own, because a results file that
merely omitted the caveat would read exactly like an honest one.

Three structural decisions, each a refusal:

* Each arm is built by its own `prepare(...)` call, so each frame carries its
  own feature columns and neither arm can be scored on a matrix built for the
  other.
* Both arms are then scored in **one** `evaluate` call on **one** frame — the
  oracle frame, whose columns are a superset of the primary frame's — so
  FR-304's "identical holdout" is a fact about the call rather than a claim
  about two calls. `assert_arms_comparable` checks the superset relation and
  the value-for-value equality of the shared columns first: if the primary
  features differed between the two builds, scoring the primary model on the
  oracle frame would silently score a different model than the one trained.
* The headline delta is bootstrapped **paired**, oracle minus primary within
  each replicate (D-5, FR-504). Subtracting two independently bootstrapped
  intervals throws away the pairing — both arms saw the same races, and a race
  that is hard for one is hard for the other — and reports an uncertainty
  several times wider than the one that was actually measured.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from f1pred.config import RAW_DIR, RESULTS_DIR
from f1pred.dataset import Experiment, prepare
from f1pred.evaluate.bootstrap import DEFAULT_REPLICATES, Interval, bootstrap_delta
from f1pred.evaluate.harness import (
    ACTUAL_COLUMN,
    evaluate,
    score_predictor,
    write_results,
)
from f1pred.evaluate.metrics import METRICS, PRIMARY_METRIC
from f1pred.evaluate.runner import baselines, train_model
from f1pred.evaluate.split import ChronologicalSplit, default_split
from f1pred.features.build import KEY_COLUMNS, build_oracle, build_primary
from f1pred.features.registry import REGISTRY, Registry
from f1pred.models.baselines import RandomOrderControl
from f1pred.models.ranker import Hyperparameters, TrainedRanker

logger = logging.getLogger(__name__)

#: The two predictor names. Distinct because the harness keys its results by
#: name: two predictors sharing one would overwrite each other in the record,
#: and the surviving line would be reported as whichever arm happened to be
#: scored last.
PRIMARY_ARM: Final[str] = "xgboost_ranker"
ORACLE_ARM: Final[str] = "xgboost_ranker_oracle"

#: Written into the record verbatim. FR-207 permits the oracle set for
#: upper-bound analysis and for nothing else, and the file has to carry that
#: sentence itself — a reader who opens the JSON without the module beside it
#: would otherwise see two scores of apparently equal standing.
INADMISSIBLE: Final[str] = (
    "INADMISSIBLE. The oracle arm reads post-race features and cannot be run "
    "before a race has happened. Its score is not a forecast and must never be "
    "quoted as model performance (FR-207). It exists here only to size the gap "
    "reported under `inflation`."
)


class OracleArmError(ValueError):
    """Raised when the two arms cannot honestly be compared."""


@dataclass(frozen=True, slots=True)
class OracleComparison:
    """One paired comparison: the record, and the two models behind it.

    The models are returned alongside the record rather than discarded because
    the record is a summary and the question "which features did the oracle arm
    actually fit on?" is answerable only from the model. Frozen for the same
    reason `TrainedRanker` is: a comparison whose parts can be swapped after the
    fact is one whose record can stop describing it.
    """

    record: dict[str, Any]
    primary: TrainedRanker
    oracle: TrainedRanker
    path: Path | None = None


# --------------------------------------------------------------------------
# The comparability check
# --------------------------------------------------------------------------


def assert_arms_comparable(primary: Experiment, oracle: Experiment) -> tuple[str, ...]:
    """Refuse to compare two arms that are not two views of one corpus.

    Returns the features the oracle arm has and the primary arm does not — the
    leak, named. Everything else here is a refusal, and each one guards a
    failure that would otherwise produce a plausible number:

    * Different row counts or a different index means the two builds saw
      different races, and the "paired" delta below would pair a driver in one
      arm with a different driver in the other.
    * A primary feature absent from the oracle frame means the primary model
      cannot be scored on that frame at all — better to say so here than to
      surface it from inside the booster.
    * A shared feature whose *values* differ between the builds means adding the
      post-race features perturbed the primary ones. The measured gap would then
      be that perturbation plus the leak, reported as the leak alone.
    * An empty extra set means the oracle build produced the primary set. There
      is nothing to measure, and a delta of zero would be reported as the
      reassuring finding that leakage is worth nothing.
    """
    if len(primary.frame) != len(oracle.frame) or not primary.frame.index.equals(
        oracle.frame.index
    ):
        raise OracleArmError(
            f"the primary arm has {len(primary.frame)} row(s) and the oracle arm "
            f"{len(oracle.frame)}; they were built from different corpora. A "
            f"delta between them would not be a delta."
        )

    for column in (*KEY_COLUMNS, ACTUAL_COLUMN, "grid_position"):
        if not primary.frame[column].equals(oracle.frame[column]):
            raise OracleArmError(
                f"the two arms disagree on {column!r}. They must be two feature "
                f"sets over one corpus, not two corpora."
            )

    extra = tuple(n for n in oracle.feature_names if n not in set(primary.feature_names))
    missing = sorted(set(primary.feature_names) - set(oracle.feature_names))
    if missing:
        raise OracleArmError(
            f"the oracle arm is missing primary feature(s) {missing}. FR-207's "
            f"oracle set is the primary set plus post-race features, so the "
            f"primary model must be scorable on the oracle frame."
        )
    if not extra:
        raise OracleArmError(
            "the oracle feature set adds nothing to the primary set. There is no "
            "gap to measure, and reporting zero would read as evidence that "
            "post-race features are worthless rather than as a build defect."
        )

    for name in primary.feature_names:
        if not primary.frame[name].equals(oracle.frame[name]):
            raise OracleArmError(
                f"feature {name!r} holds different values in the two arms. The "
                f"measured gap would include that difference and be reported as "
                f"the value of the post-race features alone."
            )
    return extra


def describe_leak(
    names: Iterable[str], *, registry: Registry = REGISTRY
) -> list[dict[str, str]]:
    """Name the leak, one entry per feature, with its declared availability.

    FR-207's set is defined by declarations (FR-202), so the description is read
    back off the registry rather than written here: a feature that changed group
    or availability would otherwise keep a stale description in every results
    file that had already been written.
    """
    out: list[dict[str, str]] = []
    for name in names:
        feature = registry.get(name)
        out.append(
            {
                "name": feature.name,
                "availability": feature.availability.value,
                "description": feature.description,
            }
        )
    return out


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


def _arm_record(
    built: Experiment,
    model: TrainedRanker,
    line: dict[str, Any],
    *,
    feature_set: str,
) -> dict[str, Any]:
    """One arm's side of the record: what it was built from and what it scored."""
    return {
        "predictor": model.name,
        "feature_set": feature_set,
        "n_features": len(built.feature_names),
        "features": list(built.feature_names),
        "corpus_rows": len(built),
        "training_rows": model.training_rows,
        "training_seasons": list(model.training_seasons),
        "trees_kept": model.trees,
        # FR-206/FR-214: a run without weather is a different experiment from
        # one with it, and after the fact only this field distinguishes them.
        "absent_sources": list(built.absent_sources),
        "metrics": line["metrics"],
        "races_scored": line["races_scored"],
    }


def _statement(delta: Interval, primary: Interval, oracle: Interval, races: int) -> str:
    """The plain-language line, computed from the numbers rather than written.

    Written by code for the same reason FR-507's verdict is: a human writing
    "the oracle arm scored higher" is a human who might still write it on the
    run where it did not. The wording branches on the sign, so a null or
    negative result reads as the finding it is instead of being smoothed over.
    """
    metric = PRIMARY_METRIC.name
    better = delta.point > 0 if PRIMARY_METRIC.higher_is_better else delta.point < 0
    head = (
        f"Primary arm {primary.point:.4f} {metric}, oracle arm {oracle.point:.4f}, "
        f"on the identical {races}-race holdout. Paired difference "
        f"{delta.point:+.4f} (95% CI {delta.low:+.4f} to {delta.high:+.4f}"
        f"{', excludes zero' if delta.excludes_zero else ', includes zero'})."
    )
    if better and delta.excludes_zero:
        return (
            f"{head} Knowing the answer inflates the primary metric by "
            f"{abs(delta.point):.4f} {metric}. That is the scale of what a "
            f"post-race feature buys — the size of v0's inflation, not a "
            f"capability anything can be built on."
        )
    if better:
        return (
            f"{head} The oracle arm scores higher, but the interval includes "
            f"zero: on this holdout the inflation from post-race features is not "
            f"separable from noise. It is an upper bound of at most "
            f"{max(abs(delta.low), abs(delta.high)):.4f} {metric}, not a measured "
            f"gap."
        )
    return (
        f"{head} The oracle arm did NOT score higher than the primary arm. "
        f"Post-race features bought nothing measurable here; the gap is reported "
        f"as it came out rather than as the inflation this run expected to find."
    )


def compare(
    *,
    seasons: Iterable[int] | None = None,
    root: Path = RAW_DIR,
    split: ChronologicalSplit | None = None,
    params: Hyperparameters | None = None,
    primary: Experiment | None = None,
    oracle: Experiment | None = None,
    replicates: int = DEFAULT_REPLICATES,
    random_control: bool = False,
    primary_name: str = PRIMARY_ARM,
    oracle_name: str = ORACLE_ARM,
    label: str = "oracle_vs_primary",
    path: Path | None = None,
    write: bool = True,
) -> OracleComparison:
    """Train both arms, score them together, and report the paired gap (FR-207).

    `random_control` is off by default, unlike the headline run. FR-303's floor
    says whether the *metric* is sound, which the headline run re-establishes on
    every pass; this run asks a question about the distance between two arms,
    and a thousand extra scored draws would not move it. Turning it on is
    available for anyone who wants the floor printed beside the gap.
    """
    if primary_name == oracle_name:
        raise OracleArmError(
            f"both arms are named {primary_name!r}. The harness keys results by "
            f"name, so one arm would overwrite the other and the record would "
            f"report a single score as though it were the comparison."
        )

    split = split or default_split()
    # Two builds, not one build reused: each arm's frame must carry its own
    # feature columns, and a shared build would hand the primary model whichever
    # set the last caller asked for.
    primary_built = (
        primary if primary is not None
        else prepare(seasons, root=root, builder=build_primary)
    )
    oracle_built = (
        oracle if oracle is not None
        else prepare(seasons, root=root, builder=build_oracle)
    )
    leak = assert_arms_comparable(primary_built, oracle_built)

    primary_model = train_model(
        primary_built, split=split, params=params, name=primary_name
    )
    oracle_model = train_model(
        oracle_built, split=split, params=params, name=oracle_name
    )

    # One frame, one call. The oracle frame carries every primary column — the
    # check above proved it, value for value — so both models and both baselines
    # face literally the same rows (FR-304).
    scoring_frame = oracle_built.frame
    record = evaluate(
        scoring_frame,
        (*baselines(), primary_model, oracle_model),
        split=split,
        replicates=replicates,
        random_control=(
            RandomOrderControl().predictors() if random_control else None
        ),
    )

    # The harness keeps its per-race frames internally, so the paired delta is
    # recomputed through the same scoring path on the same holdout rather than
    # reconstructed from the aggregates — which is impossible anyway: a mean of
    # differences is recoverable from two means, an interval on it is not.
    _, holdout = split.split(scoring_frame)
    per_race_primary = score_predictor(primary_model, holdout)
    per_race_oracle = score_predictor(oracle_model, holdout)
    paired = bootstrap_delta(
        per_race_oracle, per_race_primary, metrics=METRICS, replicates=replicates
    )

    metric = PRIMARY_METRIC.name
    primary_line = record["predictors"][primary_name]
    oracle_line = record["predictors"][oracle_name]
    primary_interval = _interval_of(primary_line, metric)
    oracle_interval = _interval_of(oracle_line, metric)

    record["label"] = label
    record["seasons"] = sorted(int(s) for s in scoring_frame["season"].unique())
    record["corpus_rows"] = len(oracle_built)
    record["arms"] = {
        "primary": _arm_record(
            primary_built, primary_model, primary_line, feature_set="primary"
        ),
        "oracle": _arm_record(
            oracle_built, oracle_model, oracle_line, feature_set="oracle"
        ),
    }
    # What the leak *consists of*. A reader who cannot see the feature names
    # cannot tell a two-column oracle set from one that reads the finishing
    # order directly, and the two mean very different things about the gap.
    record["leak"] = {
        "features_in_oracle_not_in_primary": describe_leak(leak),
        "n_features_added": len(leak),
    }
    record["oracle_minus_primary"] = {k: v.as_dict() for k, v in paired.items()}
    record["inflation"] = {
        "metric": metric,
        "primary": primary_interval.point,
        "oracle": oracle_interval.point,
        "delta": paired[metric].point,
        "ci_low": paired[metric].low,
        "ci_high": paired[metric].high,
        "excludes_zero": paired[metric].excludes_zero,
        "paired": True,
        "statement": _statement(
            paired[metric],
            primary_interval,
            oracle_interval,
            int(record["holdout_races"]),
        ),
    }
    record["admissibility"] = {
        "oracle_arm": INADMISSIBLE,
        "primary_arm": (
            "Admissible. Pre-race features only, scored on the same holdout "
            "(FR-202, FR-304)."
        ),
        # The harness computes FR-507's verdict over every predictor it was
        # handed, and it was handed the oracle arm. Left unannotated, the
        # `verdict` block of this particular file reports that something beat
        # the grid — which is true and worthless, and is exactly the sentence
        # a reader skimming for a headline would lift.
        "verdict_block": (
            "The `verdict` block is the harness's standard FR-507 output and "
            "counts the oracle arm among the predictors that beat the grid. "
            "The oracle arm beat the grid by reading the result. Only the "
            "primary arm's line there is a claim about forecasting; for the "
            "size of the gap between them, read `inflation`."
        ),
    }

    written: Path | None = None
    if write:
        # Named after the holdout it reports, matching every other file in
        # `results/`: a fixed name is one rerun away from describing a
        # different holdout than the report that cites it.
        holdout_seasons = split.holdout
        target = path or RESULTS_DIR / (
            f"{label}_{holdout_seasons[0]}_{holdout_seasons[-1]}.json"
        )
        written = write_results(record, target)
        logger.info("wrote %s", written)

    return OracleComparison(
        record=record, primary=primary_model, oracle=oracle_model, path=written
    )


def _interval_of(line: dict[str, Any], metric: str) -> Interval:
    """Rebuild an `Interval` from the dict the harness already serialised."""
    entry = line["metrics"][metric]
    return Interval(
        point=float(entry["point"]),
        low=float(entry["ci_low"]),
        high=float(entry["ci_high"]),
        confidence=float(entry["confidence"]),
    )


def render_comparison(record: dict[str, Any]) -> str:
    """The human-readable side of FR-506, for the file's machine-readable one."""
    inflation = record["inflation"]
    leak = record["leak"]["features_in_oracle_not_in_primary"]
    lines = [
        f"Oracle-variant comparison (FR-207), holdout "
        f"{record['split']['holdout_seasons'][0]}-"
        f"{record['split']['holdout_seasons'][-1]}.",
        "",
        inflation["statement"],
        "",
        f"The oracle set adds {len(leak)} post-race feature(s):",
    ]
    lines += [f"  {f['name']} — {f['description']}" for f in leak]
    lines += ["", record["admissibility"]["oracle_arm"]]
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - thin entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    comparison = compare()
    print(render_comparison(comparison.record))
    return 0


__all__ = [
    "INADMISSIBLE",
    "ORACLE_ARM",
    "PRIMARY_ARM",
    "OracleArmError",
    "OracleComparison",
    "assert_arms_comparable",
    "compare",
    "describe_leak",
    "main",
    "render_comparison",
]


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
