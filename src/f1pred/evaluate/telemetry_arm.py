"""What practice telemetry is worth, and on how little data (FR-215, AC-19, T-216).

FR-213 gives the telemetry features a named group; FR-214 makes them explicitly
null before 2018 and forbids imputing that away. Between them they leave a
feature set that describes 3,458 of the corpus's 6,911 rows and says nothing at
all about the other 3,453 — which is why FR-215 keeps the group out of the
primary set and asks for its value to be measured *separately*, on the covered
seasons only, with the reduced season count stated next to any benefit.

The reduced count is the whole reason this module exists rather than a flag on
the headline run. Two numbers produced here look exactly like two numbers from
`evaluate.runner`, and they are not comparable to them: both arms train on
2018 onward, so the primary arm reported here is *not* the headline model. It
is a control built the same way as its own comparison, on the same eight
seasons, for the sole purpose of isolating what the telemetry group adds. A
reader who takes the telemetry arm's Spearman as the project's answer has been
misled, so the season list, the training and holdout seasons, and the count of
seasons the whole corpus holds are recorded in the record itself rather than
left to a caller to remember.

Three refusals, each of them a way this measurement could quietly lie:

* **The floor is not negotiated.** Eight covered seasons is exactly FR-501's
  minimum. The window is constructed before the corpus is built, so a covered
  corpus that ever falls below it fails at `CorpusWindow` — loudly, and before
  anything has been trained — rather than reporting a thin arm as a result.
* **The delta is paired inside the bootstrap replicates** (plan D-5). Both arms
  are scored on the identical races, so the difference between them is far
  better determined than either level; subtracting two independently
  bootstrapped intervals would throw the pairing away and overstate the
  uncertainty in the one direction that makes a null look like a finding.
* **An unattached source is reported as such.** With no practice checkpoints on
  disk the telemetry columns are null throughout, the second arm is the first
  arm carrying four empty columns, and the delta is a bootstrap of zeros. That
  is not "telemetry does not help"; it is "nothing was measured", and the
  record says which of the two happened.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from f1pred.config import RAW_DIR, RESULTS_DIR, SEED
from f1pred.provenance import Scope, stamp
from f1pred.corpus import CorpusWindow
from f1pred.data import telemetry
from f1pred.dataset import Experiment, available_seasons, load_corpus, prepare
from f1pred.evaluate.bootstrap import DEFAULT_REPLICATES, bootstrap_delta
from f1pred.evaluate.harness import (
    REFERENCE_BASELINE,
    evaluate,
    score_predictor,
    write_results,
)
from f1pred.evaluate.metrics import METRICS, PRIMARY_METRIC
from f1pred.evaluate.runner import baselines, train_model
from f1pred.evaluate.split import MIN_HOLDOUT_SEASONS, ChronologicalSplit
from f1pred.features.build import ENRICHMENT_COLUMNS, FeatureMatrix, build_group
from f1pred.features.registry import REGISTRY, FeatureGroup, Registry
from f1pred.models.ranker import Hyperparameters, TrainedRanker, code_commit

logger = logging.getLogger(__name__)

#: The two arms FR-215 names. The second is spelled the way `build_group`
#: spells it, so the arm in this file and the feature set in the matrix cannot
#: drift into two different names for one thing.
PRIMARY_ARM: Final[str] = "primary"
TELEMETRY_ARM: Final[str] = f"primary+{FeatureGroup.TELEMETRY.value}"

#: Written into the results filename, which is then completed by the holdout
#: the run actually resolved to — as every other file in `results/` is.
LABEL: Final[str] = "telemetry_arm"

#: The harness keys predictors by name and a repeated name silently overwrites
#: the earlier entry, which here would mean one arm reporting the other's
#: score. The arm name is part of the model's name for exactly that reason.
MODEL_NAME_PREFIX: Final[str] = "xgboost_ranker"


class TelemetryArmError(ValueError):
    """Raised when the separate arm cannot be measured as FR-215 asks."""


# --------------------------------------------------------------------------
# Resolving the covered corpus
# --------------------------------------------------------------------------


def covered_seasons(
    root: Path = RAW_DIR, *, start: int = telemetry.TELEMETRY_START_SEASON
) -> tuple[int, ...]:
    """Collected seasons the telemetry source can speak about, ascending.

    Derived from two declarations rather than written down: what is on disk
    (`available_seasons`) intersected with where the source begins
    (`telemetry.TELEMETRY_START_SEASON`, the same constant the collector
    refuses to fetch below). A literal `2018` here would be a third place to
    edit when the source's coverage changes, and the one nobody would find.
    """
    return tuple(s for s in available_seasons(root) if s >= start)


def covered_window(seasons: Sequence[int]) -> CorpusWindow:
    """The window the covered seasons form — and the constitutional floor.

    Built before any corpus is loaded, because `CorpusWindow` already refuses a
    width below `MIN_WINDOW_WIDTH` and that refusal is the FR-501 floor. Failing
    here costs a millisecond; failing after two feature builds and two fits
    costs minutes and tempts somebody to lower the holdout to get past it.
    """
    if not seasons:
        raise TelemetryArmError(
            "no collected season falls inside the telemetry source's coverage. "
            "There is no arm to measure — collect the covered seasons first."
        )
    ordered = tuple(sorted({int(s) for s in seasons}))
    return CorpusWindow(width=len(ordered), earliest_available=ordered[0])


def season_end_dates(frame: pd.DataFrame) -> dict[int, date]:
    """Each season's last race date, so FR-509 completion is read from the data.

    Without this the window falls back to the calendar year, which is correct
    for the real corpus today and wrong for any fixture, and wrong for the real
    corpus the moment the clock passes a New Year the data has not reached.
    """
    dates = pd.to_datetime(frame["race_date"])
    return {
        int(season): pd.Timestamp(value).date()
        for season, value in dates.groupby(frame["season"]).max().items()
    }


def covered_split(
    seasons: Sequence[int],
    *,
    holdout_seasons: int = MIN_HOLDOUT_SEASONS,
    end_dates: dict[int, date] | None = None,
) -> ChronologicalSplit:
    """One chronological split, shared by both arms.

    Shared rather than built twice: two splits constructed from the same
    arguments would be equal today and are one edit away from not being, and
    the entire claim of this experiment is that the two arms faced the same
    races. `ChronologicalSplit` re-checks the FR-501 minima itself, so a
    holdout that would leave fewer than five training seasons is refused here
    too rather than only at the window.
    """
    return ChronologicalSplit(
        window=covered_window(seasons),
        holdout_seasons=holdout_seasons,
        season_end_dates=end_dates,
    )


# --------------------------------------------------------------------------
# The second arm's feature set
# --------------------------------------------------------------------------


def build_with_telemetry(df: pd.DataFrame) -> FeatureMatrix:
    """FR-215's second arm: the primary set plus the telemetry group."""
    return build_group(df, FeatureGroup.TELEMETRY)


def telemetry_source_columns(registry: Registry = REGISTRY) -> tuple[str, ...]:
    """Enrichment columns the telemetry features read, derived from the group.

    Derived rather than listed so that a fifth telemetry feature reading a
    fifth practice column cannot leave this check watching only the first four
    — which would report an arm as measured when part of its source was absent.
    """
    wanted = {
        dep
        for f in registry.group_features(FeatureGroup.TELEMETRY)
        for dep in f.depends_on
    }
    return tuple(c for c in ENRICHMENT_COLUMNS if c in wanted)


# --------------------------------------------------------------------------
# Coverage (FR-214, reported alongside the benefit per AC-19)
# --------------------------------------------------------------------------


def coverage_by_season(
    seasons: Iterable[int] | None = None, *, root: Path = RAW_DIR
) -> dict[str, dict[str, object]]:
    """Per-season coverage of *corpus rows*, recomputed from what is on disk.

    Computed over the raw corpus rather than an experiment frame, because
    FR-002 drops non-starters and the coverage T-214 published counts every
    collected row. Two files reporting "coverage" over two different
    denominators is precisely the kind of quiet disagreement AC-19 is meant to
    prevent, so the field names here are T-214's field names as well.
    """
    directory = Path(root)
    wanted = tuple(sorted({int(s) for s in seasons})) if seasons is not None else None
    corpus = load_corpus(wanted, root=directory)
    present = wanted if wanted is not None else available_seasons(directory)
    frames = [
        frame
        for frame in (
            telemetry.read_practice_checkpoint(directory, s) for s in present
        )
        if frame is not None and not frame.empty
    ]
    practice = (
        pd.concat(frames, ignore_index=True)
        if frames
        else telemetry.frame_from_rows([])
    )
    table = telemetry.coverage(corpus, practice)
    return {
        str(int(row.season)): {
            "corpus_rows": int(row.corpus_rows),
            "covered": int(row.covered_rows),
            "coverage": float(row.coverage),
            "source_covers_season": bool(row.source_covers_season),
        }
        for row in table.itertuples()
    }


def cited_coverage(
    seasons: Sequence[int], *, results_dir: Path = RESULTS_DIR
) -> tuple[Path, dict[str, Any] | None]:
    """T-214's published coverage table, if it is where its name says it is.

    Cited *and* recomputed. Citing alone would let this file rest on a number
    nobody re-derived; recomputing alone would let the two files disagree in
    silence, which is worse than either being wrong on its own.
    """
    path = Path(results_dir) / f"telemetry_coverage_{seasons[0]}_{seasons[-1]}.json"
    if not path.exists():
        return path, None
    published = json.loads(path.read_text())
    table = published.get("coverage_per_season")
    return path, table if isinstance(table, dict) else None


def coverage_disagreements(
    recomputed: dict[str, dict[str, object]],
    cited: dict[str, Any] | None,
    *,
    tolerance: float = 1e-4,
) -> dict[str, dict[str, object]]:
    """Seasons where this run and the cited file do not say the same thing.

    Returned rather than raised. A disagreement can mean the corpus advanced
    since T-214 ran, which is not an error; what would be an error is nobody
    being told. Only seasons both files describe are compared, so a corpus that
    has grown a season is not reported as sixteen disagreements.
    """
    if not cited:
        return {}
    out: dict[str, dict[str, object]] = {}
    for season, mine in recomputed.items():
        theirs = cited.get(season)
        if not isinstance(theirs, dict):
            continue
        if abs(float(mine["coverage"]) - float(theirs.get("coverage", -1.0))) > tolerance:
            out[season] = {"this_run": mine["coverage"], "cited": theirs.get("coverage")}
    return out


# --------------------------------------------------------------------------
# One arm
# --------------------------------------------------------------------------


def model_name(arm: str) -> str:
    """The predictor name for an arm. Distinct per arm, by construction."""
    return f"{MODEL_NAME_PREFIX}_{arm.replace('+', '_')}"


@dataclass(frozen=True, slots=True)
class Arm:
    """One side of the comparison, kept whole so the pairing stays checkable.

    `per_race` sits here rather than only inside `evaluation` because the
    arm-versus-arm delta has to be paired within bootstrap replicates (plan
    D-5), and a record of two finished intervals cannot be paired after the
    fact. Keeping the race-level scores is what makes the honest delta
    available at all.
    """

    name: str
    experiment: Experiment
    model: TrainedRanker
    per_race: pd.DataFrame
    evaluation: dict[str, Any]


def run_arm(
    name: str,
    built: Experiment,
    *,
    split: ChronologicalSplit,
    params: Hyperparameters | None = None,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = SEED,
) -> Arm:
    """Train and score one arm through the path the headline run uses.

    The baselines travel with each arm rather than being scored once: FR-304
    requires the identical code path, `evaluate` refuses a run with no
    `grid_order` among the predictors, and having the grid line inside both
    arms' records is what lets a reader check that the two arms really did face
    the same holdout — the baseline's score is a property of the races alone,
    so it must come out identical on both sides.
    """
    model = train_model(built, split=split, params=params, name=model_name(name))
    record = evaluate(
        built.frame,
        (*baselines(), model),
        split=split,
        random_control=None,
        replicates=replicates,
        seed=seed,
    )
    scored = model.with_evaluation_metrics(record["predictors"][model.name]["metrics"])
    _, holdout = split.split(built.frame)
    return Arm(
        name=name,
        experiment=built,
        model=scored,
        per_race=score_predictor(scored, holdout),
        evaluation=record,
    )


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def run(
    *,
    root: Path = RAW_DIR,
    seasons: Iterable[int] | None = None,
    holdout_seasons: int = MIN_HOLDOUT_SEASONS,
    params: Hyperparameters | None = None,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = SEED,
    results_dir: Path = RESULTS_DIR,
    path: Path | None = None,
) -> tuple[dict[str, Any], tuple[Arm, Arm]]:
    """Measure the telemetry group as FR-215's separate arm. Returns the record.

    The order is fixed here for the same reason `runner.run` fixes its own: one
    season list, resolved once; one split, applied by both trainers and by the
    harness; two feature builds that differ in exactly the telemetry group and
    in nothing else. A caller who could vary any of those could produce a
    "benefit" that was really a difference in corpora.
    """
    directory = Path(root)
    resolved = (
        tuple(sorted({int(s) for s in seasons}))
        if seasons is not None
        else covered_seasons(directory)
    )
    # Before the corpus is touched: a covered corpus below the FR-501 floor
    # must fail, not shrink the holdout to fit (FR-501a).
    window = covered_window(resolved)

    primary_built = prepare(resolved, root=directory)
    split = ChronologicalSplit(
        window=window,
        holdout_seasons=holdout_seasons,
        season_end_dates=season_end_dates(primary_built.frame),
    )
    telemetry_built = prepare(resolved, root=directory, builder=build_with_telemetry)

    arms = (
        run_arm(
            PRIMARY_ARM,
            primary_built,
            split=split,
            params=params,
            replicates=replicates,
            seed=seed,
        ),
        run_arm(
            TELEMETRY_ARM,
            telemetry_built,
            split=split,
            params=params,
            replicates=replicates,
            seed=seed,
        ),
    )
    primary_arm, telemetry_arm = arms

    # Paired inside the replicates, telemetry minus primary. `bootstrap_delta`
    # raises rather than dropping rows if the two arms covered different races,
    # which is the check that both arms shared one holdout.
    delta = bootstrap_delta(
        telemetry_arm.per_race,
        primary_arm.per_race,
        metrics=METRICS,
        replicates=replicates,
        seed=seed,
    )

    record = _record(
        arms=arms,
        delta=delta,
        split=split,
        seasons=resolved,
        root=directory,
        results_dir=Path(results_dir),
    )
    record["finding"] = render_finding(record)

    target = path or Path(results_dir) / (
        f"{LABEL}_{split.holdout[0]}_{split.holdout[-1]}.json"
    )
    # FR-214: the telemetry source starts in 2018 while the corpus starts in
    # 2010, so this arm reads half the window by construction. Declared, so a
    # reader cannot mistake its seasons for the evaluation's.
    stamp(
        record,
        scope=Scope.SUBSET,
        reason=(
            "practice telemetry exists only from 2018, so this arm covers 8 of "
            "the window's 16 seasons (FR-214)"
        ),
    )
    logger.info("wrote %s", write_results(record, target))
    return record, arms


def _entering_features(arms: tuple[Arm, Arm], split: ChronologicalSplit) -> list[dict[str, Any]]:
    """Which features the second arm added, and whether they hold anything.

    The names are taken from the two matrices rather than from the registry, so
    what is reported is what actually entered the fit. The non-null count is
    the difference between "telemetry did not help" and "telemetry was not
    there", and only one of those is a finding.
    """
    primary_arm, telemetry_arm = arms
    already = set(primary_arm.experiment.feature_names)
    _, holdout = split.split(telemetry_arm.experiment.frame)
    return [
        {
            "name": name,
            "holdout_rows": int(len(holdout)),
            "holdout_rows_with_a_value": int(holdout[name].notna().sum()),
        }
        for name in telemetry_arm.experiment.feature_names
        if name not in already
    ]


def _record(
    *,
    arms: tuple[Arm, Arm],
    delta: dict[str, Any],
    split: ChronologicalSplit,
    seasons: tuple[int, ...],
    root: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Assemble the machine-readable record FR-506 and FR-215 require."""
    primary_arm, telemetry_arm = arms
    corpus = available_seasons(root)
    entering = _entering_features(arms, split)
    attached = any(f["holdout_rows_with_a_value"] > 0 for f in entering)
    absent_sources = tuple(
        c
        for c in telemetry_source_columns()
        if c in telemetry_arm.experiment.absent_sources
    )

    coverage = coverage_by_season(root=root)
    cited_path, cited = cited_coverage(seasons, results_dir=results_dir)
    disagreements = coverage_disagreements(coverage, cited)
    if disagreements:
        logger.warning(
            "recomputed coverage disagrees with %s for season(s) %s",
            cited_path,
            ", ".join(sorted(disagreements)),
        )

    return {
        "task": "T-216",
        "requirements": ["FR-213", "FR-214", "FR-215", "FR-501", "AC-19"],
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "code_commit": code_commit(),
        "question": (
            "What do the telemetry features add, measured on the seasons the "
            "telemetry source covers?"
        ),
        "primary_metric": PRIMARY_METRIC.name,
        # FR-215's mandatory half: the reduced count, stated rather than implied.
        "seasons": {
            "source_start_season": telemetry.TELEMETRY_START_SEASON,
            "telemetry_covered": list(seasons),
            "n_telemetry_covered": len(seasons),
            "collected_corpus_seasons": list(corpus),
            "n_collected_corpus_seasons": len(corpus),
            "seasons_this_arm_cannot_speak_about": [
                s for s in corpus if s not in set(seasons)
            ],
            "training_seasons": list(split.training_seasons),
            "holdout_seasons": list(split.holdout),
            "reduced_season_count": (
                f"Every number in this file rests on {len(seasons)} of the "
                f"corpus's {len(corpus)} collected seasons "
                f"({seasons[0]}-{seasons[-1]}), because FR-214 leaves the "
                f"telemetry features undefined before "
                f"{telemetry.TELEMETRY_START_SEASON}."
            ),
            "not_the_headline_run": (
                "Both arms train on the covered seasons only, so the 'primary' "
                "arm here is a like-for-like control and not the project's "
                "model. Neither arm's score is comparable with "
                "results/model_vs_baselines_2023_2025.json, which rests on the "
                "full corpus; only the delta between the two arms is."
            ),
        },
        "split": split.describe(),
        "arms": {
            arm.name: {
                "feature_set": arm.experiment.matrix.set_name,
                "n_features": len(arm.experiment.feature_names),
                "predictor": arm.model.name,
                "corpus_rows": len(arm.experiment),
                # FR-206/FR-214: which enrichment this arm actually had.
                "absent_sources": list(arm.experiment.absent_sources),
                "metrics": arm.evaluation["predictors"][arm.model.name]["metrics"],
                f"delta_vs_{REFERENCE_BASELINE}": arm.evaluation["predictors"][
                    arm.model.name
                ][f"delta_vs_{REFERENCE_BASELINE}"],
                "model": arm.model.metadata(),
                "evaluation": arm.evaluation,
            }
            for arm in arms
        },
        "telemetry_minus_primary": {
            name: interval.as_dict() for name, interval in delta.items()
        },
        "pairing": (
            "Paired within bootstrap replicates on the identical holdout races "
            "(plan D-5). Subtracting the two arms' independently bootstrapped "
            "intervals would overstate the uncertainty."
        ),
        "telemetry_features": entering,
        "telemetry_attached": attached,
        "absent_telemetry_sources": list(absent_sources),
        "coverage": {
            "per_season": coverage,
            "cited_file": str(cited_path),
            "cited_file_found": cited is not None,
            "agrees_with_cited_file": None if cited is None else not disagreements,
            "disagreements": disagreements,
        },
    }


def render_finding(record: dict[str, Any]) -> str:
    """FR-215's sentence: the benefit, and the season count beside it.

    Computed rather than written, for FR-507's reason one level down: a human
    writing "on 8 of 16 seasons" is a human who might write it once and then
    forget it on the rerun that mattered.
    """
    seasons = record["seasons"]
    metric = record["primary_metric"]
    delta = record["telemetry_minus_primary"][metric]
    arms = record["arms"]
    scale = (
        f"{seasons['n_telemetry_covered']} of the corpus's "
        f"{seasons['n_collected_corpus_seasons']} seasons "
        f"({seasons['telemetry_covered'][0]}-{seasons['telemetry_covered'][-1]}), "
        f"holdout {seasons['holdout_seasons'][0]}-{seasons['holdout_seasons'][-1]}"
    )

    if not record["telemetry_attached"]:
        absent = record["absent_telemetry_sources"]
        why = f" (absent source columns: {', '.join(absent)})" if absent else ""
        return (
            f"NOT MEASURED. No telemetry value reached the holdout{why}, so the "
            f"second arm is the primary arm carrying empty columns and the "
            f"delta below is an artefact of that rather than a measurement of "
            f"telemetry. Attach the practice checkpoints and rerun. "
            f"Scale: {scale}."
        )

    head = (
        f"{metric}: {PRIMARY_ARM} {arms[PRIMARY_ARM]['metrics'][metric]['point']:.4f}, "
        f"{TELEMETRY_ARM} {arms[TELEMETRY_ARM]['metrics'][metric]['point']:.4f}. "
        f"Telemetry minus primary {delta['point']:+.4f} "
        f"(95% CI {delta['ci_low']:+.4f} to {delta['ci_high']:+.4f}, paired)."
    )
    if delta["excludes_zero"] and delta["point"] > 0:
        body = f" TELEMETRY HELPS on {scale}."
    elif delta["excludes_zero"]:
        body = f" TELEMETRY HURTS on {scale}."
    else:
        body = (
            f" NO MEASURED BENEFIT: the interval includes zero on {scale}. "
            f"A measured null is the answer FR-215 asks for, not a failed run."
        )
    return head + body


def main() -> int:  # pragma: no cover - thin entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    record, _arms = run()
    print(render_finding(record))
    return 0


__all__ = [
    "Arm",
    "LABEL",
    "MODEL_NAME_PREFIX",
    "PRIMARY_ARM",
    "TELEMETRY_ARM",
    "TelemetryArmError",
    "build_with_telemetry",
    "cited_coverage",
    "coverage_by_season",
    "coverage_disagreements",
    "covered_seasons",
    "covered_split",
    "covered_window",
    "main",
    "model_name",
    "render_finding",
    "run",
    "run_arm",
    "season_end_dates",
    "telemetry_source_columns",
]


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
