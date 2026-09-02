"""The regulation-transfer experiment (FR-511, FR-511a, FR-512, AC-12).

What a regime change costs, measured rather than argued about. For each major
reset declared in the FR-210 regime table, a model is trained on every archive
season strictly before it and scored on the reset season itself — a regime it
has never seen — and the identical protocol is run one season earlier, where
the rules are ones the model has seen. The difference between the two is the
**measured regime-transfer penalty**.

Three things about this module are deliberate, and each of them is a refusal.

**It reads the archive, not the rolling window.** FR-508's 16-season window
advances at both ends every year, so binding this experiment to it would push
2017 out of reach in 2028 and silently drop the observation count from two to
one, with nothing in the output to explain why. `corpus.archive_seasons` exists
for exactly this, and the seasons are already on disk, so the diagnostic costs
nothing and its observation count grows instead of decaying.

**It cannot borrow `ChronologicalSplit`, and does not try.** See `TransferSplit`
below: every arm here holds out a *single* season, and one arm trains on four,
both of which that type refuses by design. Weakening those checks to fit this
experiment would weaken them for the headline evaluation too, which is the one
place they are load-bearing.

**Its penalty interval is unpaired, and says so.** The two arms of a transition
are scored on *different races* — season R against season R-1 — so
`bootstrap_delta`'s pairing is not merely inconvenient here, it is undefined.
That function refuses mismatched race sets rather than aligning them, and the
refusal is right. The interval reported here resamples each arm independently
within a replicate, which is wider than a paired interval would be and is
labelled `pairing: unavailable` wherever it appears. Reporting a paired number
over two different seasons would be the more precise-looking lie.

The grid baseline is scored on every evaluated season for the same reason: a
penalty is only readable against how predictable the season was at all. A
season where every predictor did badly is not a regime the model failed to
transfer into.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from f1pred.config import (
    RAW_DIR,
    REGIMES,
    RESULTS_DIR,
    SEED,
    Regime,
    RegimeSeverity,
)
from f1pred.corpus import archive_seasons
from f1pred.dataset import Experiment, available_seasons, prepare
from f1pred.evaluate.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_REPLICATES,
    Interval,
    bootstrap_delta,
    bootstrap_metrics,
)
from f1pred.evaluate.harness import score_predictor, write_results
from f1pred.evaluate.metrics import METRICS, PRIMARY_METRIC, RACE_KEYS, Metric
from f1pred.evaluate.split import MIN_TRAINING_SEASONS
from f1pred.models.baselines import GridOrderBaseline
from f1pred.models.ranker import Hyperparameters, TrainedRanker, train

logger = logging.getLogger(__name__)

#: FR-511a's label for a transition that falls short of the FR-501 training
#: minimum. It is a string in the results file rather than an absent field
#: because a reader scanning for compliance must find something, not nothing.
DIAGNOSTIC_LABEL: Final[str] = "DIAGNOSTIC"

#: FR-512 requires the experiment at zero decay and at a non-zero one. Zero is
#: uniform weighting exactly (T-408), so the pair is a controlled comparison
#: rather than two unrelated runs.
#:
#: 0.25 gives `exp(-0.25 * age)` a half-life of about 2.8 seasons: the season
#: before a reset keeps three-quarters of its weight and a decade-old season
#: keeps eight per cent. Chosen to be large enough that the weighting could
#: plausibly help at a regime boundary — a decay small enough to change nothing
#: would answer FR-512 with a tautology — and it is not tuned against the
#: penalty it is being used to measure.
DEFAULT_NONZERO_DECAY: Final[float] = 0.25

DECAY_SETTINGS: Final[tuple[float, ...]] = (0.0, DEFAULT_NONZERO_DECAY)

#: Seasons held back from the end of training to size FR-402's interval and
#: pick the tree count. Two, not the model's default of three, and the same
#: for every arm: the shortest arm here trains on three seasons, and a slice
#: that fitted the long arms but not the short one would mean the six fits
#: were not the identical protocol FR-511 compares.
CALIBRATION_SEASONS: Final[int] = 2

#: Named once so the two arms cannot drift apart in the record.
TRANSFER_ARM: Final[str] = "transfer"
COMPARATOR_ARM: Final[str] = "comparator"

MODEL_NAME: Final[str] = "xgboost_ranker"
BASELINE_NAME: Final[str] = "grid_order"

#: Why the penalty interval is unpaired. Carried in the record itself, so a
#: reader who never opens this module still learns that the two arms were
#: scored on different races.
PAIRING_UNAVAILABLE: Final[str] = (
    "the two arms are scored on different races (season R against season R-1), "
    "so a paired delta is undefined; bootstrap_delta refuses mismatched race "
    "sets rather than aligning them. Each arm is resampled independently within "
    "a replicate, which is wider than a paired interval and is the honest width."
)


class TransferError(ValueError):
    """Raised when a transfer experiment cannot be run as specified."""


# --------------------------------------------------------------------------
# The split this experiment needs, and cannot get from `split.py`
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransferSplit:
    """Train on the seasons before one target season, evaluate on that season.

    **Why this is not a `ChronologicalSplit`.** That type enforces the
    Constitution IV.2 minima structurally — at least eight seasons, at least
    five training, at least three held out — and refuses anything else at
    construction. Every arm of FR-511 evaluates a *single* season, and the 2014
    transition trains on four. Both are unexpressible there, and that is the
    correct behaviour for the headline path: it is the one place where a thin
    split must be impossible rather than merely discouraged.

    FR-511a asks for something different from an exemption. A transition that
    knowingly falls below the five-season floor must be **expressible while
    labelled** — run, reported, counted, and barred from supporting a headline
    claim — which a type that refuses to construct it cannot deliver. So this
    type expresses it and `describe()` labels it `DIAGNOSTIC`, judging each
    transition against `MIN_TRAINING_SEASONS` independently, as FR-511a
    requires.

    Keeping it in this module rather than in `split.py` is the other half of
    the argument. The headline evaluation imports its split from `split.py` and
    would have to import *this* one deliberately, by name, from a module whose
    every output carries a compliance label. There is no route by which the
    headline path reaches a sub-minimum split by accident.

    The surface is the one `ranker.train` and `harness.evaluate` consume —
    neither does an isinstance check — so the two paths score models the same
    way even though they cannot share a split type.
    """

    training_seasons: tuple[int, ...]
    evaluated_season: int

    def __post_init__(self) -> None:
        # Normalised to a tuple of plain ints: these reach a JSON results file,
        # and `json` refuses an np.int64 with an error naming the type rather
        # than the field it came from.
        seasons = tuple(int(s) for s in self.training_seasons)
        object.__setattr__(self, "training_seasons", seasons)
        object.__setattr__(self, "evaluated_season", int(self.evaluated_season))

        if not seasons:
            raise TransferError(
                "a transfer arm with no training seasons is not an arm; the "
                "reset at the corpus start has nothing before it to train on"
            )
        if len(set(seasons)) != len(seasons):
            raise TransferError(f"training seasons {list(seasons)} contain a duplicate")
        if list(seasons) != sorted(seasons):
            raise TransferError(
                f"training seasons {list(seasons)} are not in chronological order"
            )
        offending = [s for s in seasons if s >= self.evaluated_season]
        if offending:
            raise TransferError(
                f"training season(s) {offending} are at or after the evaluated "
                f"season {self.evaluated_season}. FR-511 trains on seasons "
                f"*strictly* prior to the one being scored (FR-403)."
            )

    @classmethod
    def from_archive(
        cls, archive: Sequence[int], evaluated_season: int
    ) -> TransferSplit:
        """Every archive season strictly before `evaluated_season`, as training.

        The archive is passed in rather than resolved here so that the same
        construction serves the real corpus and a fixture, and so that no arm
        can quietly widen its own training set.
        """
        return cls(
            training_seasons=tuple(s for s in sorted(archive) if s < evaluated_season),
            evaluated_season=evaluated_season,
        )

    # -- the boundary -------------------------------------------------------

    @property
    def seasons(self) -> tuple[int, ...]:
        return (*self.training_seasons, self.evaluated_season)

    @property
    def holdout(self) -> tuple[int, ...]:
        """One season. The whole point of the experiment is that it is one."""
        return (self.evaluated_season,)

    @property
    def boundary_season(self) -> int:
        """The first held-out season. Nothing at or after it may train."""
        return self.evaluated_season

    @property
    def n_training_seasons(self) -> int:
        """FR-511a: the count that every arm reports and is judged on."""
        return len(self.training_seasons)

    @property
    def meets_constitutional_minimum(self) -> bool:
        """FR-511a, judged per transition against the FR-501 training floor.

        Only the training minimum applies. The holdout floor of three seasons
        is a property of the headline evaluation's design, and an experiment
        whose unit of observation is one reset season cannot satisfy it without
        ceasing to measure a reset.
        """
        return self.n_training_seasons >= MIN_TRAINING_SEASONS

    @property
    def label(self) -> str | None:
        """`DIAGNOSTIC` when this arm falls short, `None` when it complies."""
        return None if self.meets_constitutional_minimum else DIAGNOSTIC_LABEL

    # -- applying it --------------------------------------------------------

    def _seasons_of(self, df: pd.DataFrame) -> pd.Series:
        if "season" not in df.columns:
            raise TransferError("frame has no `season` column to split on")
        return pd.to_numeric(df["season"], errors="coerce")

    def train_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.training_seasons)

    def holdout_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.holdout)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Training rows and holdout rows, in that order.

        Seasons *after* the evaluated one belong to neither and are dropped.
        The frame handed in is the whole archive, so without this the 2014 arm
        would train on 2015 through 2025 — the future of the regime it is
        supposed to be meeting for the first time.
        """
        return df[self.train_mask(df)].copy(), df[self.holdout_mask(df)].copy()

    def assert_no_leakage(self, training: pd.DataFrame) -> None:
        """FR-403: nothing at or after the boundary may appear in training.

        Called by the trainer rather than trusted. The failure this prevents is
        specific to this experiment: an arm that leaked the reset season into
        training would report a *small* transfer penalty, which reads as good
        news and is the finding nobody would think to question.
        """
        seasons = set(self._seasons_of(training).dropna().astype(int))
        offending = sorted(s for s in seasons if s >= self.boundary_season)
        if offending:
            raise TransferError(
                f"training data contains season(s) {offending} at or after the "
                f"evaluated season {self.boundary_season} (FR-403)"
            )

    # -- reporting ----------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """What FR-511a requires of every arm in `results/`.

        `meets_constitutional_minimum` is computed against the imported
        `MIN_TRAINING_SEASONS` rather than a literal five, so that a change to
        the constitutional floor reaches this experiment instead of leaving it
        asserting an old one.
        """
        return {
            "training_seasons": list(self.training_seasons),
            "n_training_seasons": self.n_training_seasons,
            "evaluated_season": self.evaluated_season,
            # Named as the headline split names it, so the two records can be
            # read side by side and `harness.render_verdict` can read either.
            "holdout_seasons": list(self.holdout),
            "boundary_season": self.boundary_season,
            "meets_constitutional_minimum": self.meets_constitutional_minimum,
            "minimum_required": {"training": MIN_TRAINING_SEASONS},
            "label": self.label,
        }


# --------------------------------------------------------------------------
# Which transitions exist (FR-210, plan D-10)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transition:
    """One reset season, its transfer arm, and its comparator arm.

    The comparator is the identical protocol one season earlier: a model facing
    a season under rules it has already seen. Without it the transfer arm's
    score is uninterpretable — seasons differ in how predictable they are for
    reasons that have nothing to do with regulations.
    """

    regime: Regime
    transfer: TransferSplit
    comparator: TransferSplit

    @property
    def reset_season(self) -> int:
        return self.regime.start_season

    @property
    def meets_constitutional_minimum(self) -> bool:
        """Both arms, not one. A penalty is a difference between two fits.

        A transition whose comparator falls short is as unusable as one whose
        transfer arm does, because the penalty is computed from both.
        """
        return (
            self.transfer.meets_constitutional_minimum
            and self.comparator.meets_constitutional_minimum
        )

    @property
    def label(self) -> str | None:
        return None if self.meets_constitutional_minimum else DIAGNOSTIC_LABEL

    @property
    def supports_headline_claim(self) -> bool:
        """FR-511a: a short transition is reported and does not count."""
        return self.meets_constitutional_minimum


def reset_seasons(
    archive: Sequence[int], *, regimes: Sequence[Regime] = REGIMES
) -> tuple[int, ...]:
    """The declared major resets this archive can actually test (FR-210, D-10).

    Read from the regime table rather than hard-coded, so that when 2026
    completes the experiment gains an observation with no code change — which
    is why D-10 put a 2026 row in a table describing a 2010-2025 archive.

    A reset is testable only if the archive holds a season before its
    comparator: the 2010 row is both a regime boundary and the corpus start, so
    nothing precedes it to train on. Moderate regimes are excluded; FR-511 is
    about resets, and D-10 grades them.
    """
    seasons = sorted(int(s) for s in archive)
    if not seasons:
        raise TransferError("an empty archive has no transitions")
    first = seasons[0]
    return tuple(
        r.start_season
        for r in regimes
        if r.severity is RegimeSeverity.MAJOR
        and r.start_season in seasons
        # The comparator evaluates R-1 and must itself train on something.
        and r.start_season - 1 > first
    )


def transitions(
    archive: Sequence[int], *, regimes: Sequence[Regime] = REGIMES
) -> tuple[Transition, ...]:
    """Every testable transition in this archive, with both arms built."""
    table = {r.start_season: r for r in regimes}
    return tuple(
        Transition(
            regime=table[season],
            transfer=TransferSplit.from_archive(archive, season),
            comparator=TransferSplit.from_archive(archive, season - 1),
        )
        for season in reset_seasons(archive, regimes=regimes)
    )


# --------------------------------------------------------------------------
# The unpaired penalty
# --------------------------------------------------------------------------


def unpaired_delta(
    scores: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metrics: Sequence[Metric] = METRICS,
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = SEED,
) -> dict[str, Interval]:
    """`reference` minus `scores`, resampling each race set independently.

    `bootstrap_delta` is the right function everywhere the two frames cover the
    same races, and it raises rather than aligning when they do not — see D-5.
    Here they genuinely cannot: the comparator arm is scored on season R-1 and
    the transfer arm on season R, so there is no race that appears in both and
    nothing to pair. Rather than fake a pairing by position, each arm's races
    are resampled independently within the same replicate and the difference of
    the two replicate means is recorded.

    The resulting interval is wider than a paired one, and that width is real:
    it carries the between-season variability that a paired design would have
    cancelled. Every record this feeds is labelled `pairing: unavailable`.
    """
    if scores.empty or reference.empty:
        raise TransferError("an arm with no scored races has no penalty")

    rng = np.random.default_rng(seed)
    left_draws = rng.integers(0, len(reference), size=(replicates, len(reference)))
    right_draws = rng.integers(0, len(scores), size=(replicates, len(scores)))
    tail = (1.0 - confidence) / 2.0

    out: dict[str, Interval] = {}
    for metric in metrics:
        left = reference[metric.name].to_numpy(dtype="float64")
        right = scores[metric.name].to_numpy(dtype="float64")
        means = np.nanmean(left[left_draws], axis=1) - np.nanmean(
            right[right_draws], axis=1
        )
        out[metric.name] = Interval(
            point=float(np.nanmean(left) - np.nanmean(right)),
            low=float(np.nanpercentile(means, 100 * tail)),
            high=float(np.nanpercentile(means, 100 * (1.0 - tail))),
            confidence=confidence,
        )
    return out


# --------------------------------------------------------------------------
# Running one arm, one transition, and the whole experiment
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One fit, scored. Kept out of the record: `per_race` is a frame."""

    split: TransferSplit
    model: TrainedRanker
    per_race: dict[str, pd.DataFrame]
    metrics: dict[str, dict[str, Interval]]
    model_vs_grid: dict[str, Interval]
    races: int
    rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split.describe(),
            "races_scored": self.races,
            "rows_scored": self.rows,
            "training_rows": self.model.training_rows,
            "trees_kept": self.model.trees,
            "predictors": {
                name: {
                    "metrics": {k: v.as_dict() for k, v in intervals.items()},
                }
                for name, intervals in self.metrics.items()
            },
            # Paired, legitimately: both predictors saw the same races here.
            "model_minus_grid": {
                k: v.as_dict() for k, v in self.model_vs_grid.items()
            },
        }


def run_arm(
    built: Experiment,
    split: TransferSplit,
    *,
    params: Hyperparameters,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = SEED,
) -> ArmResult:
    """Fit on the arm's training seasons and score its one evaluated season.

    The model and the grid baseline both go through `harness.score_predictor`,
    the single scoring path of FR-304, so this experiment cannot score a model
    on terms the baseline did not face.
    """
    model = train(
        built.frame,
        split=split,
        features=built.feature_names,
        params=params,
        name=MODEL_NAME,
    )
    _, holdout = split.split(built.frame)
    if holdout.empty:
        raise TransferError(
            f"evaluated season {split.evaluated_season} selects no rows from "
            f"this corpus"
        )

    per_race = {
        BASELINE_NAME: score_predictor(GridOrderBaseline(), holdout),
        MODEL_NAME: score_predictor(model, holdout),
    }
    metrics = {
        name: bootstrap_metrics(frame, replicates=replicates, seed=seed)
        for name, frame in per_race.items()
    }
    return ArmResult(
        split=split,
        model=model,
        per_race=per_race,
        metrics=metrics,
        model_vs_grid=bootstrap_delta(
            per_race[MODEL_NAME],
            per_race[BASELINE_NAME],
            replicates=replicates,
            seed=seed,
        ),
        races=int(holdout[list(RACE_KEYS)].drop_duplicates().shape[0]),
        rows=int(len(holdout)),
    )


def _edge_over_grid(arm: ArmResult) -> pd.DataFrame:
    """Per race, how far the model beat the grid on that race.

    Within one arm the two predictors *were* scored on the same races, so this
    subtraction is properly paired — it is only the comparison *between* arms
    that cannot be. Differencing here first and comparing the differences
    across the boundary is therefore the sharpest reading available of "did
    the model transfer", because the part of a penalty that is simply a harder
    season cancels out of both arms.
    """
    model = arm.per_race[MODEL_NAME].set_index(list(RACE_KEYS)).sort_index()
    grid = arm.per_race[BASELINE_NAME].set_index(list(RACE_KEYS)).sort_index()
    columns = [m.name for m in METRICS]
    return (model[columns] - grid[columns]).reset_index()


def run_transition(
    built: Experiment,
    transition: Transition,
    *,
    params: Hyperparameters,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Both arms of one reset, and the penalty between them.

    The penalty is reported for the grid baseline as well as for the model.
    That number is the control: the grid order is the same kind of information
    in both seasons, so if it loses as much across the reset as the model does,
    what has been measured is a less predictable season rather than a failure
    to transfer.
    """
    arms = {
        TRANSFER_ARM: run_arm(
            built, transition.transfer, params=params, replicates=replicates, seed=seed
        ),
        COMPARATOR_ARM: run_arm(
            built,
            transition.comparator,
            params=params,
            replicates=replicates,
            seed=seed,
        ),
    }
    transfer, comparator = arms[TRANSFER_ARM], arms[COMPARATOR_ARM]

    penalty = {
        name: unpaired_delta(
            transfer.per_race[name],
            comparator.per_race[name],
            replicates=replicates,
            seed=seed,
        )
        for name in (MODEL_NAME, BASELINE_NAME)
    }
    # The difference of differences, and the sharpest reading available here.
    # Within one arm the model and the grid were scored on the same races, so
    # `_edge_over_grid` is properly paired; comparing those edges across the
    # boundary then cancels whatever made a season hard for everybody, which is
    # the part of a raw penalty that says nothing about transfer. It stays
    # `pairing: unavailable` all the same — the two arms are still different
    # races — but the quantity being differenced is now a within-season
    # contrast rather than a raw score.
    edge = unpaired_delta(
        _edge_over_grid(transfer),
        _edge_over_grid(comparator),
        replicates=replicates,
        seed=seed,
    )

    metric = PRIMARY_METRIC.name
    return {
        "reset_season": transition.reset_season,
        "regime": {
            "start_season": transition.regime.start_season,
            "severity": transition.regime.severity.value,
            "description": transition.regime.description,
        },
        # FR-511a: the label travels with the transition, not with a footnote.
        "label": transition.label,
        "meets_constitutional_minimum": transition.meets_constitutional_minimum,
        "supports_headline_claim": transition.supports_headline_claim,
        "n_training_seasons": {
            TRANSFER_ARM: transition.transfer.n_training_seasons,
            COMPARATOR_ARM: transition.comparator.n_training_seasons,
        },
        "arms": {name: arm.as_dict() for name, arm in arms.items()},
        "penalty": {
            "metric": metric,
            "definition": (
                "comparator score minus transfer score on the primary metric; "
                "positive means the reset season cost the model performance"
            ),
            "pairing": "unavailable",
            "pairing_reason": PAIRING_UNAVAILABLE,
            "interval_method": (
                f"unpaired race bootstrap, {replicates} replicates, each arm "
                f"resampled independently within a replicate"
            ),
            MODEL_NAME: penalty[MODEL_NAME][metric].as_dict(),
            # The control: how much the *grid* lost across the same boundary.
            BASELINE_NAME: penalty[BASELINE_NAME][metric].as_dict(),
            "all_metrics": {
                name: {k: v.as_dict() for k, v in intervals.items()}
                for name, intervals in penalty.items()
            },
        },
        "edge_over_grid": {
            "metric": metric,
            "definition": (
                "how much of the model's advantage over the grid was lost "
                "across the reset: (comparator model minus comparator grid) "
                "minus (transfer model minus transfer grid). Positive means "
                "the model transferred worse than the grid did. A season that "
                "was simply harder to predict cancels, because it hurt both "
                "predictors inside the same arm"
            ),
            "pairing": "paired within each arm, unpaired across them",
            "pairing_reason": PAIRING_UNAVAILABLE,
            "per_arm": {
                TRANSFER_ARM: transfer.model_vs_grid[metric].as_dict(),
                COMPARATOR_ARM: comparator.model_vs_grid[metric].as_dict(),
            },
            "lost_across_the_reset": edge[metric].as_dict(),
            "all_metrics": {k: v.as_dict() for k, v in edge.items()},
        },
    }


def collected_archive(root: Path = RAW_DIR) -> tuple[int, ...]:
    """Every collected season, from the checkpoints on disk (FR-511).

    Deliberately not `CorpusWindow`. The window is the right corpus for a
    result about present-day relevance and the wrong one for a diagnostic about
    regulatory history — see the module docstring, and FR-511's own sentence
    saying so.

    A hole in the archive is an error rather than a shorter list: a transition
    silently trained on eleven of twelve seasons would report a training-season
    count FR-511a requires to be accurate, and it would be wrong.
    """
    available = available_seasons(root)
    if not available:
        raise TransferError(
            "no season checkpoints on disk; the transfer experiment reads the "
            "collected archive and there is none"
        )
    archive = archive_seasons(max(available))
    missing = [s for s in archive if s not in available]
    if missing:
        raise TransferError(
            f"the archive {archive[0]}-{archive[-1]} is missing season(s) "
            f"{missing}. Refusing to run transitions whose training-season "
            f"counts would then misdescribe what they trained on (FR-511a)."
        )
    return archive


def run(
    *,
    experiment: Experiment | None = None,
    root: Path = RAW_DIR,
    decays: Sequence[float] = DECAY_SETTINGS,
    params: Hyperparameters | None = None,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = SEED,
    path: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """The whole experiment: every transition, at every decay setting.

    One corpus, built once, for all twelve fits. Rebuilding it per arm would
    make each arm's features a rebuild of the same computation and leave
    nothing able to prove the six models saw identical inputs.
    """
    if len(set(decays)) < 2 or not any(d > 0.0 for d in decays) or 0.0 not in decays:
        raise TransferError(
            f"decays={list(decays)}; FR-512 requires the experiment at zero "
            f"decay and at a non-zero one, so that the robustness benefit of "
            f"recency weighting is measured rather than asserted"
        )

    built = (
        experiment
        if experiment is not None
        else prepare(collected_archive(root), root=root)
    )
    archive = tuple(int(s) for s in sorted(built.frame["season"].unique()))
    found = transitions(archive)
    if not found:
        raise TransferError(
            f"the archive {archive[0]}-{archive[-1]} contains no testable major "
            f"reset from the FR-210 table"
        )
    base = params or Hyperparameters()

    runs: list[dict[str, Any]] = []
    for decay in decays:
        arm_params = replace(
            base, recency_decay=float(decay), calibration_seasons=CALIBRATION_SEASONS
        )
        logger.info("regime transfer at decay=%.3g", decay)
        runs.append(
            {
                "recency_decay": float(decay),
                "hyperparameters": arm_params.as_dict(),
                "transitions": [
                    run_transition(
                        built, t, params=arm_params, replicates=replicates, seed=seed
                    )
                    for t in found
                ],
            }
        )

    record: dict[str, Any] = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "experiment": "regulation_transfer",
        "primary_metric": PRIMARY_METRIC.name,
        # FR-511: stated in the file, because the obvious assumption about any
        # results file in this project is that it used the FR-508 window.
        "corpus": {
            "source": "full collected archive",
            "bound_to_rolling_window": False,
            "reason": (
                "FR-511: binding the experiment to a window that advances "
                "yearly would silently destroy observations as they age out"
            ),
            "seasons": list(archive),
            "first_season": archive[0],
            "last_season": archive[-1],
            "rows": len(built),
            "absent_sources": list(built.absent_sources),
        },
        "decay_settings": [float(d) for d in decays],
        "runs": runs,
    }
    record["headline"] = headline(record)
    record["recency_weighting"] = recency_effect(record)

    if write:
        written = write_results(record, path or RESULTS_DIR / "regime_transfer.json")
        logger.info("wrote %s", written)
    return record


# --------------------------------------------------------------------------
# What may be claimed from it
# --------------------------------------------------------------------------


def headline(record: dict[str, Any]) -> dict[str, Any]:
    """FR-511a: the count is of compliant transitions, and only those.

    Computed from the labels rather than written down. `n=2` is a fact about
    which transitions cleared the training floor, and a hand-written two would
    stop being true the moment a fourth reset entered the archive — or the
    moment a corpus narrowing pushed 2017 below the floor, which is the exact
    failure FR-511a exists to make visible.

    No mean is taken over the compliant transitions. Spec §3.5 is explicit that
    two observations are reported as two observations and never generalised
    into a rate, so the headline lists them.
    """
    per_decay: dict[str, Any] = {}
    compliant: list[int] = []
    diagnostic: list[int] = []
    for run_record in record["runs"]:
        key = f"{run_record['recency_decay']:g}"
        entries: dict[str, Any] = {"observations": [], "diagnostics": []}
        for t in run_record["transitions"]:
            line = {
                "reset_season": t["reset_season"],
                "n_training_seasons": t["n_training_seasons"][TRANSFER_ARM],
                "penalty": t["penalty"][MODEL_NAME],
                "grid_penalty": t["penalty"][BASELINE_NAME],
                "edge_lost_across_the_reset": (
                    t["edge_over_grid"]["lost_across_the_reset"]
                ),
                "label": t["label"],
            }
            if t["supports_headline_claim"]:
                entries["observations"].append(line)
            else:
                entries["diagnostics"].append(line)
        per_decay[key] = entries
        compliant = [e["reset_season"] for e in entries["observations"]]
        diagnostic = [e["reset_season"] for e in entries["diagnostics"]]

    return {
        "metric": record["primary_metric"],
        "n": len(compliant),
        "transitions": compliant,
        "diagnostic_transitions": diagnostic,
        "pairing": "unavailable",
        "statement": (
            f"Regulation-transfer penalty measured on n={len(compliant)} "
            f"compliant transition(s) {compliant}; "
            f"{diagnostic} reported as {DIAGNOSTIC_LABEL} and excluded from "
            f"this count (FR-511a). Two observations are two observations: "
            f"they are not averaged into a rate."
        ),
        "by_decay": per_decay,
    }


def recency_effect(record: dict[str, Any]) -> dict[str, Any]:
    """FR-512: did recency weighting actually reduce the penalty?

    Reported per transition and as a plain boolean, because "recency weighting
    makes the model robust to regulation change" is precisely the kind of claim
    FR-512 exists to stop anyone asserting for free. A penalty that grows under
    decay is a finding, not a bug.
    """
    zero = next(r for r in record["runs"] if r["recency_decay"] == 0.0)
    weighted = [r for r in record["runs"] if r["recency_decay"] != 0.0]
    baseline = {t["reset_season"]: t for t in zero["transitions"]}

    out: dict[str, Any] = {"reference_decay": 0.0, "comparisons": []}
    for run_record in weighted:
        for t in run_record["transitions"]:
            before = baseline[t["reset_season"]]["penalty"][MODEL_NAME]["point"]
            after = t["penalty"][MODEL_NAME]["point"]
            out["comparisons"].append(
                {
                    "reset_season": t["reset_season"],
                    "recency_decay": run_record["recency_decay"],
                    "label": t["label"],
                    "penalty_at_zero_decay": before,
                    "penalty_at_this_decay": after,
                    "change": after - before,
                    "reduced_the_penalty": after < before,
                }
            )
    headline_seasons = set(record["headline"]["transitions"])
    counted = [
        c for c in out["comparisons"] if c["reset_season"] in headline_seasons
    ]
    out["reduced_the_penalty_on_every_compliant_transition"] = bool(counted) and all(
        c["reduced_the_penalty"] for c in counted
    )
    out["compliant_transitions_considered"] = sorted(headline_seasons)
    return out


def render(record: dict[str, Any]) -> str:
    """The human-readable half, computed rather than written.

    Every line names its training-season count and its label, so a reader
    cannot take the diagnostic for a result by skimming.
    """
    head = record["headline"]
    lines = [
        f"Regulation-transfer penalty ({head['metric']}), "
        f"archive {record['corpus']['first_season']}-{record['corpus']['last_season']}, "
        f"not the FR-508 window.",
        head["statement"],
        "",
    ]
    for run_record in record["runs"]:
        lines.append(f"decay = {run_record['recency_decay']:g}")
        for t in run_record["transitions"]:
            p = t["penalty"][MODEL_NAME]
            grid = t["penalty"][BASELINE_NAME]
            mark = t["label"] or "compliant"
            lines.append(
                f"  {t['reset_season']}  train {t['n_training_seasons'][TRANSFER_ARM]:>2} "
                f"seasons  {mark:<10}  penalty {p['point']:+.4f} "
                f"(95% CI {p['ci_low']:+.4f} to {p['ci_high']:+.4f}"
                f"{', excludes zero' if p['excludes_zero'] else ', includes zero'})"
                f"  grid {grid['point']:+.4f}"
                f"  edge lost "
                f"{t['edge_over_grid']['lost_across_the_reset']['point']:+.4f}"
            )
        lines.append("")
    effect = record["recency_weighting"]
    lines.append(
        "Recency weighting reduced the penalty on every compliant transition: "
        f"{effect['reduced_the_penalty_on_every_compliant_transition']}"
    )
    lines.append(f"Penalty intervals are unpaired. {PAIRING_UNAVAILABLE}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(render(run()))
    return 0


__all__ = [
    "BASELINE_NAME",
    "CALIBRATION_SEASONS",
    "COMPARATOR_ARM",
    "DECAY_SETTINGS",
    "DEFAULT_NONZERO_DECAY",
    "DIAGNOSTIC_LABEL",
    "MODEL_NAME",
    "PAIRING_UNAVAILABLE",
    "TRANSFER_ARM",
    "ArmResult",
    "TransferError",
    "TransferSplit",
    "Transition",
    "collected_archive",
    "headline",
    "main",
    "recency_effect",
    "render",
    "reset_seasons",
    "run",
    "run_arm",
    "run_transition",
    "transitions",
    "unpaired_delta",
]


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
