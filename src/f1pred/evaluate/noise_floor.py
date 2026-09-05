"""The irreducible noise floor (FR-754, FR-755, OQ-4, Constitution X.5).

Two estimators, both named in the spec's own note on FR-754 method and in
OQ-4, and both run — "both are cheap, so likely both with the wider bound
reported." Neither claims precision the corpus cannot support; the spec says
so explicitly ("the estimate will be crude") and this module's job is to make
the crudeness a stated number rather than a vague gesture at "racing is
chaotic".

**Near-identical-race variance, operationalised as teammate pairs.** The spec
names "races that are near-identical in all pre-race respects (same circuit,
near-identical grid and car order)". Matching whole *races* this way turns out
to be nearly unusable: two Grands Prix at the same circuit, even a season
apart, rarely share more than a handful of drivers, so a race-level match
starves the estimator of pairs. Two **teammates in the same race** are a
near-identical pre-race pair by construction — identical car, identical
circuit, identical day — filtered to a small grid gap so "near-identical
grid order" holds too. What a pre-race model, however good, cannot explain is
then read off how much *further* apart they finish than their grid gap alone
would suggest: `Var(finish_A - finish_B)` among such pairs, halved because two
independent noise draws sum their variances, expressed as a share of the
corpus's overall finishing-position variance.

**DNF-bound decomposition, operationalised as a status-string classification.**
`status_raw` (FR-003) already carries the verbatim reason; this module sorts
its ~30 corpus values into unambiguous racing incidents (a collision predicts
from nothing pre-race) and mechanical failures (partially anticipated by
`constructor_reliability_season`, but not the specific failure on the specific
day), and reports the combined row share as a second, independent bound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from f1pred.config import RESULTS_DIR
from f1pred.provenance import Scope, stamp
from f1pred.data.schema import ClassificationOutcome
from f1pred.dataset import enrich, load_corpus

#: Unambiguous racing incidents: nothing pre-race predicts a specific
#: collision. Measured against the real corpus's `status_raw` values (T-512).
INCIDENT_STATUSES: Final[frozenset[str]] = frozenset(
    {"Collision", "Accident", "Collision damage", "Spun off"}
)

#: Mechanical failures. Not wholly unpredictable — reliability priors carry
#: some signal — but the specific failure, on the specific car, on the
#: specific day, is not something any pre-race feature claims to know.
MECHANICAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "Engine", "Gearbox", "Brakes", "Hydraulics", "Suspension", "Power Unit",
        "Wheel", "Electrical", "Oil leak", "Puncture", "Power loss", "Mechanical",
        "Wheel nut", "Transmission", "Water leak", "Turbo", "Overheating",
        "Fuel pressure", "Water pressure", "Exhaust", "Clutch", "Rear wing",
        "Throttle", "Battery", "Electronics", "Steering", "Front wing", "ERS",
        "Undertray", "Tyre", "Oil pressure", "Alternator", "Vibrations", "Damage",
    }
)

#: Grid slots this far apart or closer count as "near-identical grid order"
#: for the teammate-pair estimator.
DEFAULT_MAX_GRID_GAP: Final[int] = 3

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "round",
    "constructor_id",
    "classification_outcome",
    "status_raw",
    "grid_position",
    "classified_position",
)


class NoiseFloorError(ValueError):
    """Raised when a noise-floor estimate cannot be computed as specified."""


@dataclass(frozen=True, slots=True)
class NoiseFloorEstimate:
    method: str
    value: float
    n_observations: int
    description: str
    #: Position-standard-deviation units, for methods where that is
    #: meaningful (`teammate_pair_floor`) — `None` for a row-share method
    #: (`dnf_incident_floor`), where a single position number would imply a
    #: precision the count-based method does not have.
    position_units: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "value": self.value,
            "n_observations": self.n_observations,
            "description": self.description,
            "position_units": self.position_units,
        }


def dnf_incident_floor(raw_corpus: pd.DataFrame) -> NoiseFloorEstimate:
    """The row-share of unambiguous incidents and mechanical failures.

    A crude proxy for "share of variance", counting affected *rows* rather
    than partitioning variance — the spec's own method note calls the whole
    estimate crude, and a row share is honest about which kind of crude this
    is: it is a lower bound on how much of the corpus no pre-race feature
    could have anticipated, not a variance decomposition.
    """
    missing = [c for c in ("status_raw",) if c not in raw_corpus.columns]
    if missing:
        raise NoiseFloorError(f"cannot compute DNF bound: missing {missing}")
    if raw_corpus.empty:
        raise NoiseFloorError("cannot compute DNF bound from an empty corpus")
    status = raw_corpus["status_raw"]
    affected = status.isin(INCIDENT_STATUSES) | status.isin(MECHANICAL_STATUSES)
    return NoiseFloorEstimate(
        method="dnf_incident_bound",
        value=float(affected.mean()),
        n_observations=int(len(raw_corpus)),
        description=(
            f"share of all rows whose status_raw is an unambiguous racing "
            f"incident ({sorted(INCIDENT_STATUSES)}) or a mechanical failure "
            f"({len(MECHANICAL_STATUSES)} distinct values); a lower bound, "
            f"not a variance share"
        ),
    )


def teammate_pair_floor(
    raw_corpus: pd.DataFrame, *, max_grid_gap: int = DEFAULT_MAX_GRID_GAP
) -> NoiseFloorEstimate:
    """Residual finish-position variance among near-identical teammate pairs.

    Restricted to classified finishers on both sides of the pair — a DNF is
    already counted by `dnf_incident_floor`, and mixing the two methods would
    double-count the same rows under two different names.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_corpus.columns]
    if missing:
        raise NoiseFloorError(f"cannot compute teammate-pair bound: missing {missing}")
    if max_grid_gap < 0:
        raise NoiseFloorError(f"max_grid_gap must be non-negative, got {max_grid_gap}")

    classified = raw_corpus[
        raw_corpus["classification_outcome"] == ClassificationOutcome.CLASSIFIED.value
    ]
    if classified.empty:
        raise NoiseFloorError("no classified rows to pair")

    diffs: list[float] = []
    for _, group in classified.groupby(["season", "round", "constructor_id"], sort=False):
        if len(group) != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        grid_gap = abs(float(a["grid_position"]) - float(b["grid_position"]))
        if grid_gap > max_grid_gap:
            continue
        diffs.append(float(a["classified_position"]) - float(b["classified_position"]))

    if len(diffs) < 2:
        raise NoiseFloorError(
            f"only {len(diffs)} near-identical teammate pair(s) found within "
            f"a grid gap of {max_grid_gap}; too few to estimate a variance"
        )

    diff_array = np.array(diffs, dtype="float64")
    overall_positions = pd.to_numeric(classified["classified_position"], errors="coerce")
    overall_variance = float(overall_positions.var(ddof=1))
    if overall_variance <= 0:
        raise NoiseFloorError("corpus-wide finishing-position variance is zero or undefined")

    # Two independent noise draws (one per teammate) sum their variances in
    # their difference, so halving recovers the per-driver noise variance.
    residual_variance = float(diff_array.var(ddof=1)) / 2.0
    share = float(np.clip(residual_variance / overall_variance, 0.0, 1.0))

    return NoiseFloorEstimate(
        method="near_identical_teammate_variance",
        value=share,
        n_observations=len(diffs),
        description=(
            f"Var(finish_A - finish_B)/2 over corpus-wide Var(finish position), "
            f"for {len(diffs)} classified teammate pairs within a {max_grid_gap}-"
            f"position grid gap — identical car, identical circuit, identical "
            f"day, near-identical starting position"
        ),
        # A standard deviation, in the same position units as the MAE metric
        # T-513 reads the model's error from — a variance share alone cannot
        # be set beside "3.4 positions" without this conversion.
        position_units=float(np.sqrt(residual_variance)),
    )


@dataclass(frozen=True, slots=True)
class NoiseFloorReport:
    estimates: dict[str, NoiseFloorEstimate]
    headline_method: str
    headline_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimates": {k: v.as_dict() for k, v in self.estimates.items()},
            # OQ-4: "likely both, with the wider bound reported" — the wider
            # one is the headline precisely because the estimate is a floor,
            # and reporting the narrower of two crude bounds as if it were
            # the answer would understate how much is genuinely unexplainable.
            "headline_method": self.headline_method,
            "headline_value": self.headline_value,
        }


def estimate_noise_floor(
    raw_corpus: pd.DataFrame, *, max_grid_gap: int = DEFAULT_MAX_GRID_GAP
) -> NoiseFloorReport:
    estimates = {
        "dnf_incident_bound": dnf_incident_floor(raw_corpus),
        "near_identical_teammate_variance": teammate_pair_floor(
            raw_corpus, max_grid_gap=max_grid_gap
        ),
    }
    headline_method = max(estimates, key=lambda name: estimates[name].value)
    return NoiseFloorReport(
        estimates=estimates,
        headline_method=headline_method,
        headline_value=estimates[headline_method].value,
    )


def error_relative_to_floor(
    model_mae: float, floor_report: NoiseFloorReport
) -> dict[str, Any]:
    """FR-755: state the model's error against the floor, not against zero.

    Reads the position-unit estimate from `near_identical_teammate_variance` —
    the one bound expressed in the model's own units (T-513). `model_mae` is
    an MAE in position units, e.g. `results/model_vs_baselines_*.json`'s
    `predictors.<name>.metrics.mae.point`.
    """
    teammate = floor_report.estimates.get("near_identical_teammate_variance")
    if teammate is None or teammate.position_units is None:
        raise NoiseFloorError(
            "floor_report has no position-unit estimate to compare the model's "
            "MAE against"
        )
    floor_positions = teammate.position_units
    return {
        "model_mae_positions": model_mae,
        "floor_positions": floor_positions,
        "model_error_minus_floor_positions": model_mae - floor_positions,
        "note": (
            "the model's mean absolute error, read against a floor measured "
            "from real near-identical teammate pairs rather than against "
            "zero (FR-755, Constitution X.5); a model whose error already "
            "sits near the floor has little further room a pre-race model "
            "could close"
        ),
    }


def run_noise_floor_estimate(
    *,
    seasons: tuple[int, ...] | None = None,
    model_mae: float | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Disk to a committed noise-floor record in one call (T-512, T-513).

    `model_mae` is optional so the floor itself can be estimated and
    committed independently of any particular model run; when supplied
    (`results/model_vs_baselines_*.json`'s own MAE), the record also carries
    FR-755's interpretation of that error against the floor.
    """
    raw = enrich(load_corpus(seasons))
    report = estimate_noise_floor(raw)
    record: dict[str, Any] = {
        "n_corpus_rows": int(len(raw)),
        "seasons": sorted(int(s) for s in raw["season"].unique()),
        "noise_floor": report.as_dict(),
    }
    if model_mae is not None:
        record["model_error_in_context"] = error_relative_to_floor(model_mae, report)
    # The floor is measured over the whole corpus rather than the window: it
    # is a property of the sport, and more teammate pairs make it a better
    # estimate (AC-11's non-window scope).
    stamp(
        record,
        scope=Scope.ARCHIVE,
        reason=(
            "the noise floor is a property of the sport, estimated from every "
            "classified teammate pair on disk; restricting it to the rolling "
            "window would discard evidence without making the estimate more "
            "current"
        ),
    )
    target = path or (RESULTS_DIR / "noise_floor.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


__all__ = [
    "DEFAULT_MAX_GRID_GAP",
    "INCIDENT_STATUSES",
    "MECHANICAL_STATUSES",
    "NoiseFloorError",
    "NoiseFloorEstimate",
    "NoiseFloorReport",
    "dnf_incident_floor",
    "error_relative_to_floor",
    "estimate_noise_floor",
    "run_noise_floor_estimate",
    "teammate_pair_floor",
]
