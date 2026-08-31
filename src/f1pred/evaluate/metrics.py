"""The metric battery (FR-500, FR-502).

Nine metrics, named once, computed once, imported everywhere. FR-500 pins one
of them as **the primary metric**, and the pinning is load-bearing: every
requirement that gates a claim on performance gates on that one, so it must not
be possible for a call site to quietly gate on a different one because it
happened to look better.

**Everything is computed per race, then averaged unweighted across races.**
FR-500 says so for Spearman and the same treatment is applied to the rest, for
a reason worth stating: a 2010 race has 24 cars and a 2021 race has 20, so
pooling driver-rows would silently weight seasons by field size. A race is the
unit of the question "can this beat the grid", so a race is the unit here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats


class MetricError(ValueError):
    """Raised when a metric cannot be computed from what it was given."""


@dataclass(frozen=True, slots=True)
class Metric:
    """One named metric and the direction that counts as better."""

    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool
    description: str

    def __call__(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        return float(self.fn(predicted, actual))


# --------------------------------------------------------------------------
# The nine
# --------------------------------------------------------------------------


def _mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(predicted - actual).mean())


def _within(tolerance: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def inner(predicted: np.ndarray, actual: np.ndarray) -> float:
        return float((np.abs(predicted - actual) <= tolerance).mean())

    return inner


def _spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation within one race.

    A race in which every car is predicted the same position has no rank
    information and scipy returns NaN. That is the honest answer and it is left
    as NaN rather than coerced to zero, because zero is a real value meaning
    "no better than chance" and a degenerate prediction is not that.
    """
    if len(predicted) < 2:
        return float("nan")
    result = stats.spearmanr(predicted, actual)
    return float(result.statistic)


def _top_k_overlap(k: int) -> Callable[[np.ndarray, np.ndarray], float]:
    """Share of the actual top k that the prediction also put in its top k."""

    def inner(predicted: np.ndarray, actual: np.ndarray) -> float:
        if len(predicted) < k:
            return float("nan")
        predicted_set = set(np.argsort(predicted, kind="stable")[:k])
        actual_set = set(np.argsort(actual, kind="stable")[:k])
        return len(predicted_set & actual_set) / k

    return inner


def _podium_set_exact(predicted: np.ndarray, actual: np.ndarray) -> float:
    """1.0 only if the predicted top three are exactly the actual top three.

    Distinct from `top_3_set_accuracy`, which gives partial credit. FR-502
    lists both without defining either, so the distinction is declared here:
    one asks "how much of the podium did you get", the other "did you get the
    podium". Getting two of three right is a materially different result from
    getting all three, and a single partial-credit number hides which happened.
    """
    if len(predicted) < 3:
        return float("nan")
    predicted_set = set(np.argsort(predicted, kind="stable")[:3])
    actual_set = set(np.argsort(actual, kind="stable")[:3])
    return float(predicted_set == actual_set)


#: FR-502's nine, in the order the requirement lists them. The order is the
#: report's column order, so it stays fixed.
METRICS: Final[tuple[Metric, ...]] = (
    Metric("mae", _mae, False, "Mean absolute position error."),
    Metric(
        "exact_accuracy",
        _within(0),
        True,
        "Share of cars placed in exactly the right position.",
    ),
    Metric("within_1", _within(1), True, "Share placed within one position."),
    Metric("within_2", _within(2), True, "Share placed within two positions."),
    Metric("within_3", _within(3), True, "Share placed within three positions."),
    Metric(
        "spearman",
        _spearman,
        True,
        "Rank correlation between predicted and actual order.",
    ),
    Metric(
        "top_3_set_accuracy",
        _top_k_overlap(3),
        True,
        "Share of the actual top three that was predicted in the top three.",
    ),
    Metric(
        "top_5_set_accuracy",
        _top_k_overlap(5),
        True,
        "Share of the actual top five that was predicted in the top five.",
    ),
    Metric(
        "podium_set_accuracy",
        _podium_set_exact,
        True,
        "Races where the predicted podium set exactly matches the actual one.",
    ),
)

METRICS_BY_NAME: Final[Mapping[str, Metric]] = {m.name: m for m in METRICS}

#: FR-500. Wherever this spec or the Constitution says "the primary metric",
#: it means this one. Imported rather than re-specified, so no call site can
#: gate a claim on a metric that happens to look better.
PRIMARY_METRIC: Final[Metric] = METRICS_BY_NAME["spearman"]

PRIMARY_METRIC_NAME: Final[str] = PRIMARY_METRIC.name


# --------------------------------------------------------------------------
# Per-race computation
# --------------------------------------------------------------------------

RACE_KEYS: Final[tuple[str, str]] = ("season", "round")


def per_race_scores(
    keys: pd.DataFrame,
    predicted: pd.Series,
    actual: pd.Series,
    *,
    metrics: Sequence[Metric] = METRICS,
) -> pd.DataFrame:
    """One row per race, one column per metric.

    Returning per-race scores rather than a single number is what makes
    FR-503's bootstrap and FR-505's segmentation possible: both resample or
    group races, and neither can be reconstructed from an average.
    """
    missing = [k for k in RACE_KEYS if k not in keys.columns]
    if missing:
        raise MetricError(f"keys frame is missing {missing}")
    if not (len(keys) == len(predicted) == len(actual)):
        raise MetricError(
            f"length mismatch: keys={len(keys)} predicted={len(predicted)} "
            f"actual={len(actual)}"
        )

    frame = keys[list(RACE_KEYS)].copy().reset_index(drop=True)
    frame["_predicted"] = pd.to_numeric(predicted, errors="coerce").to_numpy()
    frame["_actual"] = pd.to_numeric(actual, errors="coerce").to_numpy()
    frame = frame.dropna(subset=["_predicted", "_actual"])

    rows = []
    for (season, rnd), race in frame.groupby(list(RACE_KEYS), sort=True):
        p = race["_predicted"].to_numpy(dtype="float64")
        a = race["_actual"].to_numpy(dtype="float64")
        row: dict[str, object] = {"season": int(season), "round": int(rnd), "cars": len(p)}
        for metric in metrics:
            row[metric.name] = metric(p, a)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(
    scores: pd.DataFrame, *, metrics: Sequence[Metric] = METRICS
) -> dict[str, float]:
    """Unweighted mean across races (FR-500).

    NaN races — a Spearman over a one-car race, say — are skipped for that
    metric rather than dragging the mean, and the count that actually
    contributed is available from `scores` for anyone who needs it.
    """
    return {m.name: float(scores[m.name].mean(skipna=True)) for m in metrics}


def score(
    keys: pd.DataFrame,
    predicted: pd.Series,
    actual: pd.Series,
    *,
    metrics: Sequence[Metric] = METRICS,
) -> dict[str, float]:
    """Per-race then averaged, in one call."""
    return aggregate(per_race_scores(keys, predicted, actual, metrics=metrics), metrics=metrics)


__all__ = [
    "METRICS",
    "METRICS_BY_NAME",
    "Metric",
    "MetricError",
    "PRIMARY_METRIC",
    "PRIMARY_METRIC_NAME",
    "RACE_KEYS",
    "aggregate",
    "per_race_scores",
    "score",
]
