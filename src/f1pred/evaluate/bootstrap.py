"""Race-level bootstrap intervals (FR-503, FR-504, plan D-5).

**Races are resampled, not driver-rows.** Twenty cars in one race are not
twenty independent observations — they compete for the same twenty positions,
so one car finishing higher forces another lower. Resampling rows would treat
that dependence as extra evidence and produce intervals narrow enough to make
almost any difference look significant. This is the single most common way an
honest-looking evaluation lies, and D-5 exists to forbid it.

**Deltas are computed inside each replicate.** Bootstrapping two predictors
separately and subtracting their intervals throws away the pairing: both are
scored on the *same* races, so a race that is hard for one is usually hard for
the other, and the difference is far better determined than either level. The
paired interval is narrower and it is the honest one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from f1pred.config import SEED
from f1pred.evaluate.metrics import METRICS, Metric, RACE_KEYS

#: Enough that the interval endpoints are stable to the precision anyone reads
#: them at. Cheap: a replicate is a mean over a few hundred numbers.
DEFAULT_REPLICATES: Final[int] = 2000

DEFAULT_CONFIDENCE: Final[float] = 0.95


class BootstrapError(ValueError):
    """Raised when a bootstrap cannot be computed as specified."""


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate and its percentile bootstrap interval."""

    point: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def excludes_zero(self) -> bool:
        """FR-507's test for whether a delta is real."""
        return (self.low > 0.0) or (self.high < 0.0)

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence": self.confidence,
            "excludes_zero": self.excludes_zero,
        }


def _race_index(scores: pd.DataFrame) -> pd.Index:
    missing = [k for k in RACE_KEYS if k not in scores.columns]
    if missing:
        raise BootstrapError(f"per-race scores are missing {missing}")
    return pd.MultiIndex.from_frame(scores[list(RACE_KEYS)])


def _replicate_draws(
    n_races: int, replicates: int, seed: int
) -> np.ndarray:
    """Indices for every replicate at once. Seeded, so a run reproduces."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_races, size=(replicates, n_races))


def _percentiles(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.nanpercentile(samples, 100 * tail)),
        float(np.nanpercentile(samples, 100 * (1.0 - tail))),
    )


def bootstrap_metrics(
    scores: pd.DataFrame,
    *,
    metrics: Sequence[Metric] = METRICS,
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = SEED,
) -> dict[str, Interval]:
    """FR-503: a 95% interval on every metric, by resampling races."""
    if scores.empty:
        raise BootstrapError("no races to bootstrap")
    draws = _replicate_draws(len(scores), replicates, seed)

    out: dict[str, Interval] = {}
    for metric in metrics:
        values = scores[metric.name].to_numpy(dtype="float64")
        resampled = values[draws]
        means = np.nanmean(resampled, axis=1)
        low, high = _percentiles(means, confidence)
        out[metric.name] = Interval(
            point=float(np.nanmean(values)), low=low, high=high, confidence=confidence
        )
    return out


def bootstrap_delta(
    scores: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metrics: Sequence[Metric] = METRICS,
    replicates: int = DEFAULT_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = SEED,
) -> dict[str, Interval]:
    """FR-504: model-minus-baseline per metric, paired within each replicate.

    Both frames are aligned on the race key first and an unmatched race is an
    error rather than a dropped row: a delta computed over different race sets
    is not a delta, and silently discarding the mismatch would hide the fact
    that the two predictors were not scored on the same thing.
    """
    left = scores.set_index(_race_index(scores)).sort_index()
    right = reference.set_index(_race_index(reference)).sort_index()
    if not left.index.equals(right.index):
        only_left = left.index.difference(right.index)
        only_right = right.index.difference(left.index)
        raise BootstrapError(
            f"the two score frames cover different races; "
            f"{len(only_left)} only in the first, {len(only_right)} only in the "
            f"second. FR-304 requires the identical holdout for both."
        )

    draws = _replicate_draws(len(left), replicates, seed)
    out: dict[str, Interval] = {}
    for metric in metrics:
        difference = (
            left[metric.name].to_numpy(dtype="float64")
            - right[metric.name].to_numpy(dtype="float64")
        )
        means = np.nanmean(difference[draws], axis=1)
        low, high = _percentiles(means, confidence)
        out[metric.name] = Interval(
            point=float(np.nanmean(difference)),
            low=low,
            high=high,
            confidence=confidence,
        )
    return out


def mean_of_intervals(intervals: Sequence[dict[str, Interval]]) -> dict[str, Interval]:
    """Average many predictors' intervals into one — FR-303's random control.

    The point estimate is the mean of the point estimates; the interval spans
    the mean of the lows to the mean of the highs. This is a summary of a
    thousand draws, not a bootstrap over them, and it is labelled as such
    wherever it is reported.
    """
    if not intervals:
        raise BootstrapError("nothing to average")
    names = intervals[0].keys()
    return {
        name: Interval(
            point=float(np.mean([i[name].point for i in intervals])),
            low=float(np.mean([i[name].low for i in intervals])),
            high=float(np.mean([i[name].high for i in intervals])),
            confidence=intervals[0][name].confidence,
        )
        for name in names
    }


__all__ = [
    "BootstrapError",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_REPLICATES",
    "Interval",
    "bootstrap_delta",
    "bootstrap_metrics",
    "mean_of_intervals",
]
