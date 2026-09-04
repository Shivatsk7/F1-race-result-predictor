"""The `Predictor` protocol (FR-304, plan D-3).

Baselines and the model implement one interface, so evaluation is written once
against it. This is the structural half of FR-304: it becomes impossible to
evaluate the model on terms the baselines did not face, because there is only
one code path and it does not know which kind of predictor it holds.

v0's baseline gap went unnoticed precisely because the two were scored by
different code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Predictor(Protocol):
    """Turns one race's frame into a predicted finishing order.

    The frame carries the race's keys and features side by side; a predictor
    takes what it needs and ignores the rest. The return is a Series of
    positions 1..n aligned to the frame's index — *positions*, not scores, so
    that nothing downstream has to know whether higher or lower was better for
    a particular implementation.
    """

    #: Appears verbatim in the results file and the report.
    name: str

    def predict(self, race: pd.DataFrame) -> pd.Series: ...


class BarePointEstimateError(ValueError):
    """Raised when a frame publishes a predicted position with no attached
    uncertainty (FR-752, Constitution X.2)."""


def assert_carries_uncertainty(
    frame: pd.DataFrame,
    *,
    point_column: str = "predicted_position",
    lower_column: str = "predicted_position_lower",
    upper_column: str = "predicted_position_upper",
) -> None:
    """FR-752: a *published* predicted position may never stand alone.

    This governs what a frame hands to a reader — `predict_with_interval`'s
    output, a simulation's aggregated table — not `Predictor.predict()`
    itself. `predict()`'s bare `pd.Series` of positions is FR-304's internal
    comparison primitive: the evaluation harness scores a predicted *order*
    against the actual one, and an interval has no meaning there. FR-752
    concerns the boundary where a position is shown to someone as a forecast,
    which is a different act from ranking one predictor against another.

    A frame with no `point_column` at all passes trivially — there is nothing
    published here to police. One that has it must carry both interval bounds
    on every row; a null bound is not an interval, only the appearance of one.
    """
    if point_column not in frame.columns:
        return
    missing = [c for c in (lower_column, upper_column) if c not in frame.columns]
    if missing:
        raise BarePointEstimateError(
            f"{point_column!r} is published without {missing} attached — "
            f"FR-752 forbids a bare point estimate"
        )
    if frame[lower_column].isna().any() or frame[upper_column].isna().any():
        raise BarePointEstimateError(
            f"{point_column!r} carries a null interval bound on at least one "
            f"row — a missing bound is not an interval, it only looks like one "
            f"until it is read"
        )


def rank_to_positions(values: pd.Series, *, ascending: bool) -> pd.Series:
    """Turn any ordering score into dense positions 1..n.

    `method="first"` rather than `"min"` or `"average"`: a finishing order is a
    permutation, and tied ranks would produce two cars in P3 and none in P4,
    which every metric here would then score against a position that does not
    exist. Ties break by the frame's existing order, which the caller has
    already made deterministic.
    """
    return (
        values.rank(method="first", ascending=ascending)
        .astype("int64")
        .rename("predicted_position")
    )


__all__ = [
    "BarePointEstimateError",
    "Predictor",
    "assert_carries_uncertainty",
    "rank_to_positions",
]
