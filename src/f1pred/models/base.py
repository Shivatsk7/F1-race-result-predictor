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


__all__ = ["Predictor", "rank_to_positions"]
