"""Sequential pairwise ratings (FR-203).

Ported from v0's `_compute_sequential_elo`, whose structure was sound: it walks
races in order, records each entrant's rating *before* the race, and only then
applies the update. That ordering is the whole reason an Elo can be a pre-race
feature at all, so it is preserved exactly and asserted by the behavioural
leakage check rather than left as a comment.

What changed from v0: the walk is driven by the chronological race ordinal
rather than by a `(year, round)` groupby, so a gap in the corpus (FR-104's
discarded race) advances the sequence by one race instead of by one calendar
slot; and ties are scored 0.5 rather than being skipped.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from f1pred.features.priors import assert_sorted, race_ordinal

#: Every entrant starts here. The value is arbitrary and cancels out of any
#: within-race comparison; it is kept at the conventional 1500 so the numbers
#: are recognisable when inspected.
DEFAULT_RATING: Final[float] = 1500.0


def sequential_elo(
    df: pd.DataFrame,
    *,
    entity: str,
    outcome: str,
    k_factor: float,
    default_rating: float = DEFAULT_RATING,
) -> pd.Series:
    """Each row's rating **as it stood before** that row's race.

    `outcome` is ordered lower-is-better — a finishing or grid position. Rows
    whose outcome is missing take part in no comparison but still receive the
    prior rating, so a driver who did not start does not silently reset.

    The update is normalised by the number of opponents, so a twenty-four-car
    grid does not move ratings twice as far as a twelve-car one.
    """
    assert_sorted(df)
    ordinals = race_ordinal(df)
    entities = df[entity].to_numpy()
    outcomes = pd.to_numeric(df[outcome], errors="coerce").to_numpy(dtype="float64")

    ratings: dict[object, float] = {}
    priors = np.empty(len(df), dtype="float64")

    order = np.asarray(ordinals)
    # Rows are already in chronological order, so races are contiguous blocks.
    boundaries = np.flatnonzero(np.diff(order)) + 1
    for block in np.split(np.arange(len(df)), boundaries):
        for i in block:
            priors[i] = ratings.get(entities[i], default_rating)

        contested = [i for i in block if not np.isnan(outcomes[i])]
        if len(contested) < 2:
            continue

        deltas: dict[object, float] = {entities[i]: 0.0 for i in block}
        for a, i in enumerate(contested):
            rating_i = priors[i]
            for j in contested[a + 1 :]:
                rating_j = priors[j]
                expected_i = 1.0 / (1.0 + 10 ** ((rating_j - rating_i) / 400.0))
                if outcomes[i] < outcomes[j]:
                    actual_i = 1.0
                elif outcomes[i] > outcomes[j]:
                    actual_i = 0.0
                else:
                    actual_i = 0.5
                delta = k_factor * (actual_i - expected_i)
                deltas[entities[i]] += delta
                deltas[entities[j]] -= delta

        normaliser = max(1, len(contested) - 1)
        for name, delta in deltas.items():
            ratings[name] = ratings.get(name, default_rating) + delta / normaliser

    return pd.Series(priors, index=df.index, dtype="float64")


__all__ = ["DEFAULT_RATING", "sequential_elo"]
