"""Strict-prior computation for rolling features (FR-203, T-204).

Every rolling, form, and aggregate quantity in this project is computed here,
through one of three functions, for one reason: FR-203 says such a feature may
read only races *strictly preceding* the race it describes, and a rule enforced
in one place can be tested in one place.

Two traps this module exists to avoid, both of which v0 fell into.

**The teammate trap.** A constructor has two entries in a race. Shifting a
driver-row series by one inside a constructor group therefore hands a driver
their own teammate's result from the very race being predicted. Everything here
aggregates to the (entity, race) grain *first*, shifts there, and only then maps
back to rows, so a shift is always a shift by one race.

**The sort trap.** `shift(1)` means "the previous row", which is the previous
*race* only if the frame is in chronological order. These functions assert that
rather than assuming it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd

#: Identifies a race. Chronology comes from `race_date`; the round is the
#: tie-break, since a date alone has repeated across a double-header weekend.
RACE_KEYS: Final[tuple[str, ...]] = ("season", "round")

#: The canonical row order every feature function may rely on. Deterministic
#: and total: no two rows share a (season, round, driver) key (schema PRIMARY_KEY).
SORT_KEYS: Final[tuple[str, ...]] = ("race_date", "season", "round", "driver_id")

Aggregation = Literal["mean", "sum", "min", "max", "count"]


class PriorError(ValueError):
    """Raised when a prior cannot be computed safely."""


def sort_canonically(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` in the canonical chronological row order, index reset.

    Sorting is by declared keys with a stable algorithm, so the result depends
    on the frame's content and not on the order it happened to arrive in
    (FR-208).
    """
    missing = [k for k in SORT_KEYS if k not in df.columns]
    if missing:
        raise PriorError(f"cannot sort: missing key columns {missing}")
    return df.sort_values(list(SORT_KEYS), kind="stable").reset_index(drop=True)


def assert_sorted(df: pd.DataFrame) -> None:
    """Raise unless `df` is in canonical order. Cheap; called by every prior."""
    keys = df[list(SORT_KEYS)]
    if not keys.equals(keys.sort_values(list(SORT_KEYS), kind="stable")):
        raise PriorError(
            "frame is not in canonical chronological order. A prior computed "
            "over unsorted rows silently reads the future; call "
            "`sort_canonically` before building features."
        )


def race_ordinal(df: pd.DataFrame) -> pd.Series:
    """A dense chronological integer per race, shared by every row of that race.

    Gaps in the corpus (FR-104's discarded races) compress rather than leave a
    hole, which is what FR-104b asks for: a rolling window treats a missing
    race as absent and keeps its true count of races.
    """
    races = (
        df[["race_date", *RACE_KEYS]]
        .drop_duplicates()
        .sort_values(["race_date", *RACE_KEYS], kind="stable")
        .reset_index(drop=True)
    )
    races["_ordinal"] = np.arange(len(races), dtype="int64")
    key = pd.MultiIndex.from_frame(df[list(RACE_KEYS)])
    lookup = races.set_index(list(RACE_KEYS))["_ordinal"]
    return pd.Series(lookup.reindex(key).to_numpy(), index=df.index, dtype="int64")


def _entity_columns(entity: str | Sequence[str]) -> list[str]:
    return [entity] if isinstance(entity, str) else list(entity)


def prior_stat(
    df: pd.DataFrame,
    *,
    entity: str | Sequence[str],
    value: str,
    how: Aggregation = "mean",
    window: int | None = None,
    within_race: Aggregation = "mean",
    min_periods: int = 1,
) -> pd.Series:
    """A statistic over an entity's races **strictly before** each row's race.

    `entity` is one or more columns identifying whose history this is — a
    driver, a constructor, or a pair such as (driver, circuit). `value` is the
    column summarised. `within_race` collapses an entity's several rows in one
    race before any shifting happens; `window` bounds the look-back to that
    many prior races, or expands over all of them when None.

    Returns float64 aligned to `df.index`, NaN where the entity has no prior
    race. The NaN is deliberate and is FR-212's business to fill: an unseeded
    rookie must not be handed a number the data does not support.
    """
    assert_sorted(df)
    keys = _entity_columns(entity)
    missing = [c for c in (*keys, value) if c not in df.columns]
    if missing:
        raise PriorError(f"prior_stat: missing columns {missing}")

    work = df[keys].copy()
    work["_ordinal"] = race_ordinal(df)
    work["_value"] = pd.to_numeric(df[value], errors="coerce")

    # 1. Collapse to one row per (entity, race). This is the teammate guard.
    per_race = (
        work.groupby([*keys, "_ordinal"], dropna=False, sort=True)["_value"]
        .agg(within_race)
        .reset_index()
        .sort_values([*keys, "_ordinal"], kind="stable")
    )

    # 2. Shift by one race, then aggregate. Shifting before aggregating is
    #    what makes the exclusion of the current race structural rather than
    #    a property of the window length.
    grouped = per_race.groupby(keys, dropna=False, sort=False)["_value"]
    shifted = grouped.shift(1)
    roller = shifted.groupby([per_race[k] for k in keys], dropna=False, sort=False)
    if window is None:
        per_race["_prior"] = roller.transform(
            lambda s: s.expanding(min_periods=min_periods).agg(how)
        )
    else:
        per_race["_prior"] = roller.transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).agg(how)
        )

    # 3. Map back onto rows.
    lookup = per_race.set_index([*keys, "_ordinal"])["_prior"]
    row_key = pd.MultiIndex.from_frame(work[[*keys, "_ordinal"]])
    return pd.Series(
        lookup.reindex(row_key).to_numpy(dtype="float64"),
        index=df.index,
        dtype="float64",
    )


def prior_rate(
    df: pd.DataFrame,
    *,
    entity: str | Sequence[str],
    mask: pd.Series,
    window: int | None = None,
    within_race: Aggregation = "mean",
) -> pd.Series:
    """The share of an entity's strictly-prior races satisfying `mask`.

    `mask` is a boolean Series aligned to `df` — win, podium, finished — and
    the rate is its prior mean. Expressed this way so that "win rate" and
    "reliability" are the same computation with a different predicate, rather
    than two hand-rolled loops that can disagree about what "prior" means.
    """
    if not mask.index.equals(df.index):
        raise PriorError("mask must be aligned to the frame's index")
    work = df.copy()
    work["_mask"] = mask.astype("float64")
    return prior_stat(
        work,
        entity=entity,
        value="_mask",
        how="mean",
        window=window,
        within_race=within_race,
    )


def prior_race_count(
    df: pd.DataFrame, *, entity: str | Sequence[str]
) -> pd.Series:
    """How many races the entity has already contested, before this one.

    Zero identifies FR-212's cold start exactly — a rookie, a new team, a
    returning entrant's first race back — without needing a separate roster.
    """
    assert_sorted(df)
    keys = _entity_columns(entity)
    work = df[keys].copy()
    work["_ordinal"] = race_ordinal(df)
    work["_one"] = 1.0
    per_race = (
        work.groupby([*keys, "_ordinal"], dropna=False, sort=True)["_one"]
        .max()
        .reset_index()
        .sort_values([*keys, "_ordinal"], kind="stable")
    )
    per_race["_prior"] = (
        per_race.groupby(keys, dropna=False, sort=False).cumcount().astype("float64")
    )
    lookup = per_race.set_index([*keys, "_ordinal"])["_prior"]
    row_key = pd.MultiIndex.from_frame(work[[*keys, "_ordinal"]])
    return pd.Series(
        lookup.reindex(row_key).to_numpy(dtype="float64"),
        index=df.index,
        dtype="float64",
    )


def within_race_percentile(
    df: pd.DataFrame, values: pd.Series, *, higher_is_better: bool
) -> pd.Series:
    """Rank `values` against the contemporaneous field, 1.0 = best (FR-209).

    This is the function that makes FR-209 satisfiable. An absolute pace number
    means something different in 2013 and in 2023; a percentile against the
    twenty cars that actually turned up means the same thing in both.

    Rows whose value is NaN stay NaN — a driver with no history is not placed
    last, which would assert evidence of being slow rather than absence of
    evidence.
    """
    if not values.index.equals(df.index):
        raise PriorError("values must be aligned to the frame's index")
    ranked = values.groupby(
        [df[k] for k in RACE_KEYS], dropna=False, sort=False
    ).rank(pct=True, ascending=higher_is_better)
    return ranked.astype("float64")


def within_season_zscore(
    df: pd.DataFrame, values: pd.Series, *, season_col: str = "season"
) -> pd.Series:
    """Standardise `values` within each season (FR-209's third admissible form).

    A season with no spread yields 0.0 rather than infinity: identical values
    carry no ranking information, and that is a zero signal, not an undefined one.
    """
    if not values.index.equals(df.index):
        raise PriorError("values must be aligned to the frame's index")
    grouped = values.groupby(df[season_col], dropna=False, sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return ((values - mean) / std.replace(0.0, np.nan)).fillna(0.0).astype("float64")


__all__ = [
    "Aggregation",
    "PriorError",
    "RACE_KEYS",
    "SORT_KEYS",
    "assert_sorted",
    "prior_race_count",
    "prior_rate",
    "prior_stat",
    "race_ordinal",
    "sort_canonically",
    "within_race_percentile",
    "within_season_zscore",
]
