"""Assembling a feature matrix from the registry (FR-202, FR-203, FR-208, FR-212, FR-216).

Nothing here names a feature. The set to build is *derived* — from the
registry, filtered by availability and group — so the primary matrix cannot
drift out of sync with the declarations the leakage checks read. That is the
whole of FR-202, and it is why `build_primary()` takes no feature list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from f1pred.config import (
    COLD_START_CONSTRUCTOR_PERCENTILE,
    COLD_START_DRIVER_PERCENTILE,
)
from f1pred.data.target import build_target
from f1pred.features import definitions as _definitions  # noqa: F401  (populates REGISTRY)
from f1pred.features.priors import RACE_KEYS, prior_race_count, sort_canonically
from f1pred.features.registry import (
    REGISTRY,
    Availability,
    Feature,
    FeatureGroup,
    Registry,
)

#: Carried alongside the matrix so a prediction can be traced to a row.
KEY_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "round",
    "race_date",
    "circuit_id",
    "driver_id",
    "constructor_id",
)

TARGET_COLUMN: Final[str] = "target_position"


class BuildError(ValueError):
    """Raised when a feature set cannot be assembled."""


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A built feature set plus everything needed to interpret a row.

    `flags` is a separate frame rather than extra columns in `X` because the
    flags are reporting obligations (FR-212, FR-216) and not model inputs. If
    they lived in `X` they would be trained on, which would teach the model to
    predict from the fact that we were unsure.
    """

    set_name: str
    X: pd.DataFrame
    keys: pd.DataFrame
    flags: pd.DataFrame
    target: pd.Series | None
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.X)


# --------------------------------------------------------------------------
# Dependency resolution
# --------------------------------------------------------------------------


def resolve_order(
    wanted: tuple[Feature, ...], *, registry: Registry, available: frozenset[str]
) -> tuple[Feature, ...]:
    """Order `wanted` and their transitive feature dependencies for computation.

    Names in `depends_on` resolve first against `available` — the raw schema
    columns — and then against the registry. Anything that resolves to neither
    is an error: v0 tolerated a missing feature by dropping it from the list at
    run time, which is how a model quietly trained on fewer features than the
    experiment it was reported as.
    """
    ordered: list[Feature] = []
    seen: set[str] = set()
    visiting: list[str] = []

    def visit(f: Feature) -> None:
        if f.name in seen:
            return
        if f.name in visiting:
            cycle = " -> ".join([*visiting[visiting.index(f.name) :], f.name])
            raise BuildError(f"circular feature dependency: {cycle}")
        visiting.append(f.name)
        for dep in f.depends_on:
            if dep in available:
                continue
            if dep not in registry:
                raise BuildError(
                    f"feature {f.name!r} depends on {dep!r}, which is neither a "
                    f"raw schema column nor a registered feature"
                )
            visit(registry.get(dep))
        visiting.pop()
        seen.add(f.name)
        ordered.append(f)

    for f in wanted:
        visit(f)
    return tuple(ordered)


# --------------------------------------------------------------------------
# Flags (FR-212, FR-216)
# --------------------------------------------------------------------------


def detect_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Every mid-corpus constructor change, one row each (FR-216).

    Returned rather than logged, because FR-216 requires transfers to be
    *recorded* and FR-511's experiment counts them. A driver's first appearance
    is not a transfer: they came from outside the corpus, which is FR-212's
    cold start, not a move.
    """
    ordered = df.sort_values(["race_date", "season", "round"], kind="stable")
    previous = ordered.groupby("driver_id", sort=False)["constructor_id"].shift(1)
    moved = previous.notna() & (ordered["constructor_id"] != previous)
    out = ordered.loc[moved, ["season", "round", "race_date", "driver_id", "constructor_id"]]
    return out.assign(previous_constructor_id=previous[moved]).reset_index(drop=True)


def _post_transfer_flag(df: pd.DataFrame) -> pd.Series:
    previous = df.groupby("driver_id", sort=False)["constructor_id"].shift(1)
    return (previous.notna() & (df["constructor_id"] != previous)).astype(bool)


def _apply_cold_start(
    matrix: pd.DataFrame,
    df: pd.DataFrame,
    features: tuple[Feature, ...],
) -> tuple[pd.DataFrame, pd.Series]:
    """FR-212: declared fallbacks for entrants with no history, and the flag.

    Only a genuine cold start is filled — zero prior races for the driver or
    the constructor. A veteran's first visit to a new circuit leaves its
    circuit-history features null on purpose: the gradient-boosted ranker
    learns a branch for missing values, which is a real answer, whereas a
    constant here would be an invented one.
    """
    # Computed here rather than read from the matrix: the flag must hold for
    # any feature set, including one that did not ask for the count features.
    driver_cold = prior_race_count(df, entity="driver_id") == 0
    constructor_cold = prior_race_count(df, entity="constructor_id") == 0
    cold = driver_cold | constructor_cold

    filled = matrix.copy()
    if cold.any():
        for f in features:
            if not f.summarises_history or f.name not in filled.columns:
                continue
            if f.never_impute:
                # FR-211: the null is the answer. See `Feature.never_impute`.
                continue
            fallback = (
                COLD_START_CONSTRUCTOR_PERCENTILE
                if f.name.startswith("constructor_")
                else COLD_START_DRIVER_PERCENTILE
            )
            rows = constructor_cold if f.name.startswith("constructor_") else driver_cold
            target_rows = rows & filled[f.name].isna()
            filled.loc[target_rows, f.name] = fallback

    return filled, cold


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


def build(
    df: pd.DataFrame,
    *,
    features: tuple[Feature, ...],
    set_name: str,
    registry: Registry = REGISTRY,
) -> FeatureMatrix:
    """Compute `features` over a collected race frame.

    Determinism (FR-208) rests on three things, all of them here: the frame is
    sorted by declared keys before anything is computed, features are evaluated
    in a dependency order derived from registration order, and the output
    columns are emitted in registration order. Nothing iterates a set.
    """
    if TARGET_COLUMN not in df.columns:
        df = build_target(df)
    work = sort_canonically(df)

    available = frozenset(work.columns)
    for f in resolve_order(features, registry=registry, available=available):
        series = f.fn(work)
        if not isinstance(series, pd.Series):
            raise BuildError(
                f"feature {f.name!r} returned {type(series).__name__}, not a Series"
            )
        if not series.index.equals(work.index):
            raise BuildError(
                f"feature {f.name!r} returned a Series not aligned to the frame"
            )
        work[f.name] = series

    names = tuple(f.name for f in features)
    matrix = work[list(names)].copy()
    matrix, cold = _apply_cold_start(matrix, work, features)

    flags = pd.DataFrame(
        {
            # FR-212: the prediction rests on declared fallbacks, not evidence.
            "low_information": cold.to_numpy(),
            # FR-216: constructor history was reset for this row.
            "post_transfer": _post_transfer_flag(work).to_numpy(),
        },
        index=work.index,
    )

    target = (
        pd.to_numeric(work[TARGET_COLUMN], errors="coerce")
        if TARGET_COLUMN in work.columns
        else None
    )

    return FeatureMatrix(
        set_name=set_name,
        X=matrix,
        keys=work[list(KEY_COLUMNS)].copy(),
        flags=flags,
        target=target,
        feature_names=names,
    )


def build_primary(df: pd.DataFrame, *, registry: Registry = REGISTRY) -> FeatureMatrix:
    """FR-202: the primary matrix. Takes no feature list, by design."""
    return build(df, features=registry.primary(), set_name="primary", registry=registry)


def build_oracle(df: pd.DataFrame, *, registry: Registry = REGISTRY) -> FeatureMatrix:
    """FR-207: primary plus post-race features, for upper-bound analysis only.

    Never the default path, and labelled in the matrix it returns so a results
    file cannot report an oracle run as an honest one.
    """
    return build(df, features=registry.oracle(), set_name="oracle", registry=registry)


def build_group(
    df: pd.DataFrame, group: FeatureGroup, *, registry: Registry = REGISTRY
) -> FeatureMatrix:
    """FR-213/FR-215: the primary set plus one named group, as a separate arm."""
    features = tuple(registry.primary()) + tuple(
        f
        for f in registry.group_features(group)
        if f.availability is Availability.PRE_RACE
    )
    return build(
        df, features=features, set_name=f"primary+{group.value}", registry=registry
    )


__all__ = [
    "BuildError",
    "FeatureMatrix",
    "KEY_COLUMNS",
    "TARGET_COLUMN",
    "build",
    "build_group",
    "build_oracle",
    "build_primary",
    "detect_transfers",
    "resolve_order",
]
