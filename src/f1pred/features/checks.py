"""The leakage firewall (FR-204, FR-205, FR-208, FR-209). Plan D-2.

Two layers, because neither alone is sufficient.

**Static** reads declarations. It fails if a post-race feature reaches the
primary matrix, or if a pre-race feature depends — at any depth — on a
post-race one. It is fast, total, and blind to what the code actually does.

**Behavioural** reads behaviour. It perturbs the outcome of a race and asserts
that not one pre-race feature value for that race moves. This is the layer that
catches an off-by-one in a rolling window, a `shift(1)` applied to driver rows
inside a constructor group, or a percentile computed after the fact — none of
which a declaration can reveal.

Both are exercised against deliberately leaky features in the test suite. A
leakage check that has never failed is not known to work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from f1pred.data.schema import NON_STARTER_OUTCOMES, ClassificationOutcome
from f1pred.features.build import TARGET_COLUMN, build
from f1pred.features.priors import RACE_KEYS, sort_canonically
from f1pred.features.registry import (
    REGISTRY,
    Availability,
    Feature,
    FeatureGroup,
    Registry,
)

#: Columns whose value is not known until the race has been run. A pre-race
#: feature may read these — every rolling form feature does — but only for
#: races strictly preceding the one it describes, which is what the
#: behavioural check verifies. Declared here so the perturbation knows exactly
#: what to scramble.
POST_RACE_COLUMNS: Final[tuple[str, ...]] = (
    "classified_position",
    "position_text",
    "classification_outcome",
    "status_raw",
    "laps_completed",
    "points",
    "final_lap_position",
    TARGET_COLUMN,
    # FR-206: observed race weather. Listed here so the behavioural check
    # actually perturbs it — without this a feature reading the race day's
    # rainfall while declaring itself pre-race would pass both layers, which
    # is the exact defect that sat in v0 for four months.
    "weather_actual_precip_mm",
    "weather_actual_temp_max_c",
    "weather_actual_wind_kmh",
)


class LeakageError(AssertionError):
    """Raised when a feature set fails a leakage check."""


@dataclass(slots=True)
class LeakageReport:
    """What a check found. Empty `violations` means the check passed."""

    check: str
    features_checked: int = 0
    races_checked: int = 0
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def raise_if_failed(self) -> LeakageReport:
        if not self.passed:
            raise LeakageError(
                f"{self.check} failed with {len(self.violations)} violation(s):\n  "
                + "\n  ".join(self.violations)
            )
        return self


# --------------------------------------------------------------------------
# Static (FR-204, FR-209)
# --------------------------------------------------------------------------


def _transitive_post_race(
    f: Feature, registry: Registry, *, _seen: frozenset[str] = frozenset()
) -> list[str]:
    """Names of post-race features `f` reaches through its dependencies."""
    found: list[str] = []
    for dep in f.depends_on:
        if dep in _seen or dep not in registry:
            continue
        upstream = registry.get(dep)
        if upstream.availability is Availability.POST_RACE:
            found.append(upstream.name)
        found.extend(
            _transitive_post_race(upstream, registry, _seen=_seen | {dep})
        )
    return found


def static_leakage_check(
    features: Sequence[Feature] | None = None, *, registry: Registry = REGISTRY
) -> LeakageReport:
    """FR-204 and FR-209, in one pass over the declarations.

    Defaults to the derived primary set. Takes an explicit list so a test can
    hand it a set that *should* fail — which is the only way to know the check
    is wired up at all.
    """
    checked = tuple(registry.primary() if features is None else features)
    report = LeakageReport(check="static leakage check", features_checked=len(checked))

    for f in checked:
        if f.availability is Availability.POST_RACE:
            report.violations.append(
                f"{f.name}: declared POST_RACE but present in the primary set "
                f"(FR-204). Post-race features belong to the oracle set only."
            )
        if f.group is not FeatureGroup.PRIMARY:
            report.violations.append(
                f"{f.name}: belongs to group {f.group.value!r}, which FR-213/"
                f"FR-215 keep out of the primary set."
            )
        upstream = _transitive_post_race(f, registry)
        if upstream:
            report.violations.append(
                f"{f.name}: declared {f.availability.value} but depends on "
                f"post-race feature(s) {sorted(set(upstream))} (FR-204). A "
                f"pre-race declaration does not survive a post-race input."
            )
        # FR-209 / T-211, scanned in the same pass as the leakage fields.
        if f.summarises_history and not f.relative and not (
            f.relative_justification or ""
        ).strip():
            report.violations.append(
                f"{f.name}: summarises historical performance but is neither "
                f"relative nor justified (FR-209)."
            )
    return report


# --------------------------------------------------------------------------
# Behavioural (FR-205)
# --------------------------------------------------------------------------


def perturb_race(
    df: pd.DataFrame, season: int, rnd: int, *, seed: int
) -> pd.DataFrame:
    """Rewrite one race's outcome, leaving every other race untouched.

    Two kinds of post-race column need two different treatments, and getting
    this wrong makes the whole check pass for the wrong reason.

    **Driver-level** columns — finishing position, laps, status — vary within a
    race, so they are *permuted* among the race's own starters. A permutation
    keeps the race internally consistent: the same set of results, handed to
    different cars.

    **Race-level** columns — the weather, above all — hold one value shared by
    every car in the race. Permuting those among rows is a no-op, so a feature
    reading race-day rainfall while declaring itself pre-race would sail
    through. They are instead *substituted* with the value from a different
    race. This was found the way such things usually are: the negative control
    for weather passed when it should have failed.

    Race-level columns are detected rather than listed, so a post-race column
    added later cannot quietly miss out on being perturbed.
    """
    out = df.copy()
    mask = (out["season"] == season) & (out["round"] == rnd)
    if "classification_outcome" in out.columns:
        # Non-starters are excluded from the corpus by FR-002 before features
        # are built, so moving a withdrawal to a different car would change
        # *which* cars exist rather than how they finished. That is a different
        # experiment, and it makes the comparison impossible.
        mask &= out["classification_outcome"].isin(
            [o.value for o in ClassificationOutcome if o not in NON_STARTER_OUTCOMES]
        )
    n = int(mask.sum())
    if n < 2:
        raise ValueError(
            f"race {season} round {rnd} has {n} permutable rows; nothing to permute"
        )

    present = [c for c in POST_RACE_COLUMNS if c in out.columns]
    in_race = out.loc[mask]
    driver_level = [c for c in present if in_race[c].nunique(dropna=False) > 1]
    race_level = [
        c for c in present if c not in driver_level and c not in IDENTITY_COLUMNS
    ]

    rng = np.random.default_rng(seed)
    if driver_level:
        order = rng.permutation(n)
        while np.array_equal(order, np.arange(n)):  # a no-op proves nothing
            order = rng.permutation(n)
        block = out.loc[mask, driver_level].to_numpy(dtype=object)[order]
        out.loc[mask, driver_level] = block

    for column in race_level:
        substitute = _substituted_value(out, mask, column, rng)
        if substitute is _UNCHANGED:
            continue
        out.loc[mask, column] = substitute

    return out.astype(df.dtypes.to_dict())


#: Post-race columns that decide whether a row exists at all. They are never
#: substituted wholesale, only permuted. In a race where every starter was
#: classified, `classification_outcome` is constant and so looks race-level —
#: substituting it with "withdrawn" from another race deletes the entire field
#: under FR-002, and the check then reports that it cannot compare the race
#: rather than that a feature leaked.
IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "classification_outcome",
    "position_text",
)


#: Sentinel: this column cannot meaningfully be perturbed, so leave it alone.
_UNCHANGED: Final[object] = object()


def _substituted_value(
    df: pd.DataFrame, mask: pd.Series, column: str, rng: np.random.Generator
) -> object:
    """A value for `column` that differs from the one this race actually had.

    Taken from another race where possible, so the substitute stays inside the
    range the data really contains — a rainfall figure that never occurred
    anywhere would test the feature against an input it will never see.

    Returns `_UNCHANGED` for a column that is null here and null everywhere
    else. Such a column holds nothing to leak, and inventing a value for it
    would only risk breaking its dtype: `final_lap_position` is a nullable
    integer that is empty for most races by design (FR-001b).
    """
    current = df.loc[mask, column].iloc[0]
    elsewhere = df.loc[~mask, column].dropna()
    candidates = elsewhere[elsewhere != current] if not elsewhere.empty else elsewhere
    if not candidates.empty:
        return candidates.iloc[int(rng.integers(len(candidates)))]
    if pd.isna(current):
        return _UNCHANGED
    if pd.api.types.is_float_dtype(df[column]):
        return current + 7.5
    if pd.api.types.is_integer_dtype(df[column]):
        return current + 7
    return _UNCHANGED


def sample_races(df: pd.DataFrame, *, count: int, seed: int) -> list[tuple[int, int]]:
    """A deterministic spread of races across the corpus.

    Spread rather than sampled at random: leakage that only shows up in a
    driver's first season, or only once a rolling window has filled, is exactly
    the kind a clustered sample misses.
    """
    races = (
        df[["season", "round", "race_date"]]
        .drop_duplicates()
        .sort_values(["race_date", "season", "round"], kind="stable")
        .reset_index(drop=True)
    )
    # Skip the first few races: nothing precedes them, so nothing can leak
    # backwards into them and they would pass vacuously.
    eligible = races.iloc[min(5, len(races) - 1) :]
    if eligible.empty:
        return []
    picks = np.linspace(0, len(eligible) - 1, num=min(count, len(eligible)))
    idx = sorted({int(round(p)) for p in picks})
    return [
        (int(eligible.iloc[i]["season"]), int(eligible.iloc[i]["round"])) for i in idx
    ]


def behavioural_leakage_check(
    df: pd.DataFrame,
    *,
    features: Sequence[Feature] | None = None,
    registry: Registry = REGISTRY,
    races: Sequence[tuple[int, int]] | None = None,
    count: int = 8,
    seed: int = 20260830,
    tolerance: float = 0.0,
) -> LeakageReport:
    """FR-205: perturb a race's result; no pre-race feature for it may move.

    Only the perturbed race's own rows are compared. Later races *should*
    change — their rolling features legitimately read the race we just
    rewrote — and asserting otherwise would be asserting that history has no
    effect on the future.
    """
    checked = tuple(registry.primary() if features is None else features)
    targets = list(races) if races is not None else sample_races(df, count=count, seed=seed)
    report = LeakageReport(
        check="behavioural leakage check",
        features_checked=len(checked),
        races_checked=len(targets),
    )

    baseline = build(df, features=checked, set_name="behavioural-baseline", registry=registry)
    base_keys = baseline.keys.reset_index(drop=True)
    base_X = baseline.X.reset_index(drop=True)

    pre_race = [f.name for f in checked if f.availability is Availability.PRE_RACE]

    for season, rnd in targets:
        perturbed_df = perturb_race(df, season, rnd, seed=seed + season * 100 + rnd)
        perturbed = build(
            perturbed_df, features=checked, set_name="behavioural-perturbed", registry=registry
        )
        p_keys = perturbed.keys.reset_index(drop=True)
        p_X = perturbed.X.reset_index(drop=True)

        b_mask = (base_keys["season"] == season) & (base_keys["round"] == rnd)
        p_mask = (p_keys["season"] == season) & (p_keys["round"] == rnd)

        # Align on driver, since the permutation can reorder rows within a race.
        b = (
            base_X.loc[b_mask.to_numpy()]
            .assign(_driver=base_keys.loc[b_mask.to_numpy(), "driver_id"].to_numpy())
            .set_index("_driver")
            .sort_index()
        )
        p = (
            p_X.loc[p_mask.to_numpy()]
            .assign(_driver=p_keys.loc[p_mask.to_numpy(), "driver_id"].to_numpy())
            .set_index("_driver")
            .sort_index()
        )
        if not b.index.equals(p.index):
            report.violations.append(
                f"{season} R{rnd}: perturbation changed which drivers are present; "
                f"the check cannot compare them"
            )
            continue

        for name in pre_race:
            left, right = b[name], p[name]
            moved = (left.isna() != right.isna()) | (
                (left - right).abs().fillna(0.0) > tolerance
            )
            if moved.any():
                who = ", ".join(map(str, left.index[moved][:3]))
                report.violations.append(
                    f"{name}: value changed for {int(moved.sum())} car(s) in "
                    f"{season} R{rnd} ({who}) when only that race's result was "
                    f"altered — the feature reads the race it is predicting (FR-205)."
                )
    return report


# --------------------------------------------------------------------------
# Determinism (FR-208)
# --------------------------------------------------------------------------


def matrix_fingerprint(matrix: pd.DataFrame) -> str:
    """A content hash covering values, column order, and dtypes."""
    import hashlib

    h = hashlib.sha256()
    for column in matrix.columns:
        h.update(column.encode())
        h.update(str(matrix[column].dtype).encode())
        h.update(pd.util.hash_pandas_object(matrix[column], index=True).values.tobytes())
    return h.hexdigest()


def determinism_check(
    df: pd.DataFrame,
    *,
    features: Sequence[Feature] | None = None,
    registry: Registry = REGISTRY,
) -> LeakageReport:
    """FR-208: two builds of the same data must be byte-identical.

    The second build is fed a row-shuffled copy of the input. Identical output
    from identically-ordered input would only prove the code is not random; it
    would not prove the canonical sort is doing its job.
    """
    checked = tuple(registry.primary() if features is None else features)
    report = LeakageReport(check="determinism check", features_checked=len(checked))

    first = build(df, features=checked, set_name="determinism-a", registry=registry)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = build(shuffled, features=checked, set_name="determinism-b", registry=registry)

    if first.feature_names != second.feature_names:
        report.violations.append(
            f"column order differs: {first.feature_names} vs {second.feature_names}"
        )
    a, b = matrix_fingerprint(first.X), matrix_fingerprint(second.X)
    if a != b:
        report.violations.append(
            f"feature matrix differs between builds of the same data ({a[:12]} "
            f"vs {b[:12]}); FR-208 requires byte-identical output"
        )
    return report


def run_all(
    df: pd.DataFrame, *, registry: Registry = REGISTRY, races: int = 8
) -> tuple[LeakageReport, ...]:
    """Every automated check, in one call, for a phase gate or a CLI command."""
    return (
        static_leakage_check(registry=registry),
        behavioural_leakage_check(df, registry=registry, count=races),
        determinism_check(df, registry=registry),
    )


__all__ = [
    "LeakageError",
    "LeakageReport",
    "IDENTITY_COLUMNS",
    "POST_RACE_COLUMNS",
    "behavioural_leakage_check",
    "determinism_check",
    "matrix_fingerprint",
    "perturb_race",
    "run_all",
    "sample_races",
    "static_leakage_check",
]
