"""Canonical driver-race row schema and validation (FR-002, FR-003, FR-105, FR-108).

The T-099 spike showed that the upstream vocabulary is richer than the spec's
first draft assumed: the raw status field carries ~30 distinct values per season,
and "did not finish" is not the same condition as "not classified". Both facts
are encoded here rather than being flattened at collection time.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import pandas as pd


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the canonical schema."""


# --------------------------------------------------------------------------
# Classification outcome (FR-003, FR-003a)
# --------------------------------------------------------------------------


class ClassificationOutcome(Enum):
    """What the *source* says happened to an entry, independent of its status.

    FR-003a: this is derived from the position sentinel, never from the status
    string. Bahrain 2022 is the case that forces the distinction — Perez and
    Verstappen both stopped with fuel-pressure failures and were still
    classified 18th and 19th.
    """

    CLASSIFIED = "classified"
    NOT_CLASSIFIED = "not_classified"
    DISQUALIFIED = "disqualified"
    WITHDRAWN = "withdrawn"


#: Ergast/jolpica `positionText` sentinels. A numeric value means classified;
#: the letters are the exceptional cases. Unknown letters raise rather than
#: defaulting, because a silent default here would misclassify real races.
_POSITION_SENTINELS: Final[dict[str, ClassificationOutcome]] = {
    "R": ClassificationOutcome.NOT_CLASSIFIED,  # retired
    "N": ClassificationOutcome.NOT_CLASSIFIED,  # not classified
    "D": ClassificationOutcome.DISQUALIFIED,
    "E": ClassificationOutcome.DISQUALIFIED,  # excluded
    "W": ClassificationOutcome.WITHDRAWN,  # withdrew — did not start
    "F": ClassificationOutcome.WITHDRAWN,  # failed to qualify — did not start
}

#: FR-002: entries that never took the start are excluded from the corpus.
NON_STARTER_OUTCOMES: Final[frozenset[ClassificationOutcome]] = frozenset(
    {ClassificationOutcome.WITHDRAWN}
)


def classify(position_text: str) -> ClassificationOutcome:
    """Map a source position sentinel to a classification outcome.

    FR-002 depends on this being the *only* way non-starts are identified. Zero
    laps completed cannot be used: a car that never started and a car that
    crashed on lap 1 both complete zero laps, and conflating them would delete
    real races from the corpus.
    """
    text = str(position_text).strip()
    if not text:
        raise SchemaError("empty positionText; cannot determine classification")
    if text.isdigit():
        return ClassificationOutcome.CLASSIFIED
    try:
        return _POSITION_SENTINELS[text.upper()]
    except KeyError:
        raise SchemaError(
            f"unrecognised positionText sentinel {text!r}. Add it to "
            f"_POSITION_SENTINELS deliberately rather than defaulting — a wrong "
            f"default here silently corrupts the FR-001 target."
        ) from None


# --------------------------------------------------------------------------
# Weekend format (FR-108)
# --------------------------------------------------------------------------


class WeekendFormat(Enum):
    CONVENTIONAL = "conventional"
    SPRINT_SETS_GRID = "sprint_sets_grid"
    SPRINT_STANDALONE = "sprint_standalone"


#: Declared mapping from the upstream vocabulary, which is larger than this enum
#: and has changed three times. Verified against fastf1 3.8.3 by the T-099 spike.
WEEKEND_FORMAT_MAP: Final[dict[str, WeekendFormat]] = {
    "conventional": WeekendFormat.CONVENTIONAL,
    # 2021-22: the sprint race set the Grand Prix grid.
    "sprint": WeekendFormat.SPRINT_SETS_GRID,
    # 2023: sprint shootout set the sprint grid; qualifying still set the GP grid.
    "sprint_shootout": WeekendFormat.SPRINT_STANDALONE,
    # 2024+: renamed again, same structure.
    "sprint_qualifying": WeekendFormat.SPRINT_STANDALONE,
}


def map_weekend_format(raw: str) -> WeekendFormat:
    """FR-108: an unrecognised raw value is an error, never a silent fallback.

    Defaulting to CONVENTIONAL would assert that no sprint happened, which for a
    sprint weekend is a factual error in the grid-formation model.
    """
    key = str(raw).strip().lower()
    try:
        return WEEKEND_FORMAT_MAP[key]
    except KeyError:
        raise SchemaError(
            f"unrecognised weekend format {raw!r}. Upstream has changed its "
            f"vocabulary before; add the new value to WEEKEND_FORMAT_MAP rather "
            f"than defaulting to CONVENTIONAL."
        ) from None


# --------------------------------------------------------------------------
# The canonical row
# --------------------------------------------------------------------------

#: column -> pandas dtype. Order is the canonical column order.
RACE_ROW_SCHEMA: Final[dict[str, str]] = {
    # Chronology (FR-105)
    "season": "int64",
    "round": "int64",
    "race_date": "datetime64[ns]",
    "circuit_id": "string",
    # Weekend structure (FR-108)
    "weekend_format": "string",
    "weekend_format_raw": "string",
    # Entry identity
    "driver_id": "string",
    "driver_code": "string",
    "constructor_id": "string",
    # Pre-race
    "grid_position": "int64",
    # Outcome (FR-001, FR-003)
    "classified_position": "int64",
    "position_text": "string",
    "classification_outcome": "string",
    "status_raw": "string",
    "laps_completed": "int64",
    "points": "float64",
    # FR-001b tie-break. Deliberately sparse: populated only for races where
    # laps completed left retirements tied, because collecting it costs roughly
    # eleven extra requests per race.
    "final_lap_position": "Int64",
}

#: Columns that may never be null. `driver_code` is exempt because some
#: historical entries genuinely have no three-letter code; `final_lap_position`
#: because it is fetched only where a tie needs breaking.
NON_NULL_COLUMNS: Final[frozenset[str]] = frozenset(
    set(RACE_ROW_SCHEMA) - {"driver_code", "points", "final_lap_position"}
)

#: FR-105: these together identify a row uniquely.
PRIMARY_KEY: Final[tuple[str, ...]] = ("season", "round", "driver_id")


def validate_frame(df: pd.DataFrame, *, strict_dtypes: bool = True) -> pd.DataFrame:
    """Validate a collected frame against the canonical schema.

    Returns the frame with columns in canonical order. Raises `SchemaError` on
    any missing column, wrong dtype, null in a non-null column, or duplicate
    primary key.
    """
    missing = [c for c in RACE_ROW_SCHEMA if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    extra = [c for c in df.columns if c not in RACE_ROW_SCHEMA]
    if extra:
        raise SchemaError(
            f"unexpected columns: {extra}. Extend RACE_ROW_SCHEMA deliberately "
            f"rather than letting undeclared fields ride along."
        )

    if strict_dtypes:
        wrong = {
            col: str(df[col].dtype)
            for col, expected in RACE_ROW_SCHEMA.items()
            if str(df[col].dtype) != expected
        }
        if wrong:
            raise SchemaError(
                f"wrong dtypes: {wrong}; expected "
                f"{ {c: RACE_ROW_SCHEMA[c] for c in wrong} }"
            )

    nulls = {
        col: int(df[col].isna().sum())
        for col in NON_NULL_COLUMNS
        if df[col].isna().any()
    }
    if nulls:
        raise SchemaError(f"nulls in non-nullable columns: {nulls}")

    if df.duplicated(subset=list(PRIMARY_KEY)).any():
        dupes = df[df.duplicated(subset=list(PRIMARY_KEY), keep=False)]
        raise SchemaError(
            f"duplicate primary keys {PRIMARY_KEY}: "
            f"{dupes[list(PRIMARY_KEY)].to_dict('records')[:5]}"
        )

    bad_outcome = set(df["classification_outcome"].dropna().unique()) - {
        o.value for o in ClassificationOutcome
    }
    if bad_outcome:
        raise SchemaError(f"unknown classification_outcome values: {bad_outcome}")

    bad_format = set(df["weekend_format"].dropna().unique()) - {
        f.value for f in WeekendFormat
    }
    if bad_format:
        raise SchemaError(f"unknown weekend_format values: {bad_format}")

    return df[list(RACE_ROW_SCHEMA)]


def empty_frame() -> pd.DataFrame:
    """A zero-row frame with correct columns and dtypes."""
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in RACE_ROW_SCHEMA.items()}
    )


__all__ = [
    "ClassificationOutcome",
    "NON_NULL_COLUMNS",
    "NON_STARTER_OUTCOMES",
    "PRIMARY_KEY",
    "RACE_ROW_SCHEMA",
    "SchemaError",
    "WEEKEND_FORMAT_MAP",
    "WeekendFormat",
    "classify",
    "empty_frame",
    "map_weekend_format",
    "validate_frame",
]
