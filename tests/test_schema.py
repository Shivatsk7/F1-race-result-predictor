"""T-101 — canonical row schema and validator (FR-002, FR-003, FR-105, FR-108)."""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.data.schema import (
    NON_NULL_COLUMNS,
    PRIMARY_KEY,
    RACE_ROW_SCHEMA,
    SchemaError,
    WeekendFormat,
    empty_frame,
    map_weekend_format,
    validate_frame,
)
from tests.fixtures.races import australia_2014


class TestRequiredFields:
    def test_carries_the_fields_the_spec_names(self):
        # FR-003 keeps these separate so an alternative target needs no recollection.
        for col in (
            "status_raw",
            "classification_outcome",
            "laps_completed",
            "classified_position",
            "weekend_format",
        ):
            assert col in RACE_ROW_SCHEMA

    def test_chronology_fields_present(self):
        # FR-105: sufficient to establish strict chronological ordering.
        for col in ("season", "round", "race_date"):
            assert col in RACE_ROW_SCHEMA

    def test_valid_frame_passes(self):
        out = validate_frame(australia_2014())
        assert list(out.columns) == list(RACE_ROW_SCHEMA)

    def test_empty_frame_is_valid(self):
        assert validate_frame(empty_frame()).empty


class TestRejection:
    def test_missing_column_rejected(self):
        df = australia_2014().drop(columns=["laps_completed"])
        with pytest.raises(SchemaError, match="missing required columns"):
            validate_frame(df)

    def test_undeclared_column_rejected(self):
        df = australia_2014().assign(mystery=1)
        with pytest.raises(SchemaError, match="unexpected columns"):
            validate_frame(df)

    def test_wrong_dtype_rejected(self):
        df = australia_2014()
        df["laps_completed"] = df["laps_completed"].astype("float64")
        with pytest.raises(SchemaError, match="wrong dtypes"):
            validate_frame(df)

    def test_null_in_non_nullable_column_rejected(self):
        df = australia_2014()
        df["circuit_id"] = df["circuit_id"].astype("object")
        df.loc[df.index[0], "circuit_id"] = None
        df["circuit_id"] = df["circuit_id"].astype("string")
        with pytest.raises(SchemaError, match="nulls in non-nullable"):
            validate_frame(df)

    def test_duplicate_primary_key_rejected(self):
        df = pd.concat([australia_2014(), australia_2014().head(1)], ignore_index=True)
        with pytest.raises(SchemaError, match="duplicate primary keys"):
            validate_frame(df)

    def test_unknown_classification_outcome_rejected(self):
        df = australia_2014()
        df.loc[df.index[0], "classification_outcome"] = "invented"
        with pytest.raises(SchemaError, match="unknown classification_outcome"):
            validate_frame(df)

    def test_unknown_weekend_format_rejected(self):
        df = australia_2014()
        df.loc[df.index[0], "weekend_format"] = "grand_prix_but_backwards"
        with pytest.raises(SchemaError, match="unknown weekend_format"):
            validate_frame(df)

    def test_driver_code_may_be_null(self):
        assert "driver_code" not in NON_NULL_COLUMNS
        df = australia_2014()
        df.loc[df.index[0], "driver_code"] = pd.NA
        validate_frame(df)


class TestWeekendFormatMapping:
    """FR-108 — the upstream vocabulary is larger than our enum, and has moved."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("conventional", WeekendFormat.CONVENTIONAL),
            ("sprint", WeekendFormat.SPRINT_SETS_GRID),
            ("sprint_shootout", WeekendFormat.SPRINT_STANDALONE),
            ("sprint_qualifying", WeekendFormat.SPRINT_STANDALONE),
        ],
    )
    def test_known_values(self, raw, expected):
        assert map_weekend_format(raw) is expected

    def test_2021_and_2023_sprints_are_not_the_same_format(self):
        # In 2021-22 the sprint set the GP grid; from 2023 it did not.
        assert map_weekend_format("sprint") is not map_weekend_format("sprint_shootout")

    def test_unknown_format_raises_rather_than_defaulting(self):
        with pytest.raises(SchemaError, match="unrecognised weekend format"):
            map_weekend_format("sprint_something_new")

    def test_case_and_whitespace_tolerated(self):
        assert map_weekend_format("  Conventional ") is WeekendFormat.CONVENTIONAL


class TestPrimaryKey:
    def test_identifies_a_row_uniquely(self):
        df = australia_2014()
        assert not df.duplicated(subset=list(PRIMARY_KEY)).any()
