"""T-603: the grid CSV schema, and every violation reported at once (FR-803).

The multi-error fixture is not invented. `tests/fixtures/2025_*.csv` are v0's
own grid files, kept from P0 for exactly this purpose: they carry the wrong
identifier vocabulary *and* the observed-weather columns that were v0's
leakage defect, so one of them exercises the all-at-once path against a file
somebody really wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from f1pred.data.grid_csv import (
    REQUIRED_COLUMNS,
    SCHEMA_DOCUMENTATION,
    GridCsvError,
    read_grid_csv,
    validate,
)

FIXTURES = Path(__file__).parent / "fixtures"

_HEADER = list(REQUIRED_COLUMNS)


def _row(**overrides: object) -> dict[str, str]:
    row = {
        "season": "2026",
        "round": "1",
        "race_date": "2026-03-08",
        "circuit_id": "albert_park",
        "driver_id": "max_verstappen",
        "constructor_id": "red_bull",
        "grid_position": "1",
    }
    row.update({k: str(v) for k, v in overrides.items()})
    return row


def _grid(n: int = 3) -> list[dict[str, str]]:
    drivers = ["max_verstappen", "norris", "leclerc", "russell", "piastri"]
    return [
        _row(driver_id=drivers[i], constructor_id=f"team_{i}", grid_position=i + 1)
        for i in range(n)
    ]


class TestValidGrid:
    def test_a_well_formed_grid_validates(self) -> None:
        entries = validate(_grid(3), _HEADER)
        assert [e.grid_position for e in entries] == [1, 2, 3]

    def test_entries_come_back_in_grid_order_whatever_order_the_file_used(self) -> None:
        rows = list(reversed(_grid(3)))
        entries = validate(rows, _HEADER)
        assert [e.grid_position for e in entries] == [1, 2, 3]
        assert entries[0].driver_id == "max_verstappen"

    def test_columns_may_appear_in_any_order(self) -> None:
        # A spreadsheet reorders columns; that is not a data defect.
        assert validate(_grid(2), list(reversed(_HEADER)))


class TestEveryViolationAtOnce:
    """FR-803's actual requirement, which is about the *count* of errors."""

    def test_a_file_with_four_separate_faults_reports_all_four(self) -> None:
        rows = _grid(3)
        rows[0]["season"] = "twenty twenty six"  # not a number
        rows[1]["circuit_id"] = "Monza Circuit"  # not a canonical id
        rows[2]["grid_position"] = "7"  # leaves a gap at 3
        rows.append(_row(driver_id="norris", grid_position="4"))  # duplicate driver

        with pytest.raises(GridCsvError) as caught:
            validate(rows, _HEADER)

        messages = [v.render() for v in caught.value.violations]
        assert len(messages) >= 4, messages
        joined = "\n".join(messages)
        assert "twenty twenty six" in joined
        assert "Monza Circuit" in joined
        assert "no gaps" in joined
        assert "already appears" in joined

    def test_the_v0_fixture_reports_every_one_of_its_faults(self) -> None:
        with pytest.raises(GridCsvError) as caught:
            read_grid_csv(FIXTURES / "2025_bahrain_starting_grid.csv")

        messages = [v.render() for v in caught.value.violations]
        # Five required columns missing, three refused weather columns, and
        # seven columns that are not part of the schema.
        assert len(messages) >= 12, messages
        for missing in ("season", "race_date", "circuit_id", "driver_id", "constructor_id"):
            assert any(f"required column '{missing}' is missing" in m for m in messages)

    def test_every_v0_fixture_is_rejected(self) -> None:
        # All six, not just the one the multi-error test reads: a schema that
        # happened to accept five of them would be a schema nobody had checked.
        for path in sorted(FIXTURES.glob("2025_*_starting_grid.csv")):
            with pytest.raises(GridCsvError):
                read_grid_csv(path)

    def test_the_error_message_numbers_the_problems(self) -> None:
        with pytest.raises(GridCsvError) as caught:
            read_grid_csv(FIXTURES / "2025_monza_starting_grid.csv")
        text = str(caught.value)
        assert "problems in" in text
        assert "\n  1. " in text and "\n  2. " in text


class TestRefusedColumns:
    """The owner's decision, 2026-09-05: refuse, do not ignore."""

    def test_observed_weather_is_refused_with_its_reason(self) -> None:
        with pytest.raises(GridCsvError) as caught:
            validate(_grid(2), [*_HEADER, "weather_mixed"])
        text = str(caught.value)
        assert "weather_mixed" in text
        assert "not knowable before the race" in text
        assert "climatological prior" in text  # what the user should use instead

    @pytest.mark.parametrize(
        "column", ["weather_dry", "weather_wet", "rainfall", "air_temp", "track_temp"]
    )
    def test_every_observed_weather_column_is_refused(self, column: str) -> None:
        with pytest.raises(GridCsvError, match="not knowable before the race"):
            validate(_grid(2), [*_HEADER, column])

    @pytest.mark.parametrize(
        "column", ["finish_position", "classified_position", "status", "points", "laps_completed"]
    )
    def test_outcome_columns_are_refused_too(self, column: str) -> None:
        # Weather was v0's leak; the outcome itself would be a larger one.
        with pytest.raises(GridCsvError, match="not knowable before the race"):
            validate(_grid(2), [*_HEADER, column])

    def test_an_unknown_column_is_flagged_but_named_differently(self) -> None:
        with pytest.raises(GridCsvError) as caught:
            validate(_grid(2), [*_HEADER, "tyre_choice"])
        text = str(caught.value)
        assert "not part of the documented schema" in text
        assert "not knowable before the race" not in text


class TestIdentifiers:
    def test_a_three_letter_code_is_refused_with_an_explanation(self) -> None:
        rows = _grid(2)
        rows[0]["driver_id"] = "VER"
        with pytest.raises(GridCsvError, match="canonical lower-case id"):
            validate(rows, _HEADER)

    def test_a_display_name_is_refused(self) -> None:
        rows = _grid(2)
        rows[0]["constructor_id"] = "Red Bull Racing"
        with pytest.raises(GridCsvError, match="canonical lower-case id"):
            validate(rows, _HEADER)

    def test_an_undeclared_circuit_suggests_a_near_match(self) -> None:
        rows = _grid(2)
        for row in rows:
            row["circuit_id"] = "albert_par"
        with pytest.raises(GridCsvError) as caught:
            validate(rows, _HEADER)
        assert "did you mean 'albert_park'" in str(caught.value)


class TestGridOrder:
    def test_two_cars_cannot_share_a_slot(self) -> None:
        rows = _grid(3)
        rows[2]["grid_position"] = "2"
        with pytest.raises(GridCsvError, match="already taken"):
            validate(rows, _HEADER)

    def test_a_gap_in_the_grid_is_refused(self) -> None:
        rows = _grid(3)
        rows[2]["grid_position"] = "9"
        with pytest.raises(GridCsvError, match="no gaps"):
            validate(rows, _HEADER)

    def test_pole_is_one_not_zero(self) -> None:
        rows = _grid(3)
        rows[0]["grid_position"] = "0"
        with pytest.raises(GridCsvError, match="must be 1 or greater"):
            validate(rows, _HEADER)


class TestOneRacePerFile:
    def test_two_rounds_in_one_file_are_refused(self) -> None:
        rows = _grid(3)
        rows[2]["round"] = "2"
        with pytest.raises(GridCsvError, match="more than one value"):
            validate(rows, _HEADER)

    def test_an_empty_file_is_refused(self) -> None:
        with pytest.raises(GridCsvError, match="no rows"):
            validate([], _HEADER)


class TestReading:
    def test_a_missing_file_names_the_schema(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="grid_position"):
            read_grid_csv(tmp_path / "absent.csv")

    def test_a_valid_file_reads_into_a_frame_in_grid_order(self, tmp_path: Path) -> None:
        path = tmp_path / "grid.csv"
        path.write_text(
            "season,round,race_date,circuit_id,driver_id,constructor_id,grid_position\n"
            "2026,1,2026-03-08,albert_park,norris,mclaren,2\n"
            "2026,1,2026-03-08,albert_park,max_verstappen,red_bull,1\n"
        )
        frame = read_grid_csv(path)
        assert list(frame["grid_position"]) == [1, 2]
        assert list(frame["driver_id"]) == ["max_verstappen", "norris"]
        assert str(frame["race_date"].dtype) == "datetime64[ns]"

    def test_the_error_names_the_file_it_came_from(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.csv"
        path.write_text("season,round\n2026,1\n")
        with pytest.raises(GridCsvError) as caught:
            read_grid_csv(path)
        assert str(path) in str(caught.value)


class TestDocumentation:
    """FR-803 says *documented*, so the documentation is checked, not assumed."""

    def test_every_required_column_appears_in_the_documentation(self) -> None:
        for column in REQUIRED_COLUMNS:
            assert column in SCHEMA_DOCUMENTATION

    def test_the_documented_example_actually_validates(self) -> None:
        # A documented example that the validator rejects is worse than none.
        lines = [
            line.strip()
            for line in SCHEMA_DOCUMENTATION.splitlines()
            if line.strip().startswith(("season,", "2026,"))
        ]
        assert len(lines) >= 3
        header = lines[0].split(",")
        rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
        assert validate(rows, header)
