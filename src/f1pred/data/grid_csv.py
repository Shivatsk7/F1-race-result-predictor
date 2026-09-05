"""The starting-grid CSV: its documented schema, and a validator that reports
every violation at once (FR-803).

A user hands this file to `f1pred predict` for a race that has not run. It
carries only what is knowable before the lights go out.

Schema
------

Seven required columns, in any order:

    season           int    the championship year, e.g. 2026
    round            int    the round number within that season, 1-based
    race_date        date   ISO 8601, YYYY-MM-DD
    circuit_id       str    a declared circuit id (see reference/circuits.py)
    driver_id        str    canonical id as used by the corpus, e.g. max_verstappen
    constructor_id   str    canonical id as used by the corpus, e.g. red_bull
    grid_position    int    1 for pole, contiguous through the field

Every row must share one season, round, race_date and circuit_id: the file
describes one race. Grid positions must run 1..N with no gap and no repeat,
because a grid with two P4s and no P5 is a transcription error, and the model's
strongest feature is grid position.

Canonical ids, not display names. `VER` and `Red Bull Racing` are how a human
writes it, but three-letter codes are reused across eras and team names change
without the entry changing, so a name-matching layer would silently predict for
the wrong driver. An unknown id is an error that names near matches.

Why observed weather is refused
-------------------------------

Any column naming the weather of the race being predicted is rejected outright
(`_INADMISSIBLE_COLUMNS`). v0's `weather_mixed` was its third most important
feature and was computed from the actual conditions of the race it was
predicting — the defect this project was rebuilt around. Weather still reaches
the model, as the circuit's climatological prior for that time of year, which
is knowable in advance (FR-206, FR-751a). What cannot be supplied is what
actually happened.

Every violation, not the first
------------------------------

`validate` collects. A user fixing a hand-written twenty-row grid should learn
about all four mistakes in one run, not discover a fifth after four more edits.
"""

from __future__ import annotations

import csv
import difflib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

import pandas as pd

from f1pred.reference import circuits

__all__ = [
    "GridCsvError",
    "GridEntry",
    "REQUIRED_COLUMNS",
    "SCHEMA_DOCUMENTATION",
    "Violation",
    "read_grid_csv",
    "validate",
]

#: FR-803's documented schema. Order here is the documented order; the reader
#: accepts any column order, because a spreadsheet reorders columns and that is
#: not a data defect.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "round",
    "race_date",
    "circuit_id",
    "driver_id",
    "constructor_id",
    "grid_position",
)

#: Columns that identify the race rather than the entry, and so must hold one
#: value across the whole file.
_RACE_SCOPED_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "round",
    "race_date",
    "circuit_id",
)

_INTEGER_COLUMNS: Final[tuple[str, ...]] = ("season", "round", "grid_position")

#: Refused on sight, with the reason. Owner decision, 2026-09-05: a file
#: carrying observed race weather is rejected rather than silently stripped,
#: because a user who wrote that column believes it was used.
_INADMISSIBLE_COLUMNS: Final[dict[str, str]] = {
    "weather_dry": "observed race-day weather",
    "weather_wet": "observed race-day weather",
    "weather_mixed": "observed race-day weather",
    "air_temp": "observed race-day weather",
    "track_temp": "observed race-day weather",
    "rainfall": "observed race-day weather",
    "finish_position": "the race outcome",
    "classified_position": "the race outcome",
    "position": "the race outcome",
    "status": "the race outcome",
    "laps_completed": "the race outcome",
    "points": "the race outcome",
    "fastest_lap": "the race outcome",
}

#: Seasons a grid file may plausibly describe. A typo'd year is otherwise
#: indistinguishable from a deliberate historical replay.
_MIN_SEASON: Final[int] = 1950
_MAX_SEASON: Final[int] = 2100

#: A grid larger than this is a duplicated-row mistake, not a bigger field.
_MAX_ENTRIES: Final[int] = 30


class GridCsvError(ValueError):
    """Raised when a grid CSV cannot be read as one valid starting grid."""

    def __init__(self, violations: list["Violation"], *, source: Path | None = None) -> None:
        self.violations = violations
        self.source = source
        super().__init__(format_violations(violations, source=source))


@dataclass(frozen=True, slots=True)
class Violation:
    """One problem with the file. `row` is the CSV line number, 1-based with
    the header as line 1, so it matches what a text editor shows."""

    row: int | None
    column: str | None
    message: str

    def render(self) -> str:
        where = ""
        if self.row is not None and self.column is not None:
            where = f"line {self.row}, column {self.column}: "
        elif self.row is not None:
            where = f"line {self.row}: "
        elif self.column is not None:
            where = f"column {self.column}: "
        return where + self.message


@dataclass(frozen=True, slots=True)
class GridEntry:
    """One validated row."""

    season: int
    round: int
    race_date: date
    circuit_id: str
    driver_id: str
    constructor_id: str
    grid_position: int


@dataclass(slots=True)
class _Collector:
    """Accumulates violations so the caller reports all of them at once."""

    violations: list[Violation] = field(default_factory=list)

    def add(self, message: str, *, row: int | None = None, column: str | None = None) -> None:
        self.violations.append(Violation(row=row, column=column, message=message))

    def __bool__(self) -> bool:
        return bool(self.violations)


def format_violations(violations: list[Violation], *, source: Path | None = None) -> str:
    """FR-803: all of them, numbered, so a user can work down the list."""
    head = f"{len(violations)} problem{'s' if len(violations) != 1 else ''} in "
    head += str(source) if source is not None else "the grid CSV"
    lines = [f"  {i}. {v.render()}" for i, v in enumerate(violations, start=1)]
    return head + ":\n" + "\n".join(lines)


def _check_columns(header: list[str], collect: _Collector) -> None:
    """Header-level checks: what is missing, what is refused, what is unknown."""
    seen = [c.strip() for c in header]
    duplicates = sorted({c for c in seen if seen.count(c) > 1})
    for name in duplicates:
        collect.add(f"appears more than once in the header", row=1, column=name)

    present = set(seen)
    for name in REQUIRED_COLUMNS:
        if name not in present:
            suggestion = _closest(name, present)
            hint = f"; did you mean the column named {suggestion!r}?" if suggestion else ""
            collect.add(f"required column {name!r} is missing{hint}", row=1)

    for name in sorted(present):
        if name in _INADMISSIBLE_COLUMNS:
            what = _INADMISSIBLE_COLUMNS[name]
            collect.add(
                f"carries {what}, which is not knowable before the race and "
                f"may not be supplied to a pre-race prediction. This is the "
                f"leakage defect the project was rebuilt to remove; remove the "
                f"column. Weather still reaches the model as the circuit's "
                f"climatological prior for that date, which is admissible.",
                row=1,
                column=name,
            )
        elif name not in REQUIRED_COLUMNS:
            collect.add(
                f"is not part of the documented schema and will not be used; "
                f"remove it so the file says what it means",
                row=1,
                column=name,
            )


def _closest(name: str, candidates: set[str]) -> str | None:
    matches = difflib.get_close_matches(name, sorted(candidates), n=1, cutoff=0.7)
    return matches[0] if matches else None


def _parse_int(raw: str, *, column: str, row: int, collect: _Collector) -> int | None:
    text = raw.strip()
    if not text:
        collect.add("is empty; every column is required on every row", row=row, column=column)
        return None
    try:
        return int(text)
    except ValueError:
        collect.add(f"expected a whole number, got {raw!r}", row=row, column=column)
        return None


def _parse_date(raw: str, *, row: int, collect: _Collector) -> date | None:
    text = raw.strip()
    if not text:
        collect.add("is empty; every column is required on every row", row=row, column="race_date")
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        collect.add(
            f"expected an ISO 8601 date such as 2026-03-08, got {raw!r}",
            row=row,
            column="race_date",
        )
        return None


def _parse_id(raw: str, *, column: str, row: int, collect: _Collector) -> str | None:
    text = raw.strip()
    if not text:
        collect.add("is empty; every column is required on every row", row=row, column=column)
        return None
    if text != text.lower() or " " in text:
        collect.add(
            f"expected a canonical lower-case id such as 'max_verstappen', got "
            f"{raw!r}; display names and three-letter codes are not accepted, "
            f"because they are reused across eras",
            row=row,
            column=column,
        )
        return None
    return text


def _check_race_scope(entries: list[dict[str, Any]], collect: _Collector) -> None:
    """One file, one race."""
    for column in _RACE_SCOPED_COLUMNS:
        values = {e[column] for e in entries if e.get(column) is not None}
        if len(values) > 1:
            rendered = ", ".join(sorted(str(v) for v in values))
            collect.add(
                f"holds more than one value across the file ({rendered}); a "
                f"grid CSV describes exactly one race",
                column=column,
            )


def _check_grid_positions(entries: list[dict[str, Any]], collect: _Collector) -> None:
    """FR-803 and the owner's ordering rule: 1..N, contiguous, unique."""
    positions = [(e["_row"], e["grid_position"]) for e in entries if e.get("grid_position") is not None]
    if not positions:
        return

    seen: dict[int, int] = {}
    for row, position in positions:
        if position < 1:
            collect.add(
                f"grid position must be 1 or greater, got {position}; pole is 1",
                row=row,
                column="grid_position",
            )
            continue
        if position in seen:
            collect.add(
                f"grid position {position} is already taken by line {seen[position]}; "
                f"two cars cannot start from one slot",
                row=row,
                column="grid_position",
            )
            continue
        seen[position] = row

    if not seen:
        return
    expected = set(range(1, len(seen) + 1))
    missing = sorted(expected - set(seen))
    if missing:
        rendered = ", ".join(str(m) for m in missing)
        collect.add(
            f"the grid has {len(seen)} entries so positions must run 1 to "
            f"{len(seen)} with no gaps; {rendered} "
            f"{'is' if len(missing) == 1 else 'are'} missing",
            column="grid_position",
        )


def _check_entry_uniqueness(entries: list[dict[str, Any]], collect: _Collector) -> None:
    seen: dict[str, int] = {}
    for entry in entries:
        driver = entry.get("driver_id")
        if driver is None:
            continue
        if driver in seen:
            collect.add(
                f"driver {driver!r} already appears on line {seen[driver]}; "
                f"a driver starts a race once",
                row=entry["_row"],
                column="driver_id",
            )
            continue
        seen[driver] = entry["_row"]


def _check_circuit(entries: list[dict[str, Any]], collect: _Collector) -> None:
    """FR-109: an undeclared circuit is an error, never a default."""
    known = set(circuits.CIRCUITS)
    reported: set[str] = set()
    for entry in entries:
        circuit_id = entry.get("circuit_id")
        if circuit_id is None or circuit_id in known or circuit_id in reported:
            continue
        reported.add(circuit_id)
        suggestion = _closest(circuit_id, known)
        hint = f"; did you mean {suggestion!r}?" if suggestion else ""
        collect.add(
            f"circuit {circuit_id!r} is not declared in reference/circuits.py"
            f"{hint}. Circuit attributes are declared data, so an unknown "
            f"circuit is missing data rather than a default",
            row=entry["_row"],
            column="circuit_id",
        )


def _check_season(entries: list[dict[str, Any]], collect: _Collector) -> None:
    for entry in entries:
        season = entry.get("season")
        if season is None:
            continue
        if not (_MIN_SEASON <= season <= _MAX_SEASON):
            collect.add(
                f"season {season} is outside {_MIN_SEASON}-{_MAX_SEASON} and "
                f"looks like a typo",
                row=entry["_row"],
                column="season",
            )
        round_number = entry.get("round")
        if round_number is not None and round_number < 1:
            collect.add(
                f"round must be 1 or greater, got {round_number}",
                row=entry["_row"],
                column="round",
            )


def validate(rows: list[dict[str, str]], header: list[str]) -> list[GridEntry]:
    """Check everything, then raise once with the whole list (FR-803)."""
    collect = _Collector()
    _check_columns(header, collect)

    if not rows:
        collect.add("the file has a header but no rows; a grid needs entries")
        raise GridCsvError(collect.violations)

    if len(rows) > _MAX_ENTRIES:
        collect.add(
            f"the file has {len(rows)} entries; a Formula 1 grid does not "
            f"exceed {_MAX_ENTRIES}, so this is most likely duplicated rows"
        )

    parsed: list[dict[str, Any]] = []
    for offset, raw in enumerate(rows):
        line = offset + 2  # header is line 1
        entry: dict[str, Any] = {"_row": line}
        for column in _INTEGER_COLUMNS:
            if column in raw:
                entry[column] = _parse_int(
                    raw[column] or "", column=column, row=line, collect=collect
                )
        if "race_date" in raw:
            entry["race_date"] = _parse_date(raw["race_date"] or "", row=line, collect=collect)
        for column in ("circuit_id", "driver_id", "constructor_id"):
            if column in raw:
                entry[column] = _parse_id(
                    raw[column] or "", column=column, row=line, collect=collect
                )
        parsed.append(entry)

    _check_season(parsed, collect)
    _check_race_scope(parsed, collect)
    _check_grid_positions(parsed, collect)
    _check_entry_uniqueness(parsed, collect)
    _check_circuit(parsed, collect)

    if collect:
        raise GridCsvError(collect.violations)

    entries = [
        GridEntry(
            season=e["season"],
            round=e["round"],
            race_date=e["race_date"],
            circuit_id=e["circuit_id"],
            driver_id=e["driver_id"],
            constructor_id=e["constructor_id"],
            grid_position=e["grid_position"],
        )
        for e in parsed
    ]
    return sorted(entries, key=lambda e: e.grid_position)


def read_grid_csv(path: Path) -> pd.DataFrame:
    """Read and validate one grid file, returning it in grid order.

    Raises `GridCsvError` listing every violation, or `FileNotFoundError`
    naming the path. The frame comes back sorted by grid position, so
    everything downstream sees the grid in the order it will start.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no grid CSV at {path}. The documented schema is: "
            f"{', '.join(REQUIRED_COLUMNS)}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if not header:
            raise GridCsvError(
                [Violation(row=None, column=None, message="the file is empty")],
                source=path,
            )
        rows = [row for row in reader]

    try:
        entries = validate(rows, header)
    except GridCsvError as error:
        raise GridCsvError(error.violations, source=path) from None

    return pd.DataFrame(
        {
            "season": pd.Series([e.season for e in entries], dtype="int64"),
            "round": pd.Series([e.round for e in entries], dtype="int64"),
            "race_date": pd.Series(
                [pd.Timestamp(e.race_date) for e in entries], dtype="datetime64[ns]"
            ),
            "circuit_id": pd.Series([e.circuit_id for e in entries], dtype="string"),
            "driver_id": pd.Series([e.driver_id for e in entries], dtype="string"),
            "constructor_id": pd.Series([e.constructor_id for e in entries], dtype="string"),
            "grid_position": pd.Series([e.grid_position for e in entries], dtype="int64"),
        }
    )


#: Rendered for `f1pred predict --schema`, and quoted in the README. One text,
#: so the documentation and the validator cannot drift apart.
SCHEMA_DOCUMENTATION: Final[str] = f"""\
Starting-grid CSV schema (FR-803)

Required columns, any order, one race per file:

  season          whole number   championship year, e.g. 2026
  round           whole number   round within the season, 1-based
  race_date       YYYY-MM-DD     the race date
  circuit_id      id             a circuit declared in reference/circuits.py
  driver_id       id             canonical corpus id, e.g. max_verstappen
  constructor_id  id             canonical corpus id, e.g. red_bull
  grid_position   whole number   1 for pole, running 1..N with no gaps

Ids are lower-case with underscores. Three-letter codes ('VER') and display
names ('Red Bull Racing') are refused: both are reused across eras, so matching
on them can silently predict for the wrong entry.

Refused columns. Anything describing the race being predicted -- observed
weather ({', '.join(sorted(c for c, w in _INADMISSIBLE_COLUMNS.items() if 'weather' in w))})
or its outcome -- is rejected rather than ignored, because a file carrying such
a column was written by someone who believed it would be used. Weather does
reach the model, as the circuit's climatological prior for that date, which is
knowable before the race.

Every violation is reported at once, not one per run.

Example:

  season,round,race_date,circuit_id,driver_id,constructor_id,grid_position
  2026,1,2026-03-08,albert_park,max_verstappen,red_bull,1
  2026,1,2026-03-08,albert_park,lando_norris,mclaren,2
"""
