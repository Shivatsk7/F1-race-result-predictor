"""The board must agree with the spec and with the code (Constitution VII).

These are not tests of the predictor. They are tests of the project's own
bookkeeping, and they exist because that bookkeeping drifted three times in one
week and every drift was found by hand:

* a requirement (FR-701) with no task at all;
* five acceptance criteria named nowhere on the board;
* a feature the board called `seasons_since_regime_start` and the code called
  `seasons_since_regime`, which makes a "done when" clause unverifiable.

Reading a 250-line board catches none of those reliably. A script catches all
of them in under a second, so the check belongs here rather than in a habit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.features.registry import REGISTRY

SPEC_DIR = Path(__file__).resolve().parents[1] / "specs" / "001-core-predictor"
SPEC = (SPEC_DIR / "spec.md").read_text()
TASKS = (SPEC_DIR / "tasks.md").read_text()

TASK_ROW = re.compile(
    r"^\|\s*([☐☑◐⊘])\s*\*\*(T-[0-9a-z]+)\*\*\s*\|([^|]*)\|([^|]*)\|", re.M
)


def defined_requirements() -> set[str]:
    """IDs the spec actually defines, as table rows — not merely mentions."""
    return {
        m.group(1)
        for m in re.finditer(r"^\|\s*\*\*((?:FR|NFR)-[0-9a-z]+)\*\*\s*\|", SPEC, re.M)
    }


def task_rows() -> list[tuple[str, str, str, str]]:
    return [m.groups() for m in TASK_ROW.finditer(TASKS)]


def cited_ids() -> set[str]:
    ids: set[str] = set()
    for _status, _tid, _name, satisfies in task_rows():
        ids |= set(re.findall(r"(?:FR|NFR|AC)-[0-9a-z]+", satisfies))
    return ids


class TestTraceability:
    def test_every_requirement_has_a_task(self):
        orphans = sorted(defined_requirements() - cited_ids())
        assert not orphans, (
            f"requirements with no task: {orphans}. Either add a task or record "
            f"the requirement as deliberately unexercised, as FR-217/FR-218 are."
        )

    def test_every_acceptance_criterion_appears_on_the_board(self):
        acs = set(re.findall(r"\*\*(AC-\d+)\*\*", SPEC))
        missing = sorted(a for a in acs if a not in TASKS)
        assert not missing, f"acceptance criteria named nowhere on the board: {missing}"

    def test_no_task_cites_an_undefined_requirement(self):
        ghosts = sorted(
            i
            for i in cited_ids()
            if i.startswith(("FR-", "NFR-")) and i not in defined_requirements()
        )
        assert not ghosts, f"tasks cite requirements the spec does not define: {ghosts}"

    def test_every_task_cites_something(self):
        # Constitution VII.2: a task tracing to no requirement is out of scope.
        naked = [
            tid for _s, tid, _n, satisfies in task_rows()
            if not re.search(r"(?:FR|NFR|AC|OQ|D)-[0-9a-z]+|Spec §|VII\.", satisfies)
        ]
        assert not naked, f"tasks citing no requirement: {naked}"


class TestBoardMatchesCode:
    """A "done when" clause that names something the code does not have is not
    checkable, which Constitution VII.4 forbids.

    The rule is deliberately blunt: a backticked snake_case identifier in a
    completed row must appear **as a whole word** somewhere in the package
    source. That admits everything the board legitimately mentions — schema
    columns, method names, dataclass fields, flag literals, keyword arguments —
    while still catching a name the code has since renamed.

    Whole-word matching is what makes it work. `seasons_since_regime` is a
    substring of `seasons_since_regime_start`, so a naive containment test
    would call the stale name present; a `\\b` boundary does not, because the
    trailing underscore is a word character. That exact rename is what this
    class was written for, and the first version of it — which filtered
    candidates by a handful of name suffixes — did not catch it.
    """

    #: Concatenated package source, searched once.
    SOURCE = "\n".join(
        p.read_text() for p in (Path(__file__).resolve().parents[1] / "src").rglob("*.py")
    )

    #: Identifiers the board may name that deliberately do not exist in the
    #: code. Each needs a reason; an entry without one is a bug being hidden.
    ALLOWED_ABSENT: dict[str, str] = {}

    def test_identifiers_named_by_completed_tasks_exist_in_the_code(self):
        missing: dict[str, str] = {}
        for m in re.finditer(r"^\|\s*(☑)\s*\*\*(T-[0-9a-z]+)\*\*.*$", TASKS, re.M):
            tid = m.group(2)
            for token in re.findall(r"`([a-z][a-z0-9_]*_[a-z0-9_]+)`", m.group(0)):
                if token in self.ALLOWED_ABSENT:
                    continue
                if re.search(rf"\b{re.escape(token)}\b", self.SOURCE):
                    continue
                missing.setdefault(token, tid)
        assert not missing, (
            f"completed tasks name identifiers the code does not have: {missing}. "
            f"The board and the code have drifted apart, which makes those "
            f"'done when' clauses unverifiable (Constitution VII.4)."
        )

    def test_the_check_would_catch_a_rename(self):
        """Guards the guard: prove whole-word matching rejects a stale prefix.

        Without this, someone 'simplifying' the regex to a containment test
        would silently disable the class and every test here would still pass.
        """
        assert re.search(r"\bseasons_since_regime_start\b", self.SOURCE)
        assert not re.search(r"\bseasons_since_regime\b", self.SOURCE)


class TestStatusDiscipline:
    def test_dropped_tasks_explain_themselves(self):
        for status, tid, _name, _sat in task_rows():
            if status != "⊘":
                continue
            row = TASKS[TASKS.index(f"**{tid}**") :].split("\n", 1)[0]
            assert "DROPPED" in row.upper(), f"{tid} is dropped without saying so"
            assert len(row) > 300, f"{tid} is dropped without a substantive reason"

    def test_completed_tasks_record_when(self):
        """Either the row carries a date, or its phase was signed off with one.

        P0 uses the second form — its six rows are dated by the "P0 COMPLETE
        (2026-08-15)" note under the table rather than individually. Both are
        legitimate; what is not legitimate is a task marked done with no date
        anywhere, which leaves no way to order it against a spec amendment.
        """
        phase_bounds = [
            (m.start(), m.group(1))
            for m in re.finditer(r"^## (Phase P\d)", TASKS, re.M)
        ]

        def phase_of(index: int) -> str:
            found = "before P0"
            for start, name in phase_bounds:
                if start <= index:
                    found = name
                else:
                    break
            return found

        dated_phases = {
            m.group(1).replace("P", "Phase P")
            for m in re.finditer(r"\*\*(P\d) COMPLETE\*\* \(\d{4}-\d{2}-\d{2}\)", TASKS)
        }

        undated = []
        for m in TASK_ROW.finditer(TASKS):
            if m.group(1) != "☑":
                continue
            # The whole line: the completion note lives in the last column,
            # well past where TASK_ROW stops matching.
            row = TASKS[m.start() : TASKS.index("\n", m.start())]
            if re.search(r"DONE \d{4}-\d{2}-\d{2}", row):
                continue
            if phase_of(m.start()) in dated_phases:
                continue
            undated.append(m.group(2))
        assert not undated, f"completed tasks with no completion date: {undated}"
