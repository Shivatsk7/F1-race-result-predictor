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

import json
import re
from pathlib import Path

import pytest


import f1pred.features.definitions  # noqa: F401
from f1pred.features.registry import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "specs" / "001-core-predictor"
SPEC = (SPEC_DIR / "spec.md").read_text()
TASKS = (SPEC_DIR / "tasks.md").read_text()
PLAN = (SPEC_DIR / "plan.md").read_text()
CONSTITUTION = (ROOT / ".specify" / "memory" / "constitution.md").read_text()

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


def defined_ids() -> set[str]:
    """Every identifier the spec, the board, or the plan actually *defines*.

    Each source declares its ids in one shape, so each is matched in that
    shape rather than by a general search: a requirement is a spec table row,
    a task is a board row, an acceptance criterion is a checklist entry, an
    open question is a table row, and a decision is a plan heading. Matching a
    bare mention instead would let a citation validate itself — the ghost id
    appears in the source file, so it would be "defined" by being cited.
    """
    ids: set[str] = set(defined_requirements())
    ids |= {m.group(1) for m in re.finditer(r"^\|\s*\*\*(US-\d+)", SPEC, re.M)}
    ids |= set(re.findall(r"\*\*(AC-\d+)\*\*", SPEC))
    ids |= set(re.findall(r"\*\*(US-\d+)[ —-]", SPEC))
    ids |= {m.group(1) for m in re.finditer(r"^\|\s*\*\*(OQ-\d+)\*\*", SPEC, re.M)}
    ids |= {tid for _s, tid, _n, _sat in task_rows()}
    ids |= {m.group(1) for m in re.finditer(r"^###\s+(D-\d+)\b", PLAN, re.M)}
    ids |= {m.group(1) for m in re.finditer(r"^###\s+(D-\d+)\b", CONSTITUTION, re.M)}
    return ids


class TestSourceCitations:
    """A requirement id in a docstring must name a requirement that exists.

    The board is checked against the spec above, and the code against the
    board. Nothing checked the third edge: the ids the *source* cites. That
    gap let `collect.py` point readers at "the FR-001c note in spec §3.0" for
    eleven days — a requirement that was never written. The zero-lap finding
    it meant is OQ-8, recorded in §7.3.

    A citation is the cheapest kind of documentation and the easiest to leave
    behind when a spec is amended, so it is worth a second of test time.
    """

    #: Package and test source, with the file each line came from.
    SOURCES = [
        (p.relative_to(ROOT), p.read_text())
        for p in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    ]

    CITE = re.compile(r"\b((?:FR|NFR|AC|OQ|US|T|D)-\d+[a-z]?)\b")

    #: This file names undefined ids on purpose, to prove the check rejects
    #: them. The exemption is scoped to this file so the same id written in
    #: the package — the actual defect — still fails. Each entry needs a
    #: reason; an entry without one is a bug being hidden.
    SELF_TEST_PATH = Path("tests") / "test_board.py"
    ALLOWED_GHOSTS: dict[str, str] = {
        "FR-001c": "the ghost citation this class was written for; asserted absent below",
    }

    def test_every_id_cited_in_source_is_defined(self):
        defined = defined_ids()
        ghosts: dict[str, str] = {}
        for path, text in self.SOURCES:
            for lineno, line in enumerate(text.splitlines(), 1):
                for cid in self.CITE.findall(line):
                    if cid in defined:
                        continue
                    if path == self.SELF_TEST_PATH and cid in self.ALLOWED_GHOSTS:
                        continue
                    ghosts.setdefault(cid, f"{path}:{lineno}")
        assert not ghosts, (
            f"source cites ids that no spec, board, or plan defines: {ghosts}. "
            f"Either the id is a typo, or it was removed by an amendment and "
            f"the citation was left behind."
        )

    def test_the_check_would_catch_a_ghost_citation(self):
        """Guards the guard: an id nobody defines must be rejected.

        Without this, a `defined_ids()` that accidentally matched bare
        mentions would pass every citation — including the one this class
        exists to catch — while still looking green.
        """
        assert "FR-001b" in defined_ids()
        assert "FR-001c" not in defined_ids()


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


class TestDocumentedNumbers:
    """Every performance figure in the docs must resolve to a committed file.

    FR-902 and AC-7 say so, and until this class existed it was a convention
    rather than a check. Conventions about numbers are exactly what this
    project failed at once already: the v0 README quoted figures the code could
    not reproduce, which is the sentence the current README opens with.

    The rule is deliberately blunt, in the manner of `TestBoardMatchesCode`
    above: **any number written to four decimal places** in the README or on
    the board must appear, rounded to four places, somewhere in the values of
    some file under `results/`. Four places is the tell — this project writes
    scores and intervals that way and writes nothing else that way, so the
    pattern selects performance claims and skips row counts, season numbers and
    hyperparameters without needing a list of which is which.

    It found two the day it was written: `0.9625` and `0.6302`, the in-sample
    and holdout scores that justify early stopping on the T-401 row. They came
    from a diagnostic run by hand in a session and committed nowhere, so the
    board's argument for its own design decision rested on a number no reader
    could check. `evaluate/fit_diagnostic.py` now produces them.
    """

    #: A four-decimal number in the docs that legitimately has no results file.
    #: Each entry needs a reason; an entry without one is a claim being hidden.
    ALLOWED_UNSOURCED: dict[str, str] = {}

    FOUR_DP = re.compile(r"[+-]?\d+\.\d{4}\b")

    @staticmethod
    def committed_values() -> set[float]:
        values: set[float] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                values.add(round(float(node), 4))

        for path in sorted((ROOT / "results").glob("*.json")):
            walk(json.loads(path.read_text()))
        return values

    @pytest.mark.parametrize("document", ["README.md", "specs/001-core-predictor/tasks.md"])
    def test_every_four_decimal_figure_resolves_to_results(self, document):
        committed = self.committed_values()
        text = (ROOT / document).read_text()
        unsourced = {
            token
            for token in (m.group(0) for m in self.FOUR_DP.finditer(text))
            if token not in self.ALLOWED_UNSOURCED
            and round(float(token), 4) not in committed
        }
        assert not unsourced, (
            f"{document} quotes {sorted(unsourced)}, which resolve to no file "
            f"in results/. Under FR-902 a figure that cannot cite a committed "
            f"file does not get written down — either commit the run that "
            f"produced it, or take the claim out."
        )

    def test_the_check_would_catch_an_unsourced_figure(self):
        """Guards the guard: a number nothing produced must be rejected.

        Without this, a `committed_values` that accidentally returned every
        float — or an empty document — would pass while proving nothing.
        """
        committed = self.committed_values()
        assert committed, "no results files were read; the check is vacuous"
        assert round(0.123456789, 4) not in committed


class TestHeadlineTableMatchesTheResultsFile:
    """T-606: the README's headline table, cell by cell, against its source.

    `TestDocumentedNumbers` above catches a four-decimal figure that resolves
    to nothing. It cannot see the headline table any more, because FR-804 now
    renders those scores at the two decimal places their intervals support —
    and `0.66` is far too common a number for a "does it appear in some results
    file" check to mean anything.

    So this checks the stronger property instead: every cell of that table is
    what `render.format_measurement` produces from the committed record. A
    re-run that moved a score would fail here rather than leaving the README
    quietly describing a previous run — which is the failure FR-902 exists to
    prevent, and the one the README's own opening paragraph is about.
    """

    RESULTS = ROOT / "results" / "model_vs_baselines_2023_2025.json"

    #: The table's rows, as (README label, predictor name in the record).
    ROWS = (
        ("**model** (XGBoost ranker)", "xgboost_ranker"),
        ("grid order", "grid_order"),
        ("championship-standings order", "standings_order"),
        ("random order (1000 seeded draws)", "random_order"),
    )

    @staticmethod
    def readme() -> str:
        return (ROOT / "README.md").read_text()

    def record(self) -> dict:
        return json.loads(self.RESULTS.read_text())

    @pytest.mark.parametrize(("label", "predictor"), ROWS)
    def test_each_row_states_what_the_results_file_holds(self, label, predictor):
        from f1pred import render

        metric = self.record()["predictors"][predictor]["metrics"]["spearman"]
        point = render.format_measurement(
            metric["point"], low=metric["ci_low"], high=metric["ci_high"]
        )
        low = render._round_outward(metric["ci_low"], len(point.split(".")[1]), up=False)
        high = render._round_outward(metric["ci_high"], len(point.split(".")[1]), up=True)

        rows = [line for line in self.readme().splitlines() if line.startswith(f"| {label} |")]
        assert len(rows) == 1, f"expected exactly one README row for {label!r}, got {rows}"
        cells = [c.strip().strip("*") for c in rows[0].split("|")]
        assert point in cells, f"{label}: README says {cells[2]!r}, results say {point!r}"
        assert f"{low} to {high}" in rows[0], (
            f"{label}: README interval disagrees with {self.RESULTS.name}, "
            f"which gives {low} to {high}"
        )

    def test_the_stated_delta_matches_the_records_verdict(self):
        from f1pred import render

        verdict = self.record()["verdict"]
        entry = next(e for e in verdict["not_beaten_by"] if e["name"] == "xgboost_ranker")
        places = render.decimals_supported_by(
            max(entry["point"] - entry["ci_low"], entry["ci_high"] - entry["point"])
        )
        sentence = (
            f"Model minus grid: {entry['point']:+.{places}f}, 95% CI "
            f"{render._round_outward(entry['ci_low'], places, up=False)} to "
            f"{entry['ci_high']:+.{places}f}"
        )
        # Rendered by hand in the README's prose; this is what it must say.
        assert "+0.00" in self.readme(), sentence

    def test_the_headline_verdict_is_the_one_the_record_computes(self):
        # If a re-run ever *did* beat the grid, the README's headline sentence
        # would be wrong and this is what would say so.
        beats = self.record()["verdict"]["any_predictor_beats_the_grid"]
        assert beats is False
        assert "**The model does not beat the starting grid.**" in self.readme()

    def test_the_within_regime_caveat_is_present_wherever_the_headline_is(self):
        """T-611, OQ-1: the holdout sits inside one regulation regime."""
        from f1pred.config import regime_for_season

        readme = self.readme()
        holdout = self.record()["split"]["holdout_seasons"]
        regimes = {regime_for_season(int(s)).start_season for s in holdout}
        assert len(regimes) == 1, (
            "the holdout now spans more than one regulation regime, so the "
            "README's within-regime caveat needs rewriting rather than keeping"
        )
        assert "within-regime caveat" in readme.lower()
        assert "entirely inside one regulation regime" in readme

    def test_the_calibration_split_matches_the_evaluation_split(self):
        """Two files, one holdout: the README quotes both as the same result."""
        calibration = json.loads(
            (ROOT / "results" / "calibration_2023_2025.json").read_text()
        )
        assert calibration["split"]["holdout_seasons"] == self.record()["split"][
            "holdout_seasons"
        ]


class TestTheOfflineGuardWorks:
    """T-609, NFR-04, NFR-05: guard the guard in `tests/conftest.py`.

    The offline fixture is autouse, so a broken version fails open — every test
    keeps passing and nothing says the network is reachable again. These are
    the tests that would notice.
    """

    def test_an_outbound_connection_is_refused(self):
        import socket

        from tests.conftest import NetworkAccessDenied

        with pytest.raises(NetworkAccessDenied, match="NFR-05"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("example.com", 80))

    def test_the_refusal_says_how_to_declare_a_networked_test(self):
        import socket

        from tests.conftest import NetworkAccessDenied

        with pytest.raises(NetworkAccessDenied, match="pytest.mark.network"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 443))

    def test_loopback_is_still_allowed(self):
        """The requirement is about the network, not about sockets."""
        import socket

        from tests.conftest import _blocked_connect

        # Reaches the real connect and fails on refused-connection, not on the
        # guard — which is what proves loopback took the allowed path.
        with pytest.raises(OSError) as caught:
            _blocked_connect(socket.socket(socket.AF_INET, socket.SOCK_STREAM), ("127.0.0.1", 1))
        assert "NFR-05" not in str(caught.value)

    def test_the_collection_tests_are_marked_and_therefore_deselected(self):
        """NFR-05's exception is data collection, and only data collection."""
        marked = re.findall(r"@pytest\.mark\.network", (ROOT / "tests").joinpath("test_fetch.py").read_text())
        assert marked, "the networked tests lost their marker"


class TestEveryResultsFileDeclaresItsCorpus:
    """AC-11 and AC-21, enforced rather than swept once.

    AC-11 asks every results file to record `W` and its concrete season list.
    AC-21 asks the collection failure manifest to be reproduced in every
    results file, so a reader can tell an *absent* race from one that never
    happened — which `FailureManifest`'s own docstring has claimed since P1 and
    which was true only of the collection reports.

    Both answer the same question — which corpus is this number about? — so
    there is one stamp and this is the check on it. Written after a sweep found
    six files carrying no window at all: a sweep fixes today's files, and a
    test fixes the next one somebody writes.
    """

    FILES = sorted((ROOT / "results").glob("*.json"))

    @staticmethod
    def block(path):
        from f1pred.provenance import PROVENANCE_KEY

        return json.loads(path.read_text()).get(PROVENANCE_KEY)

    def test_there_are_results_files_to_check(self):
        assert self.FILES, "no results files found; this check would be vacuous"

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
    def test_every_file_declares_which_corpus_produced_it(self, path):
        block = self.block(path)
        assert block is not None, (
            f"{path.name} carries no corpus provenance. Under AC-11 every "
            f"results file records its window width and season list; write it "
            f"through a stamped writer rather than by hand."
        )

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
    def test_every_file_reproduces_the_failure_manifest(self, path):
        manifest = (self.block(path) or {}).get("failure_manifest")
        assert manifest is not None, f"{path.name} does not reproduce the manifest (AC-21)"
        assert "available" in manifest

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
    def test_a_non_window_scope_says_why(self, path):
        """A file that did not use the window must say so, not stay silent."""
        from f1pred.provenance import Scope

        block = self.block(path) or {}
        if block.get("scope") != Scope.WINDOW.value:
            assert block.get("why_not_the_window"), (
                f"{path.name} declares scope {block.get('scope')!r} without a "
                f"reason. An unexplained non-window scope is indistinguishable "
                f"from a mistake."
            )

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
    def test_a_corpus_scoped_file_names_real_seasons(self, path):
        from f1pred.provenance import Scope

        block = self.block(path) or {}
        if block.get("scope") == Scope.NOT_CORPUS.value:
            return
        seasons = block.get("seasons")
        assert seasons, f"{path.name} names no seasons"
        assert block["width"] >= 1
        assert seasons == sorted(seasons), f"{path.name} lists seasons out of order"
