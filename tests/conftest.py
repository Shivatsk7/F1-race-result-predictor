"""T-609: the offline guarantee and the suite's time budget, both enforced.

NFR-05 says no pipeline stage except collection may touch the network, and
NFR-04 says the suite finishes in under sixty seconds without it. Until this
file existed, both were conventions: the `network` marker kept collection tests
out of the default run, and nothing checked that an *unmarked* test stayed
offline. A test that quietly reached the internet would pass, slowly, and the
first sign would be a red build on a machine with no connection.

So the socket is closed here for every test that is not marked `network`, and
the run's own wall-clock is measured against the budget rather than asserted to
be within it.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: NFR-04. The budget is the requirement's number, not a comfortable margin
#: around the current measurement.
SUITE_BUDGET_SECONDS = 60.0

#: Loopback stays open. Nothing here needs it, but a blanket block would fail
#: on tooling that talks to a local process, and the requirement is about *the
#: network*, not about sockets as a concept.
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class NetworkAccessDenied(RuntimeError):
    """Raised when an offline test tries to open a network connection."""


_real_connect = socket.socket.connect


def _blocked_connect(self: socket.socket, address: Any) -> None:
    host = address[0] if isinstance(address, tuple) else address
    if str(host) in _ALLOWED_HOSTS:
        return _real_connect(self, address)
    raise NetworkAccessDenied(
        f"this test tried to connect to {host!r}. Under NFR-05 no stage but "
        f"data collection may use the network, and under NFR-04 the suite must "
        f"run offline. If this test really does need the network, mark it "
        f"`@pytest.mark.network` — which also excludes it from the default run."
    )


@pytest.fixture(autouse=True)
def _offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Close the socket for every test that has not declared it needs one."""
    if request.node.get_closest_marker("network"):
        return
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)


def pytest_sessionstart(session: pytest.Session) -> None:
    session.stash_started = time.perf_counter()  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Measure the run, report it, and record it when the whole suite ran.

    Recorded rather than printed, for the reason NFR-01 gives about the
    evaluation budget: a timing nobody can check afterwards is a timing nobody
    is keeping. The file is written only for a complete, unfiltered run, so a
    single-file run during development cannot overwrite the suite's number with
    its own.
    """
    started = getattr(session, "stash_started", None)
    if started is None:
        return
    elapsed = time.perf_counter() - started

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        verdict = "within" if elapsed < SUITE_BUDGET_SECONDS else "OVER"
        reporter.write_line(
            f"NFR-04: suite took {elapsed:.1f}s, {verdict} the "
            f"{SUITE_BUDGET_SECONDS:.0f}s budget (offline; NFR-05 enforced)"
        )

    # Unfiltered means: no -k, no -m beyond the ini default, and no explicit
    # path. `pytest` with nothing after it is the run the budget describes; a
    # single-file run during development must not overwrite its number.
    option = session.config.option
    filtered = bool(option.keyword) or bool(option.file_or_dir)
    if exitstatus == 0 and not filtered:
        _record(elapsed, session.testscollected)


def _record(elapsed: float, collected: int) -> None:
    path = ROOT / "results" / "test_suite_timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "wall_clock_seconds": round(elapsed, 2),
                "budget_seconds": SUITE_BUDGET_SECONDS,
                "within_budget": elapsed < SUITE_BUDGET_SECONDS,
                "tests_collected": collected,
                "network": "blocked; see tests/conftest.py",
                "reference_machine": "Apple Silicon, macOS 25.6, Python 3.13 (plan.md §4)",
            },
            indent=2,
        )
        + "\n"
    )
