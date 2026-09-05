"""What corpus produced a number, stamped onto every results file.

Two acceptance criteria ask for the same thing from opposite directions.
AC-11 wants every results file to record the window width `W` and the concrete
season list it used. AC-21 wants the collection failure manifest reproduced in
every results file, so that a reader can tell a race that is *absent* from a
race that never happened. `FailureManifest`'s own docstring has claimed since
P1 that it "travels with the data and is reproduced in every results file" —
which was true of the collection reports and of nothing else.

Both are answers to one question: **which corpus is this number about?** So
there is one stamp, applied by the writers rather than by each caller, and one
test that fails when a file under `results/` lacks it. A convention that every
writer remembers to record its provenance is the kind of convention this
project has already watched drift twice.

Not every results file is about the rolling window. The regulation-transfer
experiment deliberately reads the whole archive (amendment 2026-08-25), and the
noise floor reads the corpus entire. Those declare `scope` accordingly rather
than reporting a window they did not use — a file claiming `W=16` when it read
the archive would be worse than one claiming nothing.
"""

from __future__ import annotations

import json
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Final

from f1pred.config import RAW_DIR, DEFAULT_WINDOW_WIDTH

__all__ = [
    "PROVENANCE_KEY",
    "Scope",
    "corpus_provenance",
    "failure_manifest_summary",
    "stamp",
]

#: The single key every results file carries.
PROVENANCE_KEY: Final[str] = "corpus_provenance"


class Scope(Enum):
    """Which slice of the collected data a results file is about."""

    #: The FR-508 rolling window. The default, and what every evaluation uses.
    WINDOW = "window"
    #: Every season on disk, window or not. FR-511's transfer experiment reads
    #: this deliberately, so that a historical observation cannot age out of
    #: the window and silently disappear from the result.
    ARCHIVE = "archive"
    #: A subset chosen for a reason the file itself states — the telemetry arm's
    #: 2018+ seasons, for instance (FR-214).
    SUBSET = "subset"
    #: Not about race data at all: a timing measurement, or a record of a
    #: measurement made against code that no longer exists.
    NOT_CORPUS = "not_corpus"


def failure_manifest_summary(root: Path = RAW_DIR) -> dict[str, Any]:
    """AC-21: the collection manifest, small enough to reproduce everywhere.

    The full failure list is reproduced only when it is non-empty. A summary
    that says `failure_count: 0` carries the whole fact; a summary that says
    `failure_count: 3` without saying which three does not, and that is exactly
    when a reader needs them.
    """
    path = Path(root) / "failure_manifest.json"
    if not path.exists():
        return {
            "available": False,
            "note": (
                f"no manifest at {path}; the corpus was not collected by "
                f"`f1pred collect`, so whether any race failed is unknown "
                f"rather than known to be none"
            ),
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "available": True,
        "races_attempted": manifest.get("races_attempted"),
        "failure_count": manifest.get("failure_count"),
        "failure_rate": manifest.get("failure_rate"),
        "max_failure_rate": manifest.get("max_failure_rate"),
    }
    failures = manifest.get("failures") or []
    if failures:
        summary["failures"] = failures
        summary["note"] = (
            "these races are absent from the corpus. Rolling features treat "
            "them as gaps rather than as races that did not happen (FR-104b)."
        )
    return summary


def corpus_provenance(
    seasons: object = None,
    *,
    scope: Scope = Scope.WINDOW,
    width: int | None = None,
    reason: str | None = None,
    root: Path = RAW_DIR,
) -> dict[str, Any]:
    """The block that says which corpus a results file is about.

    `width` is the window's `W`. For a non-window scope it is reported as the
    number of seasons actually read, alongside the scope that explains why the
    window does not apply — never omitted, because an absent field reads as an
    oversight and a present one with a scope reads as a decision.
    """
    listed = sorted({int(s) for s in seasons}) if seasons is not None else []
    if width is None:
        width = DEFAULT_WINDOW_WIDTH if scope is Scope.WINDOW else len(listed)

    block: dict[str, Any] = {
        "scope": scope.value,
        "width": width,
        "seasons": listed,
        "first_season": listed[0] if listed else None,
        "last_season": listed[-1] if listed else None,
        "resolved": date.today().isoformat(),
        "failure_manifest": failure_manifest_summary(root),
    }
    if scope is not Scope.WINDOW:
        block["why_not_the_window"] = reason or (
            "this file's scope is not the FR-508 rolling window; see the file's "
            "own description"
        )
    return block


def stamp(
    record: dict[str, Any],
    *,
    seasons: object = None,
    scope: Scope = Scope.WINDOW,
    width: int | None = None,
    reason: str | None = None,
    root: Path = RAW_DIR,
) -> dict[str, Any]:
    """Add the provenance block to `record`, in place, and return it.

    An existing block is left alone. Callers that already describe their split
    in detail — the evaluation harness writes a full `split` — get the stamp
    alongside it rather than instead of it, because the two answer different
    questions: `split` says which seasons trained and which were held out, and
    this says which corpus both halves came from.
    """
    if PROVENANCE_KEY in record:
        return record
    if seasons is None:
        seasons = _seasons_from(record)
    record[PROVENANCE_KEY] = corpus_provenance(
        seasons, scope=scope, width=width, reason=reason, root=root
    )
    return record


def _seasons_from(record: dict[str, Any]) -> list[int]:
    """Find the season list a record already carries, rather than re-deriving it.

    Re-reading the corpus from disk would risk stamping a file with the seasons
    available *now* instead of the seasons the run actually used — which is the
    precise failure the stamp exists to prevent.
    """
    split = record.get("split")
    if isinstance(split, dict):
        window = split.get("window")
        if isinstance(window, dict) and window.get("seasons"):
            return [int(s) for s in window["seasons"]]
        training = split.get("training_seasons") or []
        holdout = split.get("holdout_seasons") or []
        if training or holdout:
            return sorted({int(s) for s in [*training, *holdout]})
    for key in ("seasons", "corpus_seasons"):
        value = record.get(key)
        if isinstance(value, list) and value and all(isinstance(v, int) for v in value):
            return sorted(value)
    return []
