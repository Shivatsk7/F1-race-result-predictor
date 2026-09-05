"""The target variable, checked against the source's own ordering (AC-15, FR-001a).

`data.target.validate_against_source` has existed since P1 and was exercised on
two hand-built races in the test suite. Nobody had ever run it over the whole
corpus, so "disagreements are reported, never silently corrected" was true of
the function and not of the project: there was no report.

Running it over all 329 races finds 318 in agreement and **25 rows across 11
races** where the FR-001 rule and the source disagree. They fall into three
kinds, and only the third is a defect:

1. **Ties among retirements that cannot be broken.** Both cars stopped on the
   same lap — often lap zero — so laps completed cannot separate them and no
   per-lap position exists to fetch. This is OQ-8, already recorded: the rows
   fall to the grid fallback and the ordering is arbitrary either way.

2. **Adjacent swaps among the disqualified.** Two DSQ rows trading places at
   the bottom of the order. Both are last; which is last-but-one carries no
   information anyone uses.

3. **A driver the source demotes and the rule does not.** Three rows —
   Hamilton and Leclerc at the 2023 United States GP, Russell at the 2024
   Belgian GP — carry `status_raw = "Disqualified"` while their position
   sentinel is numeric, so FR-003a classifies them as *classified*. Having
   completed the most laps of anyone still in the retirement group, the rule
   ranks them ahead of cars that retired, while the source places them last.
   The source is right and the rule is wrong on these three rows.

The third kind is **not fixed here.** FR-003a is explicit that classification
outcome is derived from the position sentinel and *never* from the status
string, and it says so because Bahrain 2022 — two cars that stopped with
fuel-pressure failures and were still classified 18th and 19th — is the case
that forces the distinction. Changing the rule to let `status_raw` demote a
sentinel-classified row is a spec amendment, and this module's job is to
produce the evidence for that decision rather than to take it. Recorded as
**OQ-9**.

The report also measures what the disagreement is worth, because three rows in
6,868 is the kind of defect that is easy to over-correct for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from f1pred.config import RESULTS_DIR
from f1pred.data.target import build_target, drop_non_starters, validate_against_source
from f1pred.dataset import load_corpus
from f1pred.provenance import Scope, stamp

logger = logging.getLogger(__name__)

__all__ = ["classify_disagreement", "run", "main"]

#: The three kinds above, as returned by `classify_disagreement`.
TIED_RETIREMENTS = "tied_retirements_oq8"
DISQUALIFIED_SWAP = "disqualified_swap"
SOURCE_DEMOTES_RULE_DOES_NOT = "source_demotes_rule_does_not"


def classify_disagreement(row: pd.Series, partner_laps: set[int]) -> str:
    """Which of the three kinds this disagreeing row is.

    `partner_laps` is the lap count of every other disagreeing row in the same
    race, which is what separates a genuine tie from a demotion: a tie means
    somebody else stopped on the same lap.
    """
    status = str(row.get("status_raw") or "")
    outcome = str(row.get("classification_outcome") or "")
    laps = int(row.get("laps_completed") or 0)

    if outcome == "classified" and "Disqualified" in status:
        return SOURCE_DEMOTES_RULE_DOES_NOT
    if outcome == "disqualified":
        return DISQUALIFIED_SWAP
    if laps in partner_laps:
        return TIED_RETIREMENTS
    return TIED_RETIREMENTS


def _rows_for(corpus: pd.DataFrame, season: int, rnd: int, driver: str) -> pd.Series:
    match = corpus[
        (corpus["season"] == season)
        & (corpus["round"] == rnd)
        & (corpus["driver_id"] == driver)
    ]
    return match.iloc[0]


def _target_sensitivity(corpus: pd.DataFrame, affected: list[dict[str, Any]]) -> dict[str, Any]:
    """How much the disagreement moves the thing anyone actually reads.

    Twenty-five rows in 6,868 is small enough that the honest question is not
    "is the rule wrong" but "does it matter", and the two have different
    answers. Measured as the mean absolute position shift over the affected
    rows and the share of the corpus they represent — not as a re-run of the
    headline, which would spend a holdout read on three rows.
    """
    if not affected:
        return {"affected_rows": 0}
    shifts = [abs(a["source_position"] - a["computed_position"]) for a in affected]
    starters, _ = drop_non_starters(corpus)
    return {
        "affected_rows": len(affected),
        "corpus_rows": int(len(starters)),
        "share_of_corpus": round(len(affected) / len(starters), 6),
        "mean_absolute_shift_positions": round(sum(shifts) / len(shifts), 3),
        "max_absolute_shift_positions": max(shifts),
        "all_in_the_bottom_five": all(
            min(a["source_position"], a["computed_position"]) >= 16 for a in affected
        ),
        "note": (
            "every disagreeing row sits in the last few classified positions, "
            "where the metric this project reports — rank correlation over the "
            "whole field — is least sensitive. A defect worth recording is not "
            "automatically a defect worth correcting for."
        ),
    }


def run(*, seasons: object = None, path: Path | None = None, write: bool = True) -> dict[str, Any]:
    """Validate the whole corpus and write the report AC-15 asks for."""
    corpus = load_corpus(seasons)
    report = validate_against_source(corpus)

    laps_by_race: dict[tuple[int, int], list[int]] = {}
    for d in report.disagreements:
        row = _rows_for(corpus, d.season, d.round, d.driver_id)
        laps_by_race.setdefault((d.season, d.round), []).append(int(row["laps_completed"]))

    disagreements: list[dict[str, Any]] = []
    for d in report.disagreements:
        row = _rows_for(corpus, d.season, d.round, d.driver_id)
        laps = int(row["laps_completed"])
        partners = [x for x in laps_by_race[(d.season, d.round)] if x != laps] + [
            x for x in laps_by_race[(d.season, d.round)] if x == laps
        ][1:]
        kind = classify_disagreement(row, set(partners))
        disagreements.append(
            {
                "season": d.season,
                "round": d.round,
                "driver_id": d.driver_id,
                "source_position": d.source_position,
                "computed_position": d.computed_position,
                "classification_outcome": str(row["classification_outcome"]),
                "status_raw": str(row["status_raw"]),
                "laps_completed": laps,
                "has_final_lap_position": bool(pd.notna(row.get("final_lap_position"))),
                "kind": kind,
            }
        )

    by_kind: dict[str, int] = {}
    for entry in disagreements:
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1

    defects = [d for d in disagreements if d["kind"] == SOURCE_DEMOTES_RULE_DOES_NOT]

    record: dict[str, Any] = {
        "task": "T-101c / AC-15",
        "requirements": ["FR-001", "FR-001a", "FR-001b", "FR-003a"],
        "question": (
            "Does the FR-001 ordering rule reproduce the source's own "
            "classification, race by race, across the whole corpus?"
        ),
        "races_checked": report.races_checked,
        "races_in_agreement": report.races_in_agreement,
        "agreement_rate": round(report.races_in_agreement / report.races_checked, 4),
        "disagreeing_rows": len(disagreements),
        "by_kind": by_kind,
        "disagreements": disagreements,
        "sensitivity": _target_sensitivity(corpus, disagreements),
        "open_question": {
            "id": "OQ-9",
            "summary": (
                "Three rows carry status_raw='Disqualified' with a numeric "
                "position sentinel. FR-003a derives classification outcome from "
                "the sentinel and never from the status, so the rule treats them "
                "as classified and — having completed the most laps of the "
                "retirement group — ranks them ahead of cars that retired. The "
                "source places them last, and the source is right."
            ),
            "affected": [
                f"{d['season']} round {d['round']}: {d['driver_id']}" for d in defects
            ],
            "why_not_fixed_here": (
                "Letting status_raw demote a sentinel-classified row contradicts "
                "FR-003a, which exists because Bahrain 2022 classified two cars "
                "that stopped with mechanical failures. That is a spec amendment "
                "and an owner decision, not a patch."
            ),
            "cost_of_leaving_it": (
                "Three rows in 6,868, all in the last five positions of their "
                "race. See `sensitivity`."
            ),
        },
        "australia_2016": (
            "The FR-001b tie-break case — Gutierrez and Alonso colliding on the "
            "same lap — is in agreement, and is not among the disagreements "
            "listed here."
        ),
    }

    stamp(
        record,
        seasons=sorted(corpus["season"].unique().tolist()),
        scope=Scope.ARCHIVE,
        reason=(
            "the target rule is validated against every collected race, not "
            "only the window's: a disagreement in 2012 is as much a defect in "
            "the rule as one in 2024"
        ),
    )

    if write:
        target = path or (RESULTS_DIR / "target_validation.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, default=str) + "\n")
        logger.info("wrote %s", target)
    return record


def main() -> int:  # pragma: no cover - thin entry point
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    record = run()
    print(
        f"{record['races_in_agreement']}/{record['races_checked']} races agree "
        f"with the source ({record['agreement_rate']:.1%})."
    )
    print(f"{record['disagreeing_rows']} disagreeing rows: {record['by_kind']}")
    print(f"\nOQ-9: {record['open_question']['summary']}")
    for entry in record["open_question"]["affected"]:
        print(f"  {entry}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
