"""T-101e — fetching FR-001b's final-lap tie-break signal.

The rule these tests guard: retirements tied on laps completed are separated by
track position on the last lap they finished. The interesting cases are the ones
where that signal does not exist, because the rule is silent about them and the
grid fallback has to catch them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from f1pred.data.collect import (
    ZERO_LAPS,
    add_final_lap_positions,
    positions_from_lap_page,
    tie_groups_needing_lap_position,
)
from f1pred.data.schema import RACE_ROW_SCHEMA, ClassificationOutcome, WeekendFormat
from f1pred.data.target import FINAL_LAP_POSITION_COLUMN, compute_order, grid_resolved_rows

FIXTURES = Path(__file__).parent / "fixtures" / "ergast"


def _row(driver, laps, grid, *, outcome, position, rnd=1, season=2010):
    return {
        "season": season,
        "round": rnd,
        "race_date": pd.Timestamp("2010-03-28"),
        "circuit_id": "albert_park",
        "weekend_format": WeekendFormat.CONVENTIONAL.value,
        "weekend_format_raw": "conventional",
        "driver_id": driver,
        "driver_code": driver[:3].upper(),
        "constructor_id": f"team_{driver}",
        "grid_position": grid,
        "classified_position": position,
        "position_text": str(position) if outcome is ClassificationOutcome.CLASSIFIED else "R",
        "classification_outcome": outcome.value,
        "status_raw": "Finished" if outcome is ClassificationOutcome.CLASSIFIED else "Collision",
        "laps_completed": laps,
        "points": 0.0,
        "final_lap_position": None,
    }


def _frame(rows):
    df = pd.DataFrame(rows)
    for col, dtype in RACE_ROW_SCHEMA.items():
        df[col] = df[col].astype(dtype)
    return df[list(RACE_ROW_SCHEMA)]


class FakeClient:
    """Records what was asked for, so the tests can assert the request count."""

    def __init__(self, answers=None, fail_on=None):
        self.answers = answers or {}
        self.fail_on = fail_on or set()
        self.calls: list[tuple[int, int, int]] = []

    def lap_positions(self, season, round_, lap_number):
        key = (season, round_, lap_number)
        self.calls.append(key)
        if key in self.fail_on:
            raise RuntimeError("simulated transport failure")
        return self.answers.get(key, {})


class TestParser:
    """The payload shape, read from a real recorded response."""

    def test_parses_a_real_lap_page(self):
        page = json.loads((FIXTURES / "laps_2010_r02_lap09.json").read_text())
        positions = positions_from_lap_page(page)
        assert len(positions) == 19
        assert positions["webber"] == 1
        assert positions["vettel"] == 3
        # Every value is a rank, so the set is contiguous from 1.
        assert sorted(positions.values()) == list(range(1, 20))

    def test_empty_and_malformed_pages_yield_nothing_rather_than_raising(self):
        assert positions_from_lap_page([]) == {}
        assert positions_from_lap_page(None) == {}
        assert positions_from_lap_page([{"Laps": []}]) == {}
        assert positions_from_lap_page([{"Laps": [{"Timings": []}]}]) == {}

    def test_a_timing_missing_driver_or_position_is_skipped_not_guessed(self):
        page = [
            {
                "Laps": [
                    {
                        "Timings": [
                            {"driverId": "alonso", "position": "4"},
                            {"driverId": "massa"},
                            {"position": "6"},
                        ]
                    }
                ]
            }
        ]
        assert positions_from_lap_page(page) == {"alonso": 4}


class TestGroupSelection:
    """Which ties are worth a request — and which cannot be answered at all."""

    def test_untied_retirements_need_no_fetch(self):
        df = _frame([
            _row("a", 58, 1, outcome=ClassificationOutcome.CLASSIFIED, position=1),
            _row("b", 40, 2, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=2),
            _row("c", 30, 3, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=3),
        ])
        assert tie_groups_needing_lap_position(df) == []

    def test_a_tie_on_a_completed_lap_is_selected_once_for_the_group(self):
        df = _frame([
            _row("a", 58, 1, outcome=ClassificationOutcome.CLASSIFIED, position=1),
            _row("b", 12, 2, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=2),
            _row("c", 12, 3, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=3),
        ])
        # One entry, not one per tied driver: the lap is fetched once.
        assert tie_groups_needing_lap_position(df) == [(2010, 1, 12)]

    def test_a_zero_lap_tie_is_never_fetched(self):
        """The rule cannot separate drivers who completed no lap.

        A zero-lap retirement appears in no lap's timing data, so a request
        would spend a round trip to learn nothing.
        """
        df = _frame([
            _row("a", 58, 1, outcome=ClassificationOutcome.CLASSIFIED, position=1),
            _row("b", ZERO_LAPS, 2, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=2),
            _row("c", ZERO_LAPS, 3, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=3),
        ])
        assert tie_groups_needing_lap_position(df) == []

    def test_a_race_with_both_kinds_fetches_only_the_answerable_one(self):
        df = _frame([
            _row("a", 58, 1, outcome=ClassificationOutcome.CLASSIFIED, position=1),
            _row("b", ZERO_LAPS, 2, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=2),
            _row("c", ZERO_LAPS, 3, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=3),
            _row("d", 20, 4, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=4),
            _row("e", 20, 5, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=5),
        ])
        assert tie_groups_needing_lap_position(df) == [(2010, 1, 20)]


class TestEnrichment:
    def _tied_race(self):
        return _frame([
            _row("winner", 58, 1, outcome=ClassificationOutcome.CLASSIFIED, position=1),
            _row("alonso", 16, 9, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=12),
            _row("gutierrez", 16, 14, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=13),
        ])

    def test_fills_the_tied_rows_and_leaves_the_rest_alone(self):
        client = FakeClient({(2010, 1, 16): {"alonso": 12, "gutierrez": 13}})
        out, report = add_final_lap_positions(self._tied_race(), client=client, log=lambda m: None)

        tied = out[out["laps_completed"] == 16].set_index("driver_id")
        assert tied.loc["alonso", FINAL_LAP_POSITION_COLUMN] == 12
        assert tied.loc["gutierrez", FINAL_LAP_POSITION_COLUMN] == 13
        # The finisher was never tied, so nothing was written for it.
        assert pd.isna(out[out["driver_id"] == "winner"][FINAL_LAP_POSITION_COLUMN].iloc[0])
        assert report.rows_resolved == 2

    def test_one_request_per_tie_group_not_per_driver(self):
        client = FakeClient({(2010, 1, 16): {"alonso": 12, "gutierrez": 13}})
        add_final_lap_positions(self._tied_race(), client=client, log=lambda m: None)
        assert client.calls == [(2010, 1, 16)]

    def test_the_fetched_position_actually_changes_the_order(self):
        """The whole point: grid order and lap order disagree here.

        Gutierrez starts behind Alonso but is ahead on the lap they both retire
        on. Without the fetch the grid fallback reverses the true order.
        """
        race = _frame([
            _row("winner", 58, 1, outcome=ClassificationOutcome.CLASSIFIED, position=1),
            _row("alonso", 16, 9, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=13),
            _row("gutierrez", 16, 14, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=12),
        ])
        client = FakeClient({(2010, 1, 16): {"gutierrez": 12, "alonso": 13}})
        enriched, _ = add_final_lap_positions(race, client=client, log=lambda m: None)

        def order(frame):
            ranked = frame.assign(pos=compute_order(frame)).sort_values("pos")
            return list(ranked["driver_id"])

        order_without = order(race)
        order_with = order(enriched)
        assert order_without == ["winner", "alonso", "gutierrez"]  # grid fallback
        assert order_with == ["winner", "gutierrez", "alonso"]  # true track order

    def test_a_failed_fetch_leaves_the_group_null_rather_than_half_filled(self):
        client = FakeClient(fail_on={(2010, 1, 16)})
        out, report = add_final_lap_positions(self._tied_race(), client=client, log=lambda m: None)
        tied = out[out["laps_completed"] == 16]
        assert tied[FINAL_LAP_POSITION_COLUMN].isna().all()
        assert report.failed_groups == ((2010, 1, 16),)
        assert report.rows_resolved == 0
        # Still flagged as grid-resolved, so the weakness stays visible.
        assert len(grid_resolved_rows(out)) == 2

    def test_a_driver_absent_from_the_response_is_counted_not_invented(self):
        client = FakeClient({(2010, 1, 16): {"alonso": 12}})
        out, report = add_final_lap_positions(self._tied_race(), client=client, log=lambda m: None)
        assert report.rows_resolved == 1
        assert report.rows_unresolved_absent == 1
        gut = out[out["driver_id"] == "gutierrez"][FINAL_LAP_POSITION_COLUMN].iloc[0]
        assert pd.isna(gut)

    def test_zero_lap_ties_are_reported_as_permanently_unresolved(self):
        df = _frame([
            _row("winner", 58, 1, outcome=ClassificationOutcome.CLASSIFIED, position=1),
            _row("b", ZERO_LAPS, 12, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=21),
            _row("c", ZERO_LAPS, 15, outcome=ClassificationOutcome.NOT_CLASSIFIED, position=22),
        ])
        client = FakeClient()
        out, report = add_final_lap_positions(df, client=client, log=lambda m: None)
        assert client.calls == []
        assert report.rows_unresolved_zero_lap == 2
        assert len(grid_resolved_rows(out)) == 2

    def test_the_pass_is_idempotent(self):
        client = FakeClient({(2010, 1, 16): {"alonso": 12, "gutierrez": 13}})
        once, _ = add_final_lap_positions(self._tied_race(), client=client, log=lambda m: None)
        twice, _ = add_final_lap_positions(once, client=client, log=lambda m: None)
        pd.testing.assert_frame_equal(once, twice)

    def test_the_input_frame_is_not_mutated(self):
        df = self._tied_race()
        before = df.copy()
        client = FakeClient({(2010, 1, 16): {"alonso": 12, "gutierrez": 13}})
        add_final_lap_positions(df, client=client, log=lambda m: None)
        pd.testing.assert_frame_equal(df, before)


@pytest.mark.network
class TestAgainstLiveApi:
    def test_australia_2010_lap_9_matches_the_recorded_fixture(self):
        from f1pred.data.collect import ErgastClient

        live = ErgastClient().lap_positions(2010, 2, 9)
        recorded = positions_from_lap_page(
            json.loads((FIXTURES / "laps_2010_r02_lap09.json").read_text())
        )
        assert live == recorded
