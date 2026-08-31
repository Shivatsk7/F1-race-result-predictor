"""T-212 — the sprint race as a pre-race feature (FR-211, FR-110).

The sprint runs on Saturday, so its result is knowable before Sunday's lights
out. What needs proving is not that — it is that the *absence* of a sprint
survives the whole pipeline as an absence. FR-211 forbids a neutral fill,
because a neutral value is a claim about a race that did not happen, and 305 of
the corpus's 329 weekends did not have one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.data.collect import (
    SprintReport,
    add_sprint_results,
    merge_paged_sprints,
    sprint_rounds_expected,
    sprint_rows_in_page,
)
from f1pred.data.schema import RACE_ROW_SCHEMA, validate_frame
from f1pred.features.build import build_primary
from f1pred.features.checks import behavioural_leakage_check, static_leakage_check
from f1pred.features.registry import REGISTRY, Availability
from tests.fixtures.races import synthetic_corpus

FIXTURES = Path(__file__).parent / "fixtures" / "ergast"
SPRINTS = ((2014, 2), (2015, 5))


def sprint_2023_r04() -> dict:
    return json.loads((FIXTURES / "sprint_2023_r04.json").read_text())["sprints"][0]


class FakeSprintClient:
    """Serves recorded sprint pages; records which seasons were asked for."""

    def __init__(self, pages: dict[int, list], fail: set[int] | None = None):
        self._pages = pages
        self._fail = fail or set()
        self.seasons_requested: list[int] = []

    def sprint_result_pages(self, season: int):
        self.seasons_requested.append(season)
        if season in self._fail:
            raise RuntimeError("upstream unavailable")
        return self._pages.get(season, [])


def _payload_for(season: int, rnd: int, drivers: list[str]) -> dict:
    return {
        "season": str(season),
        "round": str(rnd),
        "SprintResults": [
            {
                "position": str(i),
                "positionText": str(i),
                "grid": str(len(drivers) - i + 1),
                "status": "Finished",
                "Driver": {"driverId": d},
            }
            for i, d in enumerate(drivers, start=1)
        ],
    }


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return synthetic_corpus(sprint_rounds=SPRINTS)


@pytest.fixture(scope="module")
def collected(corpus) -> pd.DataFrame:
    drivers = sorted(corpus["driver_id"].unique())
    client = FakeSprintClient(
        pages={
            season: [[_payload_for(season, rnd, drivers)]]
            for season, rnd in SPRINTS
        }
    )
    out, _ = add_sprint_results(corpus, client=client, log=lambda _: None)
    return out


class TestPayloadHandling:
    def test_a_sprint_split_across_pages_is_reassembled(self):
        # FR-104a/FR-110: the API paginates by result row, so a sprint's field
        # can straddle a page boundary exactly as a race's does.
        payload = sprint_2023_r04()
        half = len(payload["SprintResults"]) // 2
        first = dict(payload, SprintResults=payload["SprintResults"][:half])
        second = dict(payload, SprintResults=payload["SprintResults"][half:])
        merged = merge_paged_sprints([[first], [second]])
        assert len(merged[4]) == len(payload["SprintResults"]) == 20

    def test_row_counting_uses_the_sprint_payload_key(self):
        # The race endpoint calls it `Results`; counting that here would report
        # zero rows on a full page and defeat the FR-110 assertion.
        assert sprint_rows_in_page([sprint_2023_r04()]) == 20

    def test_the_recorded_fixture_is_a_real_sprint(self):
        payload = sprint_2023_r04()
        assert payload["SprintResults"][0]["Driver"]["driverId"] == "perez"
        assert int(payload["round"]) == 4


class TestExpectedRounds:
    def test_sprint_weekends_are_read_from_the_declared_format(self, corpus):
        assert sprint_rounds_expected(corpus) == set(SPRINTS)

    def test_a_corpus_with_no_sprints_expects_none(self):
        assert sprint_rounds_expected(synthetic_corpus()) == set()

    def test_no_request_is_made_for_a_corpus_with_no_sprints(self):
        client = FakeSprintClient(pages={})
        _, report = add_sprint_results(
            synthetic_corpus(), client=client, log=lambda _: None
        )
        assert client.seasons_requested == []
        assert report == SprintReport()


class TestCollection:
    def test_only_sprint_seasons_are_fetched(self, corpus):
        drivers = sorted(corpus["driver_id"].unique())
        client = FakeSprintClient(
            pages={s: [[_payload_for(s, r, drivers)]] for s, r in SPRINTS}
        )
        add_sprint_results(corpus, client=client, log=lambda _: None)
        assert client.seasons_requested == [2014, 2015]

    def test_the_result_lands_on_the_right_rows(self, collected):
        filled = collected[collected["sprint_position"].notna()]
        assert {
            (int(s), int(r)) for s, r in zip(filled["season"], filled["round"])
        } == set(SPRINTS)

    def test_conventional_weekends_keep_their_null(self, collected):
        conventional = collected[collected["weekend_format"] == "conventional"]
        assert conventional["sprint_position"].isna().all()
        assert conventional["sprint_grid_position"].isna().all()

    def test_the_frame_still_satisfies_the_schema(self, collected):
        validate_frame(collected)
        for column in ("sprint_position", "sprint_grid_position"):
            assert RACE_ROW_SCHEMA[column] == "Int64"

    def test_a_failed_season_leaves_nulls_and_is_recorded(self, corpus):
        drivers = sorted(corpus["driver_id"].unique())
        client = FakeSprintClient(
            pages={s: [[_payload_for(s, r, drivers)]] for s, r in SPRINTS},
            fail={2014},
        )
        out, report = add_sprint_results(corpus, client=client, log=lambda _: None)
        assert report.failed_seasons == (2014,)
        failed_rows = (out["season"] == 2014) & (out["round"] == 2)
        assert out.loc[failed_rows, "sprint_position"].isna().all()
        # Half a season is worse than none, so the rest is still collected.
        assert out[(out["season"] == 2015) & (out["round"] == 5)][
            "sprint_position"
        ].notna().all()

    def test_a_declared_sprint_with_no_results_is_reported_not_reconciled(self, corpus):
        drivers = sorted(corpus["driver_id"].unique())
        client = FakeSprintClient(
            pages={2014: [[_payload_for(2014, 2, drivers)]], 2015: []}
        )
        _, report = add_sprint_results(corpus, client=client, log=lambda _: None)
        assert report.rounds_missing_results == ((2015, 5),)

    def test_a_sprint_the_corpus_did_not_expect_is_also_reported(self, corpus):
        drivers = sorted(corpus["driver_id"].unique())
        client = FakeSprintClient(
            pages={
                2014: [[_payload_for(2014, 2, drivers), _payload_for(2014, 3, drivers)]],
                2015: [[_payload_for(2015, 5, drivers)]],
            }
        )
        _, report = add_sprint_results(corpus, client=client, log=lambda _: None)
        assert report.rounds_unexpected == ((2014, 3),)

    def test_a_driver_who_missed_the_grand_prix_is_skipped(self, corpus):
        client = FakeSprintClient(
            pages={2014: [[_payload_for(2014, 2, ["nobody_in_the_corpus"])]]}
        )
        out, report = add_sprint_results(corpus, client=client, log=lambda _: None)
        assert report.rows_filled == 0
        assert out["sprint_position"].isna().all()

    def test_the_input_frame_is_not_mutated(self, corpus, collected):
        assert corpus["sprint_position"].isna().all()


class TestFeature:
    """FR-211: registered pre-race, explicitly null, never imputed."""

    def test_both_sprint_features_are_pre_race(self):
        for name in ("sprint_finish_pct", "sprint_places_gained_pct"):
            assert REGISTRY.get(name).availability is Availability.PRE_RACE

    def test_they_are_declared_never_to_be_imputed(self):
        for name in ("sprint_finish_pct", "sprint_places_gained_pct"):
            assert REGISTRY.get(name).never_impute

    def test_a_sprint_weekend_produces_a_value(self, collected):
        built = build_primary(collected)
        on_sprint = (
            (built.keys["season"] == SPRINTS[0][0])
            & (built.keys["round"] == SPRINTS[0][1])
        ).to_numpy()
        assert built.X.loc[on_sprint, "sprint_finish_pct"].notna().all()

    def test_a_conventional_weekend_stays_null(self, collected):
        built = build_primary(collected)
        conventional = (built.keys["round"] == 1).to_numpy()
        assert built.X.loc[conventional, "sprint_finish_pct"].isna().all()

    def test_the_cold_start_fallback_does_not_invent_a_sprint(self, collected):
        # A rookie on a conventional weekend gets FR-212's fallbacks for their
        # form features and nothing at all for the sprint — which is the whole
        # point of FR-211's "explicit rather than imputed".
        built = build_primary(collected)
        cold = built.flags["low_information"].to_numpy()
        assert built.X.loc[cold, "sprint_finish_pct"].isna().all()

    def test_places_gained_is_signed_the_right_way(self, collected):
        built = build_primary(collected)
        # The fixture reverses the grid, so the sprint winner started last and
        # must therefore rank top on progress as well as on finishing position.
        on_sprint = (
            (built.keys["season"] == SPRINTS[0][0])
            & (built.keys["round"] == SPRINTS[0][1])
        ).to_numpy()
        rows = built.X.loc[on_sprint]
        best_finish = rows["sprint_finish_pct"].idxmax()
        assert rows.loc[best_finish, "sprint_places_gained_pct"] == 1.0

    def test_the_leakage_checks_still_pass_with_sprint_features(self, collected):
        static_leakage_check().raise_if_failed()
        behavioural_leakage_check(collected, count=4).raise_if_failed()
