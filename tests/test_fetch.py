"""T-106 — the API client and season fetcher.

Everything here is offline and driven by payloads recorded from the live API
(`tests/fixtures/ergast/`), except the two tests marked `network`, which exist
to catch upstream drift and are excluded from the default run (NFR-04, NFR-05).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from f1pred.data.collect import (
    FailureManifest,
    FailureThresholdExceeded,
    collect_corpus,
    collect_season,
    frame_from_rows,
    merge_paged_races,
    read_checkpoint,
    rows_from_race,
    rows_in_page,
)
from f1pred.data.schema import (
    RACE_ROW_SCHEMA,
    ClassificationOutcome,
    SchemaError,
    WeekendFormat,
    validate_frame,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ergast"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


class FixturePage(list):
    """A recorded page: race payloads plus the total the API reported."""

    def __init__(self, races, total_results: int):
        super().__init__(races)
        self.total_results = total_results


def australia_2016_payload() -> dict:
    return _load("race_2016_r01.json")["races"][0]


def silverstone_2021_payload() -> dict:
    return _load("race_2021_r10.json")["races"][0]


class FakeClient:
    """Stands in for `ErgastClient`, recording what was asked for."""

    def __init__(self, pages: dict[int, list[FixturePage]], formats: dict[int, dict[int, str]]):
        self._pages = pages
        self._formats = formats
        self.seasons_requested: list[int] = []

    def race_result_pages(self, season: int):
        self.seasons_requested.append(season)
        return self._pages[season]

    def event_formats(self, season: int) -> dict[int, str]:
        return self._formats[season]


def _client_for_2016() -> FakeClient:
    race = australia_2016_payload()
    return FakeClient(
        pages={2016: [FixturePage([race], total_results=22)]},
        formats={2016: {1: "conventional"}},
    )


def _season_races(n: int) -> list[dict]:
    """`n` rounds cloned from the recorded race.

    Seasons are cloned to a realistic length on purpose: in a two-race season a
    single failure is a 50% failure rate and trips FR-104c's threshold, which
    would be a fact about the fixture rather than about the code.
    """
    races = []
    for rnd in range(1, n + 1):
        race = dict(australia_2016_payload(), round=str(rnd))
        race["Results"] = [dict(r) for r in race["Results"]]
        races.append(race)
    return races


# --------------------------------------------------------------------------


class TestRowMapping:
    """FR-001..003a, FR-105, FR-108 — payload to canonical row."""

    def test_every_entry_becomes_a_row(self):
        rows = rows_from_race(
            australia_2016_payload(), weekend_format_raw="conventional"
        )
        assert len(rows) == 22

    def test_frame_satisfies_the_canonical_schema(self):
        df = frame_from_rows(
            rows_from_race(australia_2016_payload(), weekend_format_raw="conventional")
        )
        assert list(df.columns) == list(RACE_ROW_SCHEMA)
        validate_frame(df)  # raises on any dtype, null or duplicate-key defect

    def test_chronology_is_recorded(self):
        # FR-105: season, round and race date establish strict ordering.
        df = frame_from_rows(
            rows_from_race(australia_2016_payload(), weekend_format_raw="conventional")
        )
        assert df["season"].unique().tolist() == [2016]
        assert df["round"].unique().tolist() == [1]
        assert df["race_date"].unique().tolist() == [pd.Timestamp("2016-03-20")]
        assert df["circuit_id"].unique().tolist() == ["albert_park"]

    def test_grid_is_the_grid_started_from(self):
        # FR-108: never inferred from a qualifying classification.
        df = frame_from_rows(
            rows_from_race(australia_2016_payload(), weekend_format_raw="conventional")
        ).set_index("driver_id")
        assert df.at["rosberg", "grid_position"] == 2
        assert df.at["hamilton", "grid_position"] == 1

    def test_withdrawal_is_marked_by_the_source_sentinel_not_lap_count(self):
        # FR-002: Kvyat never started (positionText 'W'); Gutierrez retired on
        # lap 16. Both are kept here — exclusion is the target layer's job — but
        # they must be distinguishable, which only the sentinel does.
        df = frame_from_rows(
            rows_from_race(australia_2016_payload(), weekend_format_raw="conventional")
        ).set_index("driver_id")
        assert df.at["kvyat", "classification_outcome"] == (
            ClassificationOutcome.WITHDRAWN.value
        )
        assert df.at["kvyat", "laps_completed"] == 0
        assert df.at["gutierrez", "classification_outcome"] == (
            ClassificationOutcome.NOT_CLASSIFIED.value
        )

    def test_raw_status_is_kept_verbatim(self):
        # FR-003: ~30 distinct values a season; collapsing them at collection
        # time destroys what FR-752 needs.
        df = frame_from_rows(
            rows_from_race(australia_2016_payload(), weekend_format_raw="conventional")
        ).set_index("driver_id")
        assert df.at["rosberg", "status_raw"] == "Finished"
        assert df.at["gutierrez", "status_raw"] == "Collision"
        # Kvyat's raw status says "Retired" while his sentinel says withdrawn.
        # Keeping both is the point: a status-string rule gets this row wrong.
        assert df.at["kvyat", "status_raw"] == "Retired"

    def test_classification_outcome_comes_from_the_sentinel(self):
        # FR-003a on the recorded 2016 field: the classified tail is classified
        # regardless of what its status string says.
        df = frame_from_rows(
            rows_from_race(australia_2016_payload(), weekend_format_raw="conventional")
        )
        classified = df[df["position_text"].str.isdigit()]
        assert set(classified["classification_outcome"]) == {
            ClassificationOutcome.CLASSIFIED.value
        }

    def test_final_lap_position_is_left_unpopulated(self):
        # FR-001b's tie-break is fetched separately and only where a tie exists
        # (T-101e). It must be null here, not zero and not the grid.
        df = frame_from_rows(
            rows_from_race(australia_2016_payload(), weekend_format_raw="conventional")
        )
        assert df["final_lap_position"].isna().all()

    def test_conventional_weekend_format(self):
        rows = rows_from_race(
            australia_2016_payload(), weekend_format_raw="conventional"
        )
        assert rows[0]["weekend_format"] == WeekendFormat.CONVENTIONAL.value
        assert rows[0]["weekend_format_raw"] == "conventional"

    def test_2021_sprint_weekend_maps_to_sprint_sets_grid(self):
        # FR-108: in 2021 the sprint set the Grand Prix grid.
        rows = rows_from_race(silverstone_2021_payload(), weekend_format_raw="sprint")
        assert rows[0]["weekend_format"] == WeekendFormat.SPRINT_SETS_GRID.value
        assert rows[0]["weekend_format_raw"] == "sprint"

    def test_unrecognised_weekend_format_raises(self):
        # FR-108: upstream has renamed this vocabulary three times. A new value
        # must stop the run, not quietly become CONVENTIONAL.
        with pytest.raises(SchemaError, match="unrecognised weekend format"):
            rows_from_race(australia_2016_payload(), weekend_format_raw="sprint_v4")

    def test_unrecognised_position_sentinel_raises(self):
        race = australia_2016_payload()
        race["Results"][0]["positionText"] = "Q"
        with pytest.raises(SchemaError, match="unrecognised positionText"):
            rows_from_race(race, weekend_format_raw="conventional")

    def test_empty_results_raise_rather_than_yielding_a_small_race(self):
        race = dict(australia_2016_payload(), Results=[])
        with pytest.raises(Exception, match="not a small race"):
            rows_from_race(race, weekend_format_raw="conventional")


class TestPageAssembly:
    """FR-104a, FR-110 — the API paginates by result row, not by race."""

    def test_rows_in_page_counts_result_rows_not_races(self):
        page = FixturePage([australia_2016_payload()], total_results=22)
        assert rows_in_page(page) == 22
        assert len(page) == 1

    def test_a_race_split_across_pages_is_reassembled_whole(self):
        # Recorded from the live 2016 season read: round 5 arrives as 12 rows
        # on page 1 and 10 on page 2. Storing either alone would be a partial
        # race, which FR-104a forbids because it silently shrinks the field.
        raw = _load("season_2016_boundary_round5.json")
        pages = [FixturePage(p["races"], p["total_results"]) for p in raw]
        assert [rows_in_page(p) for p in pages] == [12, 10]

        races = merge_paged_races(pages)
        assert set(races) == {5}
        assert len(races[5]["Results"]) == 22
        drivers = [r["Driver"]["driverId"] for r in races[5]["Results"]]
        assert len(set(drivers)) == 22

    def test_merge_does_not_mutate_the_input_payloads(self):
        raw = _load("season_2016_boundary_round5.json")
        pages = [FixturePage(p["races"], p["total_results"]) for p in raw]
        merge_paged_races(pages)
        assert len(pages[0][0]["Results"]) == 12


class TestCollectSeason:
    """FR-101..106 — one season, whole races only."""

    def test_happy_path(self):
        client = _client_for_2016()
        manifest = FailureManifest()
        df, report = collect_season(2016, client=client, manifest=manifest)
        assert report.rows == 22
        assert report.races == 1
        assert report.scheduled_races == 1
        assert report.failed_rounds == ()
        assert manifest.failure_count == 0
        validate_frame(df)

    def test_a_broken_race_is_excluded_whole_and_recorded_as_a_gap(self):
        # FR-104a/b: no partial rows survive, and the round stays queryable as
        # missing rather than as a race that never happened.
        races = _season_races(20)
        races[6]["Results"][3] = dict(races[6]["Results"][3], positionText="Q")

        client = FakeClient(
            pages={2016: [FixturePage(races, total_results=440)]},
            formats={2016: {r: "conventional" for r in range(1, 21)}},
        )
        manifest = FailureManifest()
        df, report = collect_season(2016, client=client, manifest=manifest)

        assert report.failed_rounds == (7,)
        assert 7 not in set(df["round"])
        assert len(df) == 19 * 22  # not 19 races plus a fragment of the 20th
        assert manifest.missing_rounds(2016) == {7}
        assert "unrecognised positionText" in manifest.failures[0].reason
        assert manifest.failures[0].race_date == "2016-03-20"

    def test_a_round_absent_from_the_schedule_fails_rather_than_defaulting(self):
        # FR-108: an unknown weekend format is an error. Defaulting would assert
        # that no sprint happened.
        races = _season_races(20)
        formats = {r: "conventional" for r in range(1, 21)}
        del formats[12]
        client = FakeClient(
            pages={2016: [FixturePage(races, total_results=440)]},
            formats={2016: formats},
        )
        manifest = FailureManifest()
        df, report = collect_season(2016, client=client, manifest=manifest)
        assert report.failed_rounds == (12,)
        assert 12 not in set(df["round"])
        assert "absent from the 2016 event schedule" in manifest.failures[0].reason

    def test_a_scheduled_round_with_no_results_is_a_gap(self):
        client = FakeClient(
            pages={2016: [FixturePage(_season_races(19), total_results=418)]},
            formats={2016: {r: "conventional" for r in range(1, 21)}},
        )
        manifest = FailureManifest()
        df, report = collect_season(2016, client=client, manifest=manifest)
        assert report.failed_rounds == (20,)
        assert report.scheduled_races == 20
        assert manifest.failures[0].reason == (
            "scheduled round returned no race results"
        )

    def test_threshold_breach_aborts_the_season(self):
        # FR-104c: continuing past widespread failure yields a corpus that looks
        # complete and is not.
        races = _season_races(20)
        for rnd in (15, 17, 19):
            races[rnd - 1]["Results"][0] = dict(
                races[rnd - 1]["Results"][0], positionText="Q"
            )
        client = FakeClient(
            pages={2016: [FixturePage(races, total_results=440)]},
            formats={2016: {r: "conventional" for r in range(1, 21)}},
        )
        with pytest.raises(FailureThresholdExceeded, match="looks complete"):
            collect_season(2016, client=client, manifest=FailureManifest())

    def test_season_frame_carries_every_round_it_collected(self):
        formats = {r: "conventional" for r in range(1, 21)}
        formats[2] = "sprint"
        client = FakeClient(
            pages={2016: [FixturePage(_season_races(20), total_results=440)]},
            formats={2016: formats},
        )
        df, report = collect_season(2016, client=client, manifest=FailureManifest())
        assert sorted(df["round"].unique()) == list(range(1, 21))
        assert report.rows == 440
        assert set(df.loc[df["round"] == 2, "weekend_format"]) == {
            WeekendFormat.SPRINT_SETS_GRID.value
        }
        assert set(df.loc[df["round"] == 3, "weekend_format"]) == {
            WeekendFormat.CONVENTIONAL.value
        }


class TestCollectCorpus:
    """FR-102, FR-103 — checkpointing and resume."""

    def test_writes_a_checkpoint_per_season(self, tmp_path):
        client = _client_for_2016()
        reports, manifest = collect_corpus(
            [2016], root=tmp_path, client=client, log=lambda _: None
        )
        assert len(reports) == 1
        back = read_checkpoint(tmp_path, 2016)
        assert back is not None and len(back) == 22
        assert manifest.failure_count == 0

    def test_second_run_refetches_nothing(self, tmp_path):
        collect_corpus(
            [2016], root=tmp_path, client=_client_for_2016(), log=lambda _: None
        )
        second = _client_for_2016()
        reports, _ = collect_corpus(
            [2016], root=tmp_path, client=second, log=lambda _: None
        )
        assert second.seasons_requested == []
        assert reports[0].from_checkpoint
        assert reports[0].rows == 22


@pytest.mark.network
class TestLiveApi:
    """Guards against upstream drift. Excluded from the default offline run."""

    def test_live_race_payload_matches_the_recorded_fixture(self):
        from f1pred.data.collect import ErgastClient

        client = ErgastClient()
        pages = client.race_result_pages(2016)
        races = merge_paged_races(pages)
        live = races[1]
        recorded = australia_2016_payload()
        assert len(live["Results"]) == len(recorded["Results"])
        assert set(live["Results"][0]) >= {
            "position",
            "positionText",
            "grid",
            "laps",
            "status",
            "points",
            "Driver",
            "Constructor",
        }
        df = frame_from_rows(rows_from_race(live, weekend_format_raw="conventional"))
        validate_frame(df)

    def test_live_schedule_reports_the_2021_sprint_weekends(self):
        from f1pred.data.collect import ErgastClient

        formats = ErgastClient().event_formats(2021)
        assert formats[10] == "sprint"
        assert sorted(v for v in set(formats.values())) == ["conventional", "sprint"]
