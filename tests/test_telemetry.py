"""T-214, T-215 — practice pace and tyre-stint data (FR-213, FR-214, FR-215).

Everything here is offline. The client is faked and the lap frames are built by
hand, because what needs testing is the filtering and aggregation, and a
recorded session would make the tests depend on which laps a particular Friday
happened to produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.data.telemetry import (
    LONG_RUN_MIN_LAPS,
    PREFERRED_PRACTICE_ORDER,
    TELEMETRY_START_SEASON,
    TelemetryError,
    aggregate_driver,
    attach,
    collect_practice_corpus,
    collect_practice_season,
    coverage,
    frame_from_rows,
    usable_laps,
    weekend_rows,
)
from f1pred.features.build import build_group, build_primary
from f1pred.features.checks import static_leakage_check
from f1pred.features.registry import REGISTRY, Availability, FeatureGroup
from tests.fixtures.races import synthetic_corpus


def make_laps(
    driver: str = "VER",
    *,
    n: int = 10,
    stint: int = 1,
    base: float = 90.0,
    slope: float = 0.05,
    accurate: bool = True,
    deleted: bool = False,
    track_status: str = "1",
    pit_in: bool = False,
    pit_out: bool = False,
) -> pd.DataFrame:
    """A synthetic stint: lap times drifting by `slope` seconds per lap of life."""
    seconds = [base + slope * i for i in range(n)]
    return pd.DataFrame(
        {
            "Driver": [driver] * n,
            "LapNumber": list(range(1, n + 1)),
            "Stint": [float(stint)] * n,
            "TyreLife": [float(i + 1) for i in range(n)],
            "LapTime": pd.to_timedelta(seconds, unit="s"),
            "IsAccurate": [accurate] * n,
            "Deleted": [deleted] * n,
            "TrackStatus": [track_status] * n,
            "PitInTime": pd.Series(
                [1.0 if pit_in else None] * n, dtype="float64"
            ).pipe(pd.to_timedelta, unit="s"),
            "PitOutTime": pd.Series(
                [1.0 if pit_out else None] * n, dtype="float64"
            ).pipe(pd.to_timedelta, unit="s"),
        }
    )


class FakePracticeClient:
    def __init__(self, sessions, names=None, fail=(), rate_limit_then_ok=False):
        self._sessions = sessions
        self._names = names or {}
        self._fail = set(fail)
        self.calls: list[tuple[int, int, str]] = []
        self._rate_limited = rate_limit_then_ok

    def practice_names(self, season, *, all_sessions=False):
        return self._names.get(season, {})

    def session_laps(self, season, round_, name):
        self.calls.append((season, round_, name))
        if (season, round_, name) in self._fail:
            raise RuntimeError("session unavailable")
        return self._sessions.get((season, round_, name), make_laps())


class TestLapFiltering:
    def test_inaccurate_laps_are_dropped(self):
        assert usable_laps(make_laps(accurate=False)).empty

    def test_deleted_laps_are_dropped(self):
        assert usable_laps(make_laps(deleted=True)).empty

    def test_in_and_out_laps_are_dropped(self):
        assert usable_laps(make_laps(pit_in=True)).empty
        assert usable_laps(make_laps(pit_out=True)).empty

    def test_a_safety_car_lap_is_dropped(self):
        assert usable_laps(make_laps(track_status="14")).empty

    def test_a_yellow_flag_lap_is_kept(self):
        # Deliberate: a yellow is usually in a sector the driver is nowhere
        # near, and excluding every lap that saw one costs about half the data.
        assert len(usable_laps(make_laps(track_status="12"))) == 10

    def test_a_lap_far_off_the_drivers_best_is_dropped(self):
        laps = pd.concat([make_laps(n=5, slope=0.0), make_laps(n=1, base=200.0)])
        assert len(usable_laps(laps)) == 5

    def test_an_empty_frame_survives(self):
        assert usable_laps(pd.DataFrame()).empty


class TestAggregation:
    def test_a_long_stint_yields_a_pace(self):
        laps = usable_laps(make_laps(n=LONG_RUN_MIN_LAPS + 3)).assign(session_name="P2")
        got = aggregate_driver(laps)
        assert got["long_run_laps"] == LONG_RUN_MIN_LAPS + 3
        assert got["long_run_pace_s"] == pytest.approx(
            laps["lap_seconds"].median()
        )

    def test_a_short_stint_yields_no_pace(self):
        laps = usable_laps(make_laps(n=LONG_RUN_MIN_LAPS - 1)).assign(session_name="P2")
        got = aggregate_driver(laps)
        assert np.isnan(got["long_run_pace_s"])
        assert got["long_run_laps"] == 0
        # But the best lap and stint shape are still recorded.
        assert got["best_lap_s"] > 0
        assert got["stints"] == 1

    def test_the_pace_trend_recovers_a_known_slope(self):
        laps = usable_laps(make_laps(n=10, slope=0.08)).assign(session_name="P2")
        assert aggregate_driver(laps)["stint_pace_trend_s_per_lap"] == pytest.approx(
            0.08, abs=1e-9
        )

    def test_stints_are_centred_before_pooling(self):
        # Two stints with the same shape but very different levels. Without
        # centring, the level difference would dominate and the recovered
        # slope would be wrong.
        a = make_laps(n=8, stint=1, base=90.0, slope=0.05)
        b = make_laps(n=8, stint=2, base=95.0, slope=0.05)
        laps = usable_laps(pd.concat([a, b], ignore_index=True)).assign(session_name="P2")
        assert aggregate_driver(laps)["stint_pace_trend_s_per_lap"] == pytest.approx(
            0.05, abs=1e-9
        )

    def test_stints_are_distinguished_across_sessions(self):
        # Stint 1 of FP1 and stint 1 of FP2 are different runs on different
        # tyres; pooling them would invent one long stint that never happened.
        a = make_laps(n=6, stint=1).assign(session_name="P1")
        b = make_laps(n=6, stint=1).assign(session_name="P2")
        laps = pd.concat([usable_laps(a), usable_laps(b)], ignore_index=True)
        laps["session_name"] = ["P1"] * 6 + ["P2"] * 6
        assert aggregate_driver(laps)["stints"] == 2


class TestWeekendRows:
    def test_one_row_per_driver(self):
        sessions = {
            "Practice 2": pd.concat(
                [make_laps("VER", n=8), make_laps("HAM", n=8)], ignore_index=True
            )
        }
        rows = weekend_rows(2023, 5, sessions)
        assert {r["driver_code"] for r in rows} == {"VER", "HAM"}

    def test_a_washed_out_session_still_records_the_drivers(self):
        # "Ran and set no representative lap" is not "no data", and the
        # difference matters to the coverage report.
        sessions = {"Practice 2": make_laps("VER", n=6, accurate=False)}
        rows = weekend_rows(2023, 5, sessions)
        assert len(rows) == 1
        assert rows[0]["driver_code"] == "VER"
        assert np.isnan(rows[0]["long_run_pace_s"])

    def test_the_frame_has_the_declared_dtypes(self):
        df = frame_from_rows(weekend_rows(2023, 5, {"Practice 2": make_laps(n=8)}))
        assert str(df["season"].dtype) == "int64"
        assert str(df["long_run_pace_s"].dtype) == "float64"


class TestCollection:
    def test_only_the_preferred_session_is_read(self):
        client = FakePracticeClient(
            sessions={}, names={2023: {1: ("Practice 2",), 2: ("Practice 1",)}}
        )
        collect_practice_season(2023, client=client, log=lambda _: None)
        assert [c[2] for c in client.calls] == ["Practice 2", "Practice 1"]

    def test_fp2_is_preferred_over_fp1_and_fp3(self):
        assert PREFERRED_PRACTICE_ORDER[0] == "Practice 2"

    def test_a_failing_session_does_not_lose_the_season(self):
        client = FakePracticeClient(
            sessions={},
            names={2023: {1: ("Practice 2",), 2: ("Practice 2",)}},
            fail=[(2023, 1, "Practice 2")],
        )
        frame, report = collect_practice_season(2023, client=client, log=lambda _: None)
        assert len(report.sessions_failed) == 1
        assert report.rounds_collected == 1
        assert not frame.empty

    def test_a_pre_2018_season_is_refused(self):
        client = FakePracticeClient(sessions={}, names={})
        with pytest.raises(TelemetryError, match="precedes the telemetry source"):
            collect_practice_corpus([2017], client=client, log=lambda _: None)

    def test_the_source_start_season_matches_the_requirement(self):
        assert TELEMETRY_START_SEASON == 2018


class TestCoverageAndAttach:
    """FR-214: coverage reported per season, nulls never imputed."""

    @pytest.fixture
    def corpus(self):
        return synthetic_corpus(seasons=(2018, 2019), rounds=3)

    @pytest.fixture
    def practice(self, corpus):
        covered = corpus[(corpus["season"] == 2019)]
        return frame_from_rows(
            [
                {
                    "season": 2019,
                    "round": int(r),
                    "driver_code": str(d),
                    "practice_sessions": 1,
                    "practice_laps": 10,
                    "long_run_pace_s": 90.0 + i,
                    "long_run_laps": 8,
                    "best_lap_s": 88.0,
                    "stint_pace_trend_s_per_lap": 0.05,
                    "longest_stint_laps": 8,
                    "stints": 2,
                }
                for i, (r, d) in enumerate(
                    zip(covered["round"], covered["driver_code"])
                )
            ]
        )

    def test_coverage_is_reported_per_season(self, corpus, practice):
        got = coverage(corpus, practice).set_index("season")
        assert got.loc[2018, "coverage"] == 0.0
        assert got.loc[2019, "coverage"] == 1.0

    def test_attaching_preserves_the_row_count(self, corpus, practice):
        assert len(attach(corpus, practice)) == len(corpus)

    def test_uncovered_rows_keep_their_null(self, corpus, practice):
        merged = attach(corpus, practice)
        assert merged.loc[merged["season"] == 2018, "long_run_pace_s"].isna().all()

    def test_a_duplicated_practice_key_is_caught_not_silently_fanned_out(
        self, corpus, practice
    ):
        doubled = pd.concat([practice, practice], ignore_index=True)
        with pytest.raises(TelemetryError, match="fanned out"):
            attach(corpus, doubled)


class TestFeatureGroup:
    """FR-215: a named group that may never reach the primary set."""

    def test_the_telemetry_features_are_in_their_own_group(self):
        group = REGISTRY.group_features(FeatureGroup.TELEMETRY)
        assert len(group) >= 3
        for f in group:
            assert f.availability is Availability.PRE_RACE
            assert f.never_impute, f.name

    def test_none_of_them_reach_the_primary_set(self):
        primary = {f.name for f in REGISTRY.primary()}
        for f in REGISTRY.group_features(FeatureGroup.TELEMETRY):
            assert f.name not in primary

    def test_the_static_check_rejects_one_placed_in_the_primary_set(self):
        # The gate that makes FR-215 enforceable rather than a convention.
        telemetry = REGISTRY.group_features(FeatureGroup.TELEMETRY)
        report = static_leakage_check(list(REGISTRY.primary()) + list(telemetry))
        assert not report.passed
        assert any("telemetry" in v for v in report.violations)

    def test_the_separate_arm_is_the_primary_set_plus_the_group(self):
        corpus = synthetic_corpus(seasons=(2018, 2019), rounds=3)
        practice = frame_from_rows(
            [
                {
                    "season": int(s),
                    "round": int(r),
                    "driver_code": str(d),
                    "practice_sessions": 1,
                    "practice_laps": 10,
                    "long_run_pace_s": 90.0 + i % 7,
                    "long_run_laps": 8,
                    "best_lap_s": 88.0,
                    "stint_pace_trend_s_per_lap": 0.05,
                    "longest_stint_laps": 8,
                    "stints": 2,
                }
                for i, (s, r, d) in enumerate(
                    zip(corpus["season"], corpus["round"], corpus["driver_code"])
                )
            ]
        )
        arm = build_group(attach(corpus, practice), FeatureGroup.TELEMETRY)
        base = build_primary(corpus)
        assert arm.set_name == "primary+telemetry"
        assert set(base.feature_names) < set(arm.feature_names)
        assert "practice_long_run_pace_pct" in arm.feature_names
