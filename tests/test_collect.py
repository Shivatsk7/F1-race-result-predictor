"""T-102..T-105b — collection guards. All offline (NFR-05)."""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.data.collect import (
    CollectionError,
    FailureManifest,
    FailureThresholdExceeded,
    IncompletePageError,
    check_season_counts,
    checkpoint_path,
    completed_seasons,
    paginate,
    read_checkpoint,
    seasons_to_collect,
    with_retry,
    write_checkpoint,
)
from tests.fixtures.races import australia_2014, australia_2016


class FakePage:
    """Mimics the API response shape: a row payload plus a reported total."""

    def __init__(self, rows: int, total: int):
        self.rows = rows
        self.total_results = total


def _rows(page: FakePage) -> int:
    return page.rows


class TestRetry:
    """T-105b / FR-111 — transient failure is not absence."""

    def test_succeeds_without_retry(self):
        calls = []
        assert with_retry(lambda: calls.append(1) or "ok", what="x", sleep=lambda _: None) == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise TimeoutError("read timed out")
            return "ok"

        assert with_retry(flaky, what="x", sleep=lambda _: None) == "ok"
        assert state["n"] == 3

    def test_exhausted_retries_raise_rather_than_returning_empty(self):
        # FR-111: never record this as "no data exists".
        def always_fails():
            raise TimeoutError("read timed out")

        with pytest.raises(CollectionError, match="attempts exhausted"):
            with_retry(always_fails, what="quali 2014", attempts=3, sleep=lambda _: None)

    def test_backoff_is_exponential(self):
        delays: list[float] = []

        def always_fails():
            raise TimeoutError()

        with pytest.raises(CollectionError):
            with_retry(always_fails, what="x", attempts=4, sleep=delays.append)
        assert delays == [1.0, 2.0, 4.0]


class TestPagination:
    """T-105a / FR-110 — completeness asserted, never inferred."""

    def test_single_page_when_total_fits(self):
        pages = paginate(
            lambda limit, offset: FakePage(20, 20),
            what="results", rows_in=_rows, sleep=lambda _: None,
        )
        assert len(pages) == 1

    def test_pages_to_exhaustion(self):
        # 2014 race results: 407 rows at 100/page = 5 pages.
        def fetch(limit, offset):
            return FakePage(min(limit, max(0, 407 - offset)), 407)

        pages = paginate(fetch, what="results 2014", rows_in=_rows, sleep=lambda _: None)
        assert len(pages) == 5
        assert sum(p.rows for p in pages) == 407

    def test_truncated_read_raises(self):
        # The exact failure mode the spike hit: a request for 1000 returns 100
        # and looks successful. Here the server stops yielding rows early.
        def fetch(limit, offset):
            return FakePage(100 if offset == 0 else 0, 407)

        with pytest.raises(IncompletePageError, match="pagination stalled"):
            paginate(fetch, what="results 2014", rows_in=_rows, sleep=lambda _: None)

    def test_2023_sprints_reach_six_not_two(self):
        # 120 sprint result rows = 6 races. A naive single call sees 30 rows
        # and reports 2 races, which is what the spike's first pass did.
        def fetch(limit, offset):
            return FakePage(min(limit, max(0, 120 - offset)), 120)

        pages = paginate(fetch, what="sprints 2023", rows_in=_rows, sleep=lambda _: None)
        assert sum(p.rows for p in pages) // 20 == 6

    def test_shifting_total_raises(self):
        def fetch(limit, offset):
            return FakePage(100, 407 if offset == 0 else 500)

        with pytest.raises(IncompletePageError, match="total changed mid-pagination"):
            paginate(fetch, what="results", rows_in=_rows, sleep=lambda _: None)

    def test_pagination_retries_transient_failures(self):
        state = {"n": 0}

        def fetch(limit, offset):
            state["n"] += 1
            if state["n"] == 1:
                raise TimeoutError("read timed out")
            return FakePage(20, 20)

        pages = paginate(fetch, what="results", rows_in=_rows, sleep=lambda _: None)
        assert len(pages) == 1


class TestFailureManifest:
    """T-104 / FR-104a..c — a gap, not a deletion."""

    def test_failures_are_recorded_with_reason(self):
        m = FailureManifest()
        m.record_success()
        m.record_failure(2014, 7, "session load failed", "2014-06-08")
        assert m.failure_count == 1
        assert m.failures[0].reason == "session load failed"
        assert m.as_dict()["failures"][0]["race_date"] == "2014-06-08"

    def test_run_continues_past_a_single_failure(self):
        m = FailureManifest()
        m.record_failure(2014, 7, "boom")
        for _ in range(20):
            m.record_success()
        m.check_threshold()  # does not raise
        assert m.failure_rate < 0.05

    def test_missing_rounds_are_queryable_as_gaps(self):
        # FR-104b: rolling features need "absent" distinguishable from
        # "did not happen".
        m = FailureManifest()
        m.record_failure(2014, 7, "boom")
        m.record_failure(2014, 11, "boom")
        m.record_failure(2015, 3, "boom")
        assert m.missing_rounds(2014) == {7, 11}
        assert m.missing_rounds(2015) == {3}
        assert m.missing_rounds(2016) == set()

    def test_threshold_breach_aborts(self):
        m = FailureManifest(max_failure_rate=0.10)
        for _ in range(8):
            m.record_success()
        m.record_failure(2014, 1, "boom")
        m.record_failure(2014, 2, "boom")
        with pytest.raises(FailureThresholdExceeded, match="looks complete"):
            m.check_threshold()

    def test_clean_run_summary(self):
        m = FailureManifest()
        for _ in range(5):
            m.record_success()
        assert "no failures" in m.summary()

    def test_empty_manifest_has_zero_rate(self):
        assert FailureManifest().failure_rate == 0.0


class TestSeasonCounts:
    """T-105 / FR-106 — row-count sanity."""

    def test_normal_season_is_not_flagged(self):
        df = pd.concat([australia_2014(), australia_2014().assign(round=2)])
        counts = check_season_counts(df)
        assert len(counts) == 1
        assert not counts[0].is_anomalous
        assert counts[0].warning() is None

    def test_short_race_is_flagged(self):
        full = australia_2014()
        stub = full.head(4).assign(round=2)
        counts = check_season_counts(pd.concat([full, stub]))
        assert counts[0].is_anomalous
        assert "deviates" in counts[0].warning()

    def test_expected_count_adapts_to_field_size(self):
        # 2010 ran 24 cars, recent seasons 20. A fixed constant would fire
        # every season; the modal field size does not.
        df = pd.concat([australia_2014(), australia_2014().assign(round=2)])
        counts = check_season_counts(df)
        assert counts[0].expected_rows == 44

    def test_mode_resists_the_anomaly_the_median_absorbs(self):
        # With a short race present, the median moves toward it and the
        # deviation vanishes. The mode stays on the real field size.
        full = australia_2014()
        stub = full.head(4).assign(round=2)
        df = pd.concat([full, stub])
        per_race = df.groupby("round").size()
        assert int(per_race.median()) == 13  # would imply expected == actual
        assert check_season_counts(df)[0].expected_rows == 44

    def test_one_short_race_in_a_full_season_does_not_trip_it(self):
        # FR-106 is a season-level check by design; this documents the limit.
        races = [australia_2014().assign(round=r) for r in range(1, 21)]
        races.append(australia_2014().head(4).assign(round=21))
        counts = check_season_counts(pd.concat(races))
        assert not counts[0].is_anomalous

    def test_counts_reported_per_season(self):
        df = pd.concat([australia_2014(), australia_2016()])
        counts = check_season_counts(df)
        assert [c.season for c in counts] == [2014, 2016]


class TestCheckpointing:
    """T-102 / T-103 — FR-102, FR-103."""

    def test_write_then_read_round_trips(self, tmp_path):
        df = australia_2014()
        write_checkpoint(df, tmp_path, 2014)
        back = read_checkpoint(tmp_path, 2014)
        assert back is not None
        assert len(back) == len(df)
        assert back["driver_id"].tolist() == df["driver_id"].tolist()

    def test_missing_checkpoint_reads_as_none(self):
        assert read_checkpoint("/nonexistent", 1999) is None

    def test_second_run_refetches_nothing(self, tmp_path):
        seasons = [2010, 2011, 2012]
        assert seasons_to_collect(tmp_path, seasons) == seasons
        for s in seasons:
            write_checkpoint(australia_2014(), tmp_path, s)
        assert seasons_to_collect(tmp_path, seasons) == []

    def test_resume_skips_only_completed_seasons(self, tmp_path):
        seasons = [2010, 2011, 2012, 2013]
        write_checkpoint(australia_2014(), tmp_path, 2010)
        write_checkpoint(australia_2014(), tmp_path, 2012)
        assert completed_seasons(tmp_path, seasons) == {2010, 2012}
        assert seasons_to_collect(tmp_path, seasons) == [2011, 2013]

    def test_interrupted_write_leaves_no_usable_checkpoint(self, tmp_path):
        # The partial file must not be mistaken for a checkpoint.
        partial = checkpoint_path(tmp_path, 2014).with_suffix(".parquet.partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"truncated garbage")
        assert seasons_to_collect(tmp_path, [2014]) == [2014]
        assert read_checkpoint(tmp_path, 2014) is None
