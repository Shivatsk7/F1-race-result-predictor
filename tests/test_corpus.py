"""T-108 — CorpusWindow resolution (FR-508, FR-509, FR-510)."""

from __future__ import annotations

from datetime import date

import pytest

from f1pred.config import DEFAULT_WINDOW_WIDTH, MIN_WINDOW_WIDTH
from f1pred.corpus import (
    CorpusWindow,
    WindowTooNarrowError,
    admissible,
    archive_seasons,
    default_window,
)


class TestDefaults:
    def test_default_width_is_sixteen(self):
        assert DEFAULT_WINDOW_WIDTH == 16

    def test_resolves_to_2010_2025_during_2026(self):
        # OQ-6: the owner chose a 2010 start; 16 wide lands on 2010-2025.
        window = default_window(as_of=date(2026, 8, 25))
        assert window.resolve() == tuple(range(2010, 2026))

    def test_resolved_list_travels_with_the_width(self):
        described = default_window(as_of=date(2026, 8, 25)).describe()
        assert described["width"] == 16
        assert described["first_season"] == 2010
        assert described["last_season"] == 2025
        assert len(described["seasons"]) == 16


class TestSeasonBoundary:
    """FR-508: the window advances at both ends as a season completes."""

    END_DATES = {
        2024: date(2024, 12, 8),
        2025: date(2025, 12, 7),
        2026: date(2026, 12, 6),
    }

    def test_before_2026_finale_window_ends_at_2025(self):
        window = CorpusWindow(width=16, as_of=date(2026, 12, 1))
        assert window.resolve(self.END_DATES) == tuple(range(2010, 2026))

    def test_after_2026_finale_window_advances_both_ends(self):
        window = CorpusWindow(width=16, as_of=date(2026, 12, 20))
        resolved = window.resolve(self.END_DATES)
        # 2026 enters, 2010 leaves — both ends move, width is unchanged.
        assert resolved == tuple(range(2011, 2027))
        assert len(resolved) == 16

    def test_width_is_invariant_across_the_boundary(self):
        before = CorpusWindow(width=16, as_of=date(2026, 12, 1)).resolve(self.END_DATES)
        after = CorpusWindow(width=16, as_of=date(2026, 12, 20)).resolve(self.END_DATES)
        assert len(before) == len(after) == 16


class TestInProgressSeasonExcluded:
    """FR-509: an in-progress season provably cannot enter the corpus."""

    def test_current_season_absent_without_end_dates(self):
        window = CorpusWindow(width=16, as_of=date(2026, 8, 25))
        assert 2026 not in window.resolve()

    def test_current_season_absent_when_finale_is_future(self):
        end_dates = {2025: date(2025, 12, 7), 2026: date(2026, 12, 6)}
        window = CorpusWindow(width=16, as_of=date(2026, 8, 25))
        assert 2026 not in window.resolve(end_dates)

    def test_season_admitted_only_after_its_final_round(self):
        end_dates = {2025: date(2025, 12, 7)}
        day_before = CorpusWindow(width=16, as_of=date(2025, 12, 6))
        day_after = CorpusWindow(width=16, as_of=date(2025, 12, 8))
        with pytest.raises(ValueError, match="no season"):
            day_before.resolve(end_dates)
        assert day_after.latest_completed_season(end_dates) == 2025


class TestConfigurableWidth:
    """FR-510: W is configurable, but has a hard floor."""

    def test_narrower_width_is_expressible(self):
        window = CorpusWindow(width=10, as_of=date(2026, 8, 25))
        assert window.resolve() == tuple(range(2016, 2026))

    def test_width_below_floor_fails_loudly(self):
        with pytest.raises(WindowTooNarrowError, match="IV.2 floor"):
            CorpusWindow(width=MIN_WINDOW_WIDTH - 1)

    def test_floor_itself_is_allowed(self):
        window = CorpusWindow(width=MIN_WINDOW_WIDTH, as_of=date(2026, 8, 25))
        assert len(window.resolve()) == MIN_WINDOW_WIDTH

    def test_window_reaching_past_available_data_fails_loudly(self):
        # Never silently truncate: a short corpus must be a deliberate choice.
        window = CorpusWindow(width=30, as_of=date(2026, 8, 25))
        with pytest.raises(WindowTooNarrowError, match="before the earliest"):
            window.resolve()


class TestHelpers:
    def test_admissible_filters_and_preserves_order(self):
        window = default_window(as_of=date(2026, 8, 25))
        assert admissible([2026, 2025, 2009, 2010], window) == (2025, 2010)

    def test_archive_ignores_the_rolling_window(self):
        # FR-511 reads the archive so ageing seasons out cannot destroy
        # transfer observations.
        assert archive_seasons(2026) == tuple(range(2010, 2027))
        assert len(archive_seasons(2026)) > DEFAULT_WINDOW_WIDTH
