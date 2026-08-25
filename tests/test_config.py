"""Declared regime data (FR-210, plan D-10)."""

from __future__ import annotations

import pytest

from f1pred.config import (
    CORPUS_START_SEASON,
    REGIMES,
    RegimeSeverity,
    regime_for_season,
    seasons_since_regime_start,
)


class TestRegimeTable:
    def test_corpus_starts_at_a_declared_regime_boundary(self):
        # OQ-6: 2010 is both the corpus start and a regime boundary.
        assert CORPUS_START_SEASON == 2010
        assert any(r.start_season == CORPUS_START_SEASON for r in REGIMES)

    def test_boundaries_are_irregular(self):
        # FR-210: no fixed period is correct, which is why this is a table.
        starts = [r.start_season for r in REGIMES]
        gaps = {b - a for a, b in zip(starts, starts[1:])}
        assert len(gaps) > 1, f"gaps {gaps} look periodic; the real cycle is not"

    def test_table_is_sorted_and_unique(self):
        starts = [r.start_season for r in REGIMES]
        assert starts == sorted(starts)
        assert len(starts) == len(set(starts))

    def test_extends_past_the_corpus(self):
        # 2026 is declared so the transfer experiment gains an observation
        # with no code change once that season completes.
        assert max(r.start_season for r in REGIMES) >= 2026


class TestRegimeLookup:
    @pytest.mark.parametrize(
        "season,expected_start",
        [
            (2010, 2010),
            (2013, 2011),
            (2014, 2014),
            (2016, 2014),
            (2017, 2017),
            (2021, 2019),
            (2022, 2022),
            (2025, 2022),
            (2026, 2026),
        ],
    )
    def test_resolves_to_the_regime_in_force(self, season, expected_start):
        assert regime_for_season(season).start_season == expected_start

    def test_major_resets_are_marked(self):
        for season in (2014, 2017, 2022):
            assert regime_for_season(season).severity is RegimeSeverity.MAJOR

    def test_seasons_elapsed_is_zero_in_a_reset_year(self):
        assert seasons_since_regime_start(2022) == 0
        assert seasons_since_regime_start(2025) == 3

    def test_undeclared_earlier_season_raises(self):
        # A missing regime is missing data, not a default.
        with pytest.raises(ValueError, match="precedes the first declared regime"):
            regime_for_season(2005)
