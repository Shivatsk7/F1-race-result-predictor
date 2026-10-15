"""Declared regime data (FR-210, plan D-10) and the exam plan (Constitution IV.4)."""

from __future__ import annotations

import pytest

import json
from datetime import date

from f1pred.config import (
    CORPUS_START_SEASON,
    EXAM_PLAN,
    EXAM_PLAN_PATH,
    REGIMES,
    RegimeSeverity,
    SeasonRole,
    regime_for_season,
    role_for_season,
    seasons_since_regime_start,
    write_exam_plan,
)
from f1pred.corpus import CorpusWindow


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


class TestExamPlan:
    """Constitution IV.4: fixed season roles, changed only by amendment."""

    def test_every_season_2010_to_2026_has_exactly_one_role(self):
        for season in range(2010, 2027):
            holders = [r for r in SeasonRole if season in EXAM_PLAN.seasons(r)]
            assert len(holders) == 1, f"{season} holds {holders}"
            assert role_for_season(season) is holders[0]

    def test_roles_match_the_ratified_table(self):
        assert EXAM_PLAN.seasons(SeasonRole.LEARNING) == tuple(range(2010, 2020))
        assert EXAM_PLAN.seasons(SeasonRole.PRACTICE) == (2020, 2021, 2022)
        assert EXAM_PLAN.seasons(SeasonRole.FINAL) == (2023, 2024, 2025)
        assert EXAM_PLAN.seasons(SeasonRole.BONUS) == (2026,)

    def test_practice_exam_meets_the_three_season_minimum(self):
        # IV.4c: the pooled practice exam satisfies IV.2's holdout floor.
        assert len(EXAM_PLAN.seasons(SeasonRole.PRACTICE)) >= 3

    def test_roles_unchanged_after_the_2026_rollover(self):
        # The scenario the rolling window failed: on 2027-03-01 the window
        # advances to admit 2026 and drop 2010, and the roles must not follow.
        before = {s: role_for_season(s) for s in range(2010, 2027)}
        window = CorpusWindow(as_of=date(2027, 3, 1)).resolve()
        assert 2026 in window and 2010 not in window
        after = {s: role_for_season(s) for s in range(2010, 2027)}
        assert after == before
        assert role_for_season(2026) is SeasonRole.BONUS

    def test_an_unplanned_season_raises_rather_than_defaulting(self):
        for season in (2009, 2027):
            with pytest.raises(ValueError, match="amend the plan"):
                role_for_season(season)

    def test_spans_are_contiguous_and_ordered(self):
        spans = EXAM_PLAN.spans
        assert spans[0].first == CORPUS_START_SEASON
        for a, b in zip(spans, spans[1:]):
            assert b.first == a.last + 1

    def test_bonus_exam_stays_sealed_until_the_rework_is_complete(self):
        assert EXAM_PLAN.rework_complete is False


class TestExamPlanFile:
    @staticmethod
    def plan_in(path):
        """The file's plan, without the corpus stamp, which carries a date."""
        from f1pred.provenance import PROVENANCE_KEY

        record = json.loads(path.read_text())
        record.pop(PROVENANCE_KEY)
        return record

    def test_committed_file_matches_the_constant(self):
        # Fails if someone edits results/exam_plan.json by hand, or changes
        # EXAM_PLAN without regenerating the file with write_exam_plan().
        assert EXAM_PLAN_PATH.exists(), "run f1pred.config.write_exam_plan()"
        assert self.plan_in(EXAM_PLAN_PATH) == EXAM_PLAN.as_dict()

    def test_writer_round_trips(self, tmp_path):
        target = write_exam_plan(tmp_path / "exam_plan.json")
        assert self.plan_in(target) == EXAM_PLAN.as_dict()

