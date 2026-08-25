"""T-101a / T-101c / T-101d — the FR-001 target and its validation."""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.data.schema import ClassificationOutcome, SchemaError, classify
from f1pred.data.target import (
    build_target,
    compute_order,
    drop_non_starters,
    validate_against_source,
)
from tests.fixtures.races import australia_2014, australia_2016, bahrain_2022_tail


class TestNonStarters:
    """T-101d / FR-002 — withdrawal vs. lap-1 retirement."""

    def test_australia_2016_keeps_21_starters_and_drops_one_withdrawal(self):
        starters, excluded = drop_non_starters(australia_2016())
        assert len(starters) == 21
        assert len(excluded) == 1
        assert excluded["driver_id"].tolist() == ["kvyat"]

    def test_zero_lap_retirements_are_retained(self):
        # Massa and Kobayashi completed zero laps but did start.
        starters, excluded = drop_non_starters(australia_2014())
        assert len(starters) == 22
        assert excluded.empty
        zero_lap = starters[starters["laps_completed"] == 0]["driver_id"].tolist()
        assert sorted(zero_lap) == ["kobayashi", "massa"]

    def test_exclusion_carries_its_reason(self):
        _, excluded = drop_non_starters(australia_2016())
        assert excluded["status_raw"].tolist() == ["Withdrew"]

    def test_lap_count_alone_cannot_discriminate(self):
        # The point of FR-002: both are zero-lap rows, only the sentinel differs.
        df = australia_2016()
        zero_lap = df[df["laps_completed"] == 0]
        assert len(zero_lap) == 1
        assert zero_lap["position_text"].iloc[0] == "W"


class TestOrdering:
    """T-101a / FR-001 — all four cases."""

    def test_australia_2014_matches_the_source_exactly(self):
        starters, _ = drop_non_starters(australia_2014())
        computed = compute_order(starters)
        pd.testing.assert_series_equal(
            computed.astype("int64"),
            starters["classified_position"].astype("int64"),
            check_names=False,
        )

    def test_disqualification_sorts_last_despite_full_distance(self):
        starters, _ = drop_non_starters(australia_2014())
        computed = compute_order(starters)
        ric = starters.index[starters["driver_id"] == "ricciardo"][0]
        assert computed[ric] == len(starters)
        # He completed more laps than anyone who retired, and still sorts last.
        assert starters.at[ric, "laps_completed"] == 57

    def test_retirements_ordered_by_distance_descending(self):
        starters, _ = drop_non_starters(australia_2014())
        out = starters.assign(pos=compute_order(starters))
        retired = out[out["classification_outcome"]
                      == ClassificationOutcome.NOT_CLASSIFIED.value]
        retired = retired.sort_values("pos")
        laps = retired["laps_completed"].tolist()
        assert laps == sorted(laps, reverse=True)

    def test_ties_broken_by_grid_position(self):
        starters, _ = drop_non_starters(australia_2014())
        out = starters.assign(pos=compute_order(starters))
        tied = out[out["laps_completed"] == 0].sort_values("pos")
        # Massa started 9th, Kobayashi 14th; the better grid slot sorts first.
        assert tied["driver_id"].tolist() == ["massa", "kobayashi"]

    def test_positions_are_a_dense_permutation(self):
        starters, _ = drop_non_starters(australia_2016())
        computed = compute_order(starters)
        assert sorted(computed.tolist()) == list(range(1, len(starters) + 1))

    def test_build_target_adds_column_and_drops_non_starters(self):
        out = build_target(australia_2016())
        assert "target_position" in out.columns
        assert len(out) == 21
        assert "kvyat" not in out["driver_id"].tolist()

    def test_ordering_requires_non_starters_removed(self):
        with pytest.raises(SchemaError, match="non-starters must be removed"):
            compute_order(australia_2016())


class TestNotFinishedIsNotNotClassified:
    """FR-003a — Bahrain 2022 is the case that forces the distinction."""

    def test_late_retirements_stay_classified(self):
        df = bahrain_2022_tail()
        perez = df[df["driver_id"] == "perez"].iloc[0]
        assert perez["status_raw"] == "Fuel pressure"
        assert perez["classification_outcome"] == ClassificationOutcome.CLASSIFIED.value

    def test_classified_retirements_outrank_unclassified_ones(self):
        starters, _ = drop_non_starters(bahrain_2022_tail())
        out = starters.assign(pos=compute_order(starters))
        by_driver = dict(zip(out["driver_id"], out["pos"]))
        # Gasly ran 44 laps, Verstappen 54 — but Gasly was not classified,
        # so he sorts behind both classified retirements regardless.
        assert by_driver["perez"] < by_driver["verstappen"] < by_driver["gasly"]

    def test_a_status_based_rule_would_get_this_wrong(self):
        df = bahrain_2022_tail()
        not_finished = df[df["status_raw"] != "Finished"]
        still_classified = not_finished[
            not_finished["classification_outcome"]
            == ClassificationOutcome.CLASSIFIED.value
        ]
        assert len(still_classified) == 2


class TestSourceValidation:
    """T-101c / FR-001a — validate, never silently override."""

    def test_australia_2014_agrees(self):
        report = validate_against_source(australia_2014())
        assert report.agrees
        assert report.races_checked == 1
        assert report.races_in_agreement == 1
        assert "agrees" in report.summary()

    def test_australia_2016_disagreement_is_reported_not_corrected(self):
        # Gutierrez and Alonso both retired on lap 16. The source places
        # Gutierrez ahead; the FR-001 grid tie-break places Alonso ahead.
        # FR-001a requires this be surfaced, not papered over.
        report = validate_against_source(australia_2016())
        assert not report.agrees
        assert report.races_in_disagreement == 1
        drivers = {d.driver_id for d in report.disagreements}
        assert drivers == {"gutierrez", "alonso"}

    def test_disagreement_records_both_positions(self):
        report = validate_against_source(australia_2016())
        by_driver = {d.driver_id: d for d in report.disagreements}
        assert by_driver["alonso"].source_position == 21
        assert by_driver["alonso"].computed_position == 20
        assert by_driver["gutierrez"].source_position == 20
        assert by_driver["gutierrez"].computed_position == 21

    def test_source_column_is_left_untouched(self):
        df = australia_2016()
        before = df["classified_position"].tolist()
        validate_against_source(df)
        assert df["classified_position"].tolist() == before


class TestClassifySentinels:
    def test_numeric_is_classified(self):
        assert classify("7") is ClassificationOutcome.CLASSIFIED

    @pytest.mark.parametrize(
        "sentinel,expected",
        [
            ("R", ClassificationOutcome.NOT_CLASSIFIED),
            ("N", ClassificationOutcome.NOT_CLASSIFIED),
            ("D", ClassificationOutcome.DISQUALIFIED),
            ("E", ClassificationOutcome.DISQUALIFIED),
            ("W", ClassificationOutcome.WITHDRAWN),
            ("F", ClassificationOutcome.WITHDRAWN),
        ],
    )
    def test_known_sentinels(self, sentinel, expected):
        assert classify(sentinel) is expected

    def test_unknown_sentinel_raises_rather_than_defaulting(self):
        with pytest.raises(SchemaError, match="unrecognised positionText"):
            classify("Z")

    def test_empty_raises(self):
        with pytest.raises(SchemaError, match="empty positionText"):
            classify("  ")
