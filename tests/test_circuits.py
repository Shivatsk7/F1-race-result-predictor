"""T-101b — declared circuit attributes (FR-109)."""

from __future__ import annotations

import pytest

from f1pred.reference.circuits import (
    CIRCUITS,
    CircuitType,
    OvertakingTier,
    UnknownCircuitError,
    assert_all_declared,
    get,
)


class TestCoverage:
    def test_every_corpus_circuit_is_declared(self):
        # Enumerated from the source for 2010-2025, not from memory.
        assert len(CIRCUITS) == 35

    def test_every_entry_has_both_attributes(self):
        # FR-109 names circuit type and overtaking tier as the minimum.
        for cid, c in CIRCUITS.items():
            assert isinstance(c.circuit_type, CircuitType), cid
            assert isinstance(c.overtaking_tier, OvertakingTier), cid

    def test_ids_are_self_consistent(self):
        for cid, c in CIRCUITS.items():
            assert c.circuit_id == cid

    def test_all_three_types_are_represented(self):
        types = {c.circuit_type for c in CIRCUITS.values()}
        assert types == set(CircuitType)

    def test_all_tiers_are_represented(self):
        tiers = {c.overtaking_tier for c in CIRCUITS.values()}
        assert tiers == set(OvertakingTier)


class TestLookup:
    def test_known_circuit_resolves(self):
        monaco = get("monaco")
        assert monaco.circuit_type is CircuitType.STREET
        assert monaco.overtaking_tier is OvertakingTier.VERY_HARD

    def test_permanent_circuit_resolves(self):
        assert get("monza").circuit_type is CircuitType.PERMANENT
        assert get("monza").overtaking_tier is OvertakingTier.EASY

    def test_hybrid_circuit_resolves(self):
        assert get("albert_park").circuit_type is CircuitType.HYBRID

    def test_missing_entry_raises_rather_than_defaulting(self):
        # FR-109: absence is an error. A default would flow into FR-505's
        # segmentation and FR-602's overtaking model unnoticed.
        with pytest.raises(UnknownCircuitError, match="no declared entry"):
            get("nurburgring_nordschleife")

    def test_corpus_check_names_every_missing_circuit(self):
        with pytest.raises(UnknownCircuitError, match="zolder"):
            assert_all_declared(["monaco", "zolder", "kyalami"])

    def test_corpus_check_passes_on_declared_ids(self):
        assert_all_declared(["monaco", "monza", "spa"])

    def test_corpus_check_accepts_the_full_declared_set(self):
        assert_all_declared(CIRCUITS.keys())


class TestTierOrdering:
    def test_tiers_are_ordered(self):
        assert OvertakingTier.EASY.rank < OvertakingTier.MODERATE.rank
        assert OvertakingTier.MODERATE.rank < OvertakingTier.HARD.rank
        assert OvertakingTier.HARD.rank < OvertakingTier.VERY_HARD.rank

    def test_monaco_is_the_hardest_declared(self):
        worst = max(CIRCUITS.values(), key=lambda c: c.overtaking_tier.rank)
        assert worst.circuit_id == "monaco"
