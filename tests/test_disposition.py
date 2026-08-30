"""T-202 — every v0 feature is ported or dropped with a reason (FR-202)."""

from __future__ import annotations

import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.features.disposition import (
    DISPOSITIONS,
    NEW_FEATURES,
    V0_FEATURES,
    Disposition,
    dropped_names,
    ported_names,
)
from f1pred.features.registry import REGISTRY, FeatureGroup


class TestLedgerIsComplete:
    def test_every_v0_feature_has_a_disposition(self):
        assert set(V0_FEATURES) == set(DISPOSITIONS)

    def test_the_ledger_invents_nothing(self):
        assert not set(DISPOSITIONS) - set(V0_FEATURES)

    def test_a_disposition_is_a_port_or_a_reason_but_not_both(self):
        with pytest.raises(ValueError):
            Disposition(ported_as="x", reason="y")
        with pytest.raises(ValueError):
            Disposition()

    def test_every_drop_carries_a_substantive_reason(self):
        for name in dropped_names():
            reason = DISPOSITIONS[name].reason
            assert len(reason) > 60, f"{name}: {reason!r} is not an explanation"


class TestPortsAreReal:
    def test_every_port_names_a_registered_feature(self):
        registered = set(REGISTRY.names())
        missing = set(ported_names()) - registered
        assert not missing, f"ported to features that do not exist: {sorted(missing)}"

    def test_no_two_v0_features_port_to_the_same_name(self):
        ports = ported_names()
        assert len(ports) == len(set(ports))

    def test_every_model_facing_feature_is_accounted_for(self):
        # Either it descends from a v0 feature, or it is listed as new. A
        # feature that appeared from nowhere is as much a gap in the record as
        # a v0 one that vanished.
        model_facing = {
            f.name for f in REGISTRY if f.group is not FeatureGroup.INTERMEDIATE
        }
        unaccounted = model_facing - set(ported_names()) - set(NEW_FEATURES)
        assert not unaccounted, sorted(unaccounted)


class TestTheLeakIsGone:
    def test_the_v0_weather_features_did_not_survive(self):
        for name in ("weather_dry", "weather_wet", "weather_mixed"):
            assert DISPOSITIONS[name].ported_as is None
            assert name not in REGISTRY

    def test_the_reason_names_the_rule_that_forbids_them(self):
        assert "III.2" in DISPOSITIONS["weather_mixed"].reason
