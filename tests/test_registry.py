"""T-201, T-203, T-211 — the feature registry and what it refuses (FR-201, FR-202, FR-209)."""

from __future__ import annotations

import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401  (populates the registry)
from f1pred.features.registry import (
    REGISTRY,
    Availability,
    Feature,
    FeatureGroup,
    Registry,
    RegistryError,
    feature,
)


def _fresh() -> Registry:
    return Registry("test")


def _fn(df: pd.DataFrame) -> pd.Series:
    """A trivial feature."""
    return pd.Series(0.0, index=df.index)


class TestDeclaration:
    def test_decorator_registers_under_the_function_name(self):
        reg = _fresh()

        @feature(Availability.PRE_RACE, registry=reg)
        def some_feature(df):
            """What it measures."""
            return pd.Series(dtype="float64")

        assert reg.get("some_feature").description == "What it measures."

    def test_a_feature_needs_a_docstring(self):
        reg = _fresh()
        with pytest.raises(RegistryError, match="no description"):

            @feature(Availability.PRE_RACE, registry=reg)
            def undocumented(df):
                return pd.Series(dtype="float64")

    def test_duplicate_names_are_refused(self):
        reg = _fresh()
        reg.register(Feature("x", _fn, Availability.PRE_RACE, FeatureGroup.PRIMARY, "d"))
        with pytest.raises(RegistryError, match="already registered"):
            reg.register(
                Feature("x", _fn, Availability.PRE_RACE, FeatureGroup.PRIMARY, "d")
            )

    def test_a_feature_may_not_shadow_a_raw_column(self):
        # `depends_on` resolves against raw columns and features alike, so a
        # shared name makes every dependency on it ambiguous.
        reg = _fresh()
        with pytest.raises(RegistryError, match="collides with a raw schema column"):
            reg.register(
                Feature(
                    "grid_position", _fn, Availability.PRE_RACE, FeatureGroup.PRIMARY, "d"
                )
            )


class TestRelativeConstraint:
    """FR-209: a historical-performance feature is relative, or says why not."""

    def test_absolute_historical_feature_is_refused(self):
        reg = _fresh()
        with pytest.raises(RegistryError, match="neither relative nor justified"):
            reg.register(
                Feature(
                    "team_pace",
                    _fn,
                    Availability.PRE_RACE,
                    FeatureGroup.PRIMARY,
                    "Absolute team pace.",
                    summarises_history=True,
                )
            )

    def test_a_written_justification_admits_it(self):
        reg = _fresh()
        f = reg.register(
            Feature(
                "team_pace",
                _fn,
                Availability.PRE_RACE,
                FeatureGroup.PRIMARY,
                "Absolute team pace.",
                summarises_history=True,
                relative_justification="No contemporaneous field exists for it.",
            )
        )
        assert f.relative_justification

    def test_intermediates_are_exempt(self):
        # They are stepping stones to the relative form, not model inputs.
        reg = _fresh()
        reg.register(
            Feature(
                "team_pace",
                _fn,
                Availability.PRE_RACE,
                FeatureGroup.INTERMEDIATE,
                "Absolute team pace.",
                summarises_history=True,
            )
        )
        assert "team_pace" in reg

    def test_cannot_be_relative_and_justify_not_being_relative(self):
        reg = _fresh()
        with pytest.raises(RegistryError, match="Only one can be true"):
            reg.register(
                Feature(
                    "team_pace",
                    _fn,
                    Availability.PRE_RACE,
                    FeatureGroup.PRIMARY,
                    "d",
                    summarises_history=True,
                    relative=True,
                    relative_justification="because",
                )
            )


class TestDerivedSets:
    """FR-202: the primary set is derived, never hand-written."""

    def test_primary_set_is_pre_race_and_primary_group_only(self):
        for f in REGISTRY.primary():
            assert f.availability is Availability.PRE_RACE, f.name
            assert f.group is FeatureGroup.PRIMARY, f.name

    def test_no_post_race_feature_is_primary(self):
        post = REGISTRY.select(availability=Availability.POST_RACE)
        assert post, "the oracle set needs post-race features to mean anything"
        assert not set(post) & set(REGISTRY.primary())

    def test_oracle_set_strictly_contains_the_primary_set(self):
        # FR-207: the oracle is the primary set plus post-race features.
        assert set(REGISTRY.primary()) < set(REGISTRY.oracle())

    def test_intermediates_reach_no_model(self):
        assert not {
            f for f in REGISTRY.group_features(FeatureGroup.INTERMEDIATE)
        } & set(REGISTRY.oracle())

    def test_every_primary_historical_feature_is_relative(self):
        for f in REGISTRY.primary():
            if f.summarises_history:
                assert f.relative or f.relative_justification, f.name

    def test_forking_does_not_touch_the_default_registry(self):
        before = len(REGISTRY)
        fork = REGISTRY.fork()
        fork.register(
            Feature("bolted_on", _fn, Availability.PRE_RACE, FeatureGroup.PRIMARY, "d")
        )
        assert len(REGISTRY) == before
        assert "bolted_on" not in REGISTRY


class TestUnregisteredFeaturesCannotBeBuilt:
    def test_lookup_of_an_unregistered_name_raises(self):
        with pytest.raises(RegistryError, match="no feature named"):
            REGISTRY.get("weather_mixed")
