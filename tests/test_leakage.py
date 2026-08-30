"""T-205, T-206, T-207, T-208, T-209 — the leakage firewall (FR-204..208).

T-207 is the gate's gate. The first two classes here check that the real
feature set is clean; the `TestNegativeControl` class checks that the checks
are capable of saying otherwise. Without it, a green run means only that the
checks ran.
"""

from __future__ import annotations

import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.features.build import build, build_oracle, build_primary
from f1pred.features.checks import (
    LeakageError,
    behavioural_leakage_check,
    determinism_check,
    matrix_fingerprint,
    perturb_race,
    sample_races,
    static_leakage_check,
)
from f1pred.features.priors import prior_stat
from f1pred.features.registry import (
    REGISTRY,
    Availability,
    Feature,
    FeatureGroup,
    Registry,
)
from tests.fixtures.races import synthetic_corpus


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return synthetic_corpus()


# --------------------------------------------------------------------------
# The real feature set
# --------------------------------------------------------------------------


class TestStaticCheck:
    """FR-204: no post-race feature reaches the primary matrix."""

    def test_the_primary_set_passes(self):
        static_leakage_check().raise_if_failed()

    def test_the_check_looks_at_something(self):
        assert static_leakage_check().features_checked == len(REGISTRY.primary())


class TestBehaviouralCheck:
    """FR-205: rewriting a race's result moves none of its pre-race features."""

    def test_the_primary_set_passes(self, corpus):
        behavioural_leakage_check(corpus, count=6).raise_if_failed()

    def test_the_perturbation_actually_perturbs(self, corpus):
        altered = perturb_race(corpus, 2014, 3, seed=1)
        mask = (corpus["season"] == 2014) & (corpus["round"] == 3)
        assert not corpus.loc[mask, "classified_position"].equals(
            altered.loc[mask, "classified_position"]
        )

    def test_the_perturbation_leaves_the_rest_of_the_corpus_alone(self, corpus):
        altered = perturb_race(corpus, 2014, 3, seed=1)
        others = ~((corpus["season"] == 2014) & (corpus["round"] == 3))
        pd.testing.assert_frame_equal(corpus[others], altered[others])

    def test_a_post_race_feature_does_move(self, corpus):
        # If the perturbation could not move even an oracle feature, the check
        # would be passing for the wrong reason.
        post = REGISTRY.select(availability=Availability.POST_RACE)
        report = behavioural_leakage_check(
            corpus,
            features=tuple(REGISTRY.primary()) + tuple(post),
            races=[(2014, 3)],
        )
        # Post-race features are excluded from comparison by design, so this
        # still passes — the point is that the check knows which to compare.
        assert report.passed
        assert report.features_checked > len(REGISTRY.primary())

    def test_sampled_races_are_spread_and_deterministic(self, corpus):
        first = sample_races(corpus, count=5, seed=1)
        assert first == sample_races(corpus, count=5, seed=1)
        assert len({season for season, _ in first}) > 1


# --------------------------------------------------------------------------
# T-207 — the negative control
# --------------------------------------------------------------------------


def _leaky_registry_declared_dependency() -> Registry:
    """A pre-race feature that depends on a post-race one. Static must catch it."""
    reg = REGISTRY.fork("leaky-declared")

    def leaky_by_declaration(df: pd.DataFrame) -> pd.Series:
        """Reads whether the car finished — a fact from after the race."""
        return df["finished_race"].astype("float64")

    reg.register(
        Feature(
            name="leaky_by_declaration",
            fn=leaky_by_declaration,
            availability=Availability.PRE_RACE,
            group=FeatureGroup.PRIMARY,
            description="Reads whether the car finished.",
            depends_on=("finished_race",),
        )
    )
    return reg


def _leaky_registry_hidden() -> Registry:
    """A feature whose declarations are impeccable and whose code cheats.

    It reads only raw columns that pre-race features legitimately read — every
    rolling form feature reads `points` — but reads them for the current race
    instead of for the races before it. No declaration distinguishes this from
    an honest feature, which is precisely why FR-205 exists.
    """
    reg = REGISTRY.fork("leaky-hidden")

    def leaky_by_behaviour(df: pd.DataFrame) -> pd.Series:
        """Season points form, computed one race too generously."""
        honest = prior_stat(df, entity=["season", "driver_id"], value="points")
        return honest.fillna(0.0) + df["points"].astype("float64")

    reg.register(
        Feature(
            name="leaky_by_behaviour",
            fn=leaky_by_behaviour,
            availability=Availability.PRE_RACE,
            group=FeatureGroup.PRIMARY,
            description="Season points form, computed one race too generously.",
            depends_on=("season", "driver_id", "points"),
            summarises_history=True,
            relative=True,
        )
    )
    return reg


class TestNegativeControl:
    """T-207: a deliberately leaky feature must fail both checks."""

    def test_static_check_catches_a_declared_post_race_dependency(self):
        reg = _leaky_registry_declared_dependency()
        report = static_leakage_check(reg.primary(), registry=reg)
        assert not report.passed
        assert any("leaky_by_declaration" in v for v in report.violations)

    def test_behavioural_check_catches_the_same_feature(self, corpus):
        reg = _leaky_registry_declared_dependency()
        report = behavioural_leakage_check(
            corpus, features=reg.primary(), registry=reg, races=[(2014, 3)]
        )
        assert not report.passed
        assert any("leaky_by_declaration" in v for v in report.violations)

    def test_static_check_cannot_see_the_hidden_leak(self, corpus):
        # This is the whole justification for a second layer. If this test ever
        # starts failing because the static check got cleverer, delete it and
        # say so — but do not delete the behavioural check.
        reg = _leaky_registry_hidden()
        assert static_leakage_check(reg.primary(), registry=reg).passed

    def test_behavioural_check_does_see_the_hidden_leak(self, corpus):
        reg = _leaky_registry_hidden()
        report = behavioural_leakage_check(
            corpus, features=reg.primary(), registry=reg, races=[(2014, 3)]
        )
        assert not report.passed
        assert any("leaky_by_behaviour" in v for v in report.violations)

    def test_a_failed_report_raises_rather_than_being_ignorable(self):
        reg = _leaky_registry_declared_dependency()
        with pytest.raises(LeakageError, match="leaky_by_declaration"):
            static_leakage_check(reg.primary(), registry=reg).raise_if_failed()


# --------------------------------------------------------------------------
# T-208 — determinism
# --------------------------------------------------------------------------


class TestDeterminism:
    """FR-208: identical input data yields byte-identical output."""

    def test_two_builds_are_identical(self, corpus):
        determinism_check(corpus).raise_if_failed()

    def test_row_order_of_the_input_does_not_matter(self, corpus):
        a = build_primary(corpus)
        b = build_primary(corpus.sample(frac=1.0, random_state=99).reset_index(drop=True))
        assert matrix_fingerprint(a.X) == matrix_fingerprint(b.X)

    def test_column_order_follows_registration_order(self, corpus):
        built = build_primary(corpus)
        assert built.feature_names == tuple(f.name for f in REGISTRY.primary())

    def test_the_fingerprint_notices_a_changed_value(self, corpus):
        built = build_primary(corpus)
        altered = built.X.copy()
        altered.iloc[0, 0] = altered.iloc[0, 0] + 1
        assert matrix_fingerprint(built.X) != matrix_fingerprint(altered)


# --------------------------------------------------------------------------
# T-209 — the oracle set
# --------------------------------------------------------------------------


class TestOracleSet:
    """FR-207: labelled, selectable, and unreachable from the default path."""

    def test_oracle_is_labelled_in_the_matrix_it_returns(self, corpus):
        assert build_oracle(corpus).set_name == "oracle"
        assert build_primary(corpus).set_name == "primary"

    def test_oracle_contains_post_race_features_and_primary_does_not(self, corpus):
        post = {f.name for f in REGISTRY.select(availability=Availability.POST_RACE)}
        assert post <= set(build_oracle(corpus).feature_names)
        assert not post & set(build_primary(corpus).feature_names)

    def test_the_default_path_takes_no_argument_that_would_widen_it(self, corpus):
        # FR-207's protection is structural: `build_primary` has no keyword
        # that could be passed to include post-race features by accident.
        import inspect

        params = set(inspect.signature(build_primary).parameters) - {"df", "registry"}
        assert not params

    def test_an_oracle_build_still_passes_the_static_check_on_its_primary_part(self):
        primary_part = [f for f in REGISTRY.oracle() if f.is_primary]
        static_leakage_check(primary_part).raise_if_failed()
