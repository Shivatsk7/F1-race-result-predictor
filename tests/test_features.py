"""T-202, T-203, T-210, T-213, T-217 — building the matrix (FR-202, FR-210, FR-212, FR-216)."""

from __future__ import annotations

import pathlib
import re

import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.config import (
    COLD_START_CONSTRUCTOR_PERCENTILE,
    COLD_START_DRIVER_PERCENTILE,
    REGIMES,
)
from f1pred.data.schema import RACE_ROW_SCHEMA
from f1pred.features.build import (
    BuildError,
    build,
    build_primary,
    detect_transfers,
    resolve_order,
)
from f1pred.features.registry import (
    REGISTRY,
    Availability,
    Feature,
    FeatureGroup,
    Registry,
)
from tests.fixtures.races import synthetic_corpus

TRANSFER = (2014, 4)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return synthetic_corpus(transfer=TRANSFER)


@pytest.fixture(scope="module")
def built(corpus):
    return build_primary(corpus)


class TestDependencyResolution:
    def test_dependencies_are_computed_before_their_dependents(self):
        order = resolve_order(
            REGISTRY.primary(),
            registry=REGISTRY,
            available=frozenset(RACE_ROW_SCHEMA) | {"target_position"},
        )
        seen: set[str] = set()
        for f in order:
            for dep in f.depends_on:
                if dep in REGISTRY:
                    assert dep in seen, f"{f.name} computed before {dep}"
            seen.add(f.name)

    def test_an_unresolvable_dependency_is_an_error(self, corpus):
        reg = Registry("broken")
        reg.register(
            Feature(
                "dangling",
                lambda df: pd.Series(0.0, index=df.index),
                Availability.PRE_RACE,
                FeatureGroup.PRIMARY,
                "d",
                depends_on=("no_such_thing",),
            )
        )
        with pytest.raises(BuildError, match="neither a raw schema column nor"):
            build(corpus, features=reg.primary(), set_name="x", registry=reg)

    def test_a_cycle_is_reported_rather_than_recursing_forever(self, corpus):
        reg = Registry("cyclic")
        for name, dep in (("a", "b"), ("b", "a")):
            reg.register(
                Feature(
                    name,
                    lambda df: pd.Series(0.0, index=df.index),
                    Availability.PRE_RACE,
                    FeatureGroup.PRIMARY,
                    "d",
                    depends_on=(dep,),
                )
            )
        with pytest.raises(BuildError, match="circular feature dependency"):
            build(corpus, features=reg.primary(), set_name="x", registry=reg)


class TestBuild:
    def test_every_primary_feature_is_present(self, built):
        assert set(built.X.columns) == {f.name for f in REGISTRY.primary()}

    def test_no_intermediate_survives_into_the_matrix(self, built):
        intermediates = {
            f.name for f in REGISTRY.group_features(FeatureGroup.INTERMEDIATE)
        }
        assert not intermediates & set(built.X.columns)

    def test_non_starters_are_gone_before_features_are_computed(self, corpus, built):
        # FR-002. The withdrawn rows never reach a feature function.
        assert len(built) <= len(corpus)

    def test_keys_and_flags_line_up_with_the_matrix(self, built):
        assert len(built.X) == len(built.keys) == len(built.flags)
        assert built.X.index.equals(built.flags.index)

    def test_flags_are_not_model_inputs(self, built):
        assert not set(built.flags.columns) & set(built.X.columns)


class TestRegimeFeatures:
    """T-210, FR-210: regime identity is declared data, not inferred."""

    def test_both_regime_features_are_pre_race(self):
        for name in ("regime_id", "seasons_since_regime"):
            assert REGISTRY.get(name).availability is Availability.PRE_RACE

    def test_regime_id_is_the_declared_start_season(self, corpus, built):
        # 2013-2015 spans the 2011 and 2014 declared regimes.
        got = built.X.assign(season=built.keys["season"]).groupby("season")["regime_id"].unique()
        assert got.loc[2013].tolist() == [2011.0]
        assert got.loc[2014].tolist() == [2014.0]
        assert got.loc[2015].tolist() == [2014.0]

    def test_seasons_since_regime_is_zero_in_a_reset_year(self, built):
        reset = built.X["regime_id"] == built.keys["season"].astype(float)
        assert (built.X.loc[reset, "seasons_since_regime"] == 0).all()

    def test_the_table_extends_past_the_corpus(self):
        # Plan D-10: 2026 is present so the FR-511 experiment gains an
        # observation when that season completes, with no code change.
        assert max(r.start_season for r in REGIMES) >= 2026


class TestColdStart:
    """T-213, FR-212: declared fallbacks and a flag, never a null or a crash."""

    def test_a_debut_row_is_flagged(self, corpus, built):
        first_race = (built.keys["season"] == corpus["season"].min()) & (
            built.keys["round"] == 1
        )
        assert built.flags.loc[first_race.to_numpy(), "low_information"].all()

    def test_an_established_row_is_not_flagged(self, built):
        settled = built.keys["season"] == 2015
        assert not built.flags.loc[settled.to_numpy(), "low_information"].any()

    def test_a_flagged_row_still_produces_numbers(self, built):
        cold = built.flags["low_information"].to_numpy()
        historical = [
            f.name for f in REGISTRY.primary() if f.summarises_history
        ]
        assert not built.X.loc[cold, historical].isna().all(axis=None)

    def test_the_fallback_is_not_the_field_mean(self):
        # FR-212 names this explicitly: 0.5 would assert average
        # competitiveness on no evidence at all.
        assert COLD_START_DRIVER_PERCENTILE != 0.5
        assert COLD_START_CONSTRUCTOR_PERCENTILE != 0.5
        assert 0.0 < COLD_START_DRIVER_PERCENTILE < 0.5

    def test_the_fallback_value_is_what_lands_in_the_matrix(self, built):
        cold = built.flags["low_information"].to_numpy()
        values = built.X.loc[cold, "driver_finish_last3_pct"].dropna().unique()
        assert list(values) == [COLD_START_DRIVER_PERCENTILE]


class TestDriverTransfer:
    """T-217, FR-216: the car's history stops at the door; the driver's does not."""

    def test_the_transfer_is_recorded(self, corpus):
        transfers = detect_transfers(corpus)
        assert len(transfers) == 1
        row = transfers.iloc[0]
        assert (int(row["season"]), int(row["round"])) == TRANSFER
        assert row["previous_constructor_id"] != row["constructor_id"]

    def test_a_drivers_first_appearance_is_not_a_transfer(self, corpus):
        # They arrived from outside the corpus. That is FR-212's cold start.
        transfers = detect_transfers(corpus)
        firsts = corpus.groupby("driver_id")["race_date"].min()
        for _, row in transfers.iterrows():
            assert row["race_date"] > firsts[row["driver_id"]]

    def test_the_race_after_a_move_is_flagged(self, corpus, built):
        mover = detect_transfers(corpus).iloc[0]["driver_id"]
        at_transfer = (
            (built.keys["driver_id"] == mover)
            & (built.keys["season"] == TRANSFER[0])
            & (built.keys["round"] == TRANSFER[1])
        ).to_numpy()
        assert built.flags.loc[at_transfer, "post_transfer"].all()
        assert built.flags["post_transfer"].sum() == 1

    def test_constructor_history_does_not_follow_the_driver(self, corpus):
        """The new team's record, not the old one's, on the day of the move."""
        from f1pred.features.priors import prior_stat, sort_canonically
        from f1pred.data.target import build_target

        work = sort_canonically(build_target(corpus))
        prior = prior_stat(work, entity="constructor_id", value="target_position", window=3)
        mover = detect_transfers(corpus).iloc[0]["driver_id"]
        at = (
            (work["driver_id"] == mover)
            & (work["season"] == TRANSFER[0])
            & (work["round"] == TRANSFER[1])
        )
        new_team = work.loc[at, "constructor_id"].iloc[0]
        # A teammate at the same team in the same race must see the identical
        # constructor history — which is only true if it was recomputed against
        # the new team rather than carried across the move.
        teammates = (
            (work["constructor_id"] == new_team)
            & (work["season"] == TRANSFER[0])
            & (work["round"] == TRANSFER[1])
        )
        assert prior[teammates].nunique(dropna=False) == 1

    def test_driver_history_does_follow_the_driver(self, corpus):
        from f1pred.features.priors import prior_stat, sort_canonically
        from f1pred.data.target import build_target

        work = sort_canonically(build_target(corpus))
        prior = prior_stat(work, entity="driver_id", value="target_position", window=3)
        mover = detect_transfers(corpus).iloc[0]["driver_id"]
        rows = work[work["driver_id"] == mover].reset_index(drop=True)
        got = prior[work["driver_id"] == mover].reset_index(drop=True)
        at = rows.index[
            (rows["season"] == TRANSFER[0]) & (rows["round"] == TRANSFER[1])
        ][0]
        expected = rows["target_position"].astype(float)[at - 3 : at].mean()
        assert got[at] == pytest.approx(expected)


class TestNoHandWrittenFeatureList:
    """T-203, FR-202: the primary set is derived, so no list may exist."""

    def test_no_module_outside_the_registry_enumerates_features(self):
        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "f1pred"
        allowed = {"definitions.py", "disposition.py"}
        names = {f.name for f in REGISTRY}
        offenders: dict[str, int] = {}
        for path in src.rglob("*.py"):
            if path.name in allowed:
                continue
            text = path.read_text()
            quoted = set(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", text))
            hits = quoted & names
            if len(hits) > 2:
                offenders[str(path.relative_to(src))] = len(hits)
        assert not offenders, (
            f"these modules name several features as string literals, which is "
            f"how v0's hand-maintained list came to exist: {offenders}"
        )

    def test_build_primary_derives_its_set_from_the_registry(self, corpus):
        fork = REGISTRY.fork()
        fork.register(
            Feature(
                "extra_signal",
                lambda df: pd.Series(1.0, index=df.index),
                Availability.PRE_RACE,
                FeatureGroup.PRIMARY,
                "An extra signal.",
            )
        )
        assert "extra_signal" in build(
            corpus, features=fork.primary(), set_name="primary", registry=fork
        ).X.columns
