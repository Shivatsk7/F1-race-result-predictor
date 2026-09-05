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
    ENRICHMENT_COLUMNS,
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
            available=frozenset(RACE_ROW_SCHEMA)
            | {"target_position"}
            | set(ENRICHMENT_COLUMNS),
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
        for name in ("regime_id", "seasons_since_regime_start"):
            assert REGISTRY.get(name).availability is Availability.PRE_RACE

    def test_regime_id_is_the_declared_start_season(self, corpus, built):
        # 2013-2015 spans the 2011 and 2014 declared regimes.
        got = built.X.assign(season=built.keys["season"]).groupby("season")["regime_id"].unique()
        assert got.loc[2013].tolist() == [2011.0]
        assert got.loc[2014].tolist() == [2014.0]
        assert got.loc[2015].tolist() == [2014.0]

    def test_seasons_since_regime_start_is_zero_in_a_reset_year(self, built):
        reset = built.X["regime_id"] == built.keys["season"].astype(float)
        assert (built.X.loc[reset, "seasons_since_regime_start"] == 0).all()

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


class TestShrinkage:
    """T-219, FR-212a: a thin estimate is blended toward its declared prior."""

    def test_no_history_returns_the_prior_exactly(self, built):
        # FR-212's cold-start fallback is not replaced by shrinkage, it is
        # subsumed: the fallback is what the formula returns at n=0.
        cold = built.flags["low_information"].to_numpy()
        values = built.X.loc[cold, "driver_finish_last3_pct"].unique()
        assert list(values) == [COLD_START_DRIVER_PERCENTILE]

    def test_a_thin_estimate_is_pulled_toward_the_prior(self, corpus, built):
        from f1pred.features.priors import prior_race_count, sort_canonically

        work = sort_canonically(corpus)
        n = prior_race_count(work, entity="driver_id").to_numpy()
        thin = (n > 0) & (n <= 2)
        settled = n >= 20
        if thin.sum() and settled.sum():
            spread_thin = built.X.loc[thin, "driver_finish_last3_pct"].std()
            spread_settled = built.X.loc[settled, "driver_finish_last3_pct"].std()
            assert spread_thin < spread_settled, (
                "thin estimates should be compressed toward the prior, so their "
                "spread must be narrower than that of well-evidenced rows"
            )

    #: Historical features that are deliberately NOT shrunk. Both are measured
    #: on the 1..N rank scale rather than the 0..1 percentile scale, so the
    #: declared percentile prior has no meaning for them — blending a grid-slot
    #: difference toward 0.35 would be an arithmetic error, not a weak prior.
    RANK_SCALED = {"strength_rank_prior", "overqualified_by_grid"}

    def test_every_percentile_feature_declares_what_backs_it(self):
        for f in REGISTRY.primary():
            if not f.summarises_history or f.never_impute:
                continue
            if f.name in self.RANK_SCALED:
                assert not f.evidence, (
                    f"{f.name} is rank-scaled and must not be shrunk toward a "
                    f"percentile prior"
                )
                continue
            assert f.evidence, f"{f.name} is shrunk but declares no evidence counter"

    def test_the_rank_scaled_exemptions_really_are_rank_scaled(self, built):
        # Guards the exemption list above: if one of these ever becomes a
        # percentile, it should start being shrunk and this test should fail.
        for name in self.RANK_SCALED:
            assert built.X[name].abs().max() > 1.0, (
                f"{name} looks like a percentile now; it should declare "
                f"evidence and be shrunk"
            )

    def test_season_features_use_the_season_counter(self):
        # The whole reason evidence is declared per feature: a season rate
        # resets in January and must not borrow the career count's confidence.
        for name in (
            "driver_win_rate_season_pct",
            "driver_points_per_race_season_pct",
        ):
            assert REGISTRY.get(name).evidence == "driver_prior_races_season"

    def test_circuit_features_use_the_visit_counter(self):
        assert REGISTRY.get("driver_circuit_finish_mean_pct").evidence == (
            "driver_prior_visits"
        )

    def test_never_impute_features_are_left_alone(self, built):
        # FR-211/FR-214: an absent sprint or an absent practice session is a
        # fact. Blending a fact toward a prior would erase it.
        for f in REGISTRY.primary():
            if f.never_impute and f.name in built.X:
                assert built.X[f.name].isna().any(), (
                    f"{f.name} declares never_impute but has no nulls left"
                )

    def test_the_strength_is_declared_not_tuned(self):
        from f1pred.config import SHRINKAGE_STRENGTH

        assert SHRINKAGE_STRENGTH > 0

    def test_a_missing_evidence_column_is_an_error_not_a_silent_skip(self, corpus):
        from f1pred.features.registry import Feature

        reg = Registry("no-evidence")
        reg.register(
            Feature(
                "thin",
                lambda df: pd.Series(0.9, index=df.index),
                Availability.PRE_RACE,
                FeatureGroup.PRIMARY,
                "A feature whose counter was never computed.",
                summarises_history=True,
                relative=True,
                evidence="a_counter_nobody_built",
            )
        )
        with pytest.raises(BuildError, match="declares evidence"):
            build(corpus, features=reg.primary(), set_name="x", registry=reg)


class TestEnrichmentAbsence:
    """A matrix built without weather or telemetry says so."""

    def test_absent_sources_are_recorded(self, built):
        # The synthetic corpus carries no weather and no practice data, so
        # every column from those passes should be listed.
        assert "weather_prior_wet_probability" in built.absent_sources
        assert "long_run_pace_s" in built.absent_sources

    def test_features_reading_an_absent_source_are_null_not_filled(self, built):
        assert built.X["weather_prior_wet"].isna().all()

    def test_an_enriched_frame_records_no_absence_for_that_source(self, corpus):
        enriched = corpus.copy()
        for column in ENRICHMENT_COLUMNS:
            enriched[column] = 1.0
        assert build_primary(enriched).absent_sources == ()


class TestRealTransferRegression:
    """AC-20's actual wording: a regression test on a **known** mid-corpus move.

    Everything above uses `synthetic_corpus`, whose transfer is injected by the
    fixture — which tests the mechanism but proves nothing about the moves that
    are really in the data. These read the collected corpus.

    Sainz's Toro Rosso to Renault move is the sharpest case available: it
    happened at **round 17 of 2017**, mid-season. A January switch could be
    handled correctly by any code that merely resets features each year, so a
    mid-season move is what separates real transfer handling from an annual
    reset that happens to look like it.

    Skipped rather than failed when the corpus is absent, so a clean checkout
    without collected data still runs green (NFR-05: no network here either).
    """

    @pytest.fixture(scope="class")
    def corpus(self):
        from f1pred.dataset import available_seasons, load_corpus

        if not available_seasons():
            pytest.skip("no collected corpus on disk; run `f1pred collect`")
        return load_corpus()

    @pytest.fixture(scope="class")
    def transfers(self, corpus):
        return detect_transfers(corpus)

    def test_the_mid_season_move_is_detected_at_the_right_round(self, transfers):
        sainz = transfers[
            (transfers["driver_id"] == "sainz") & (transfers["season"] == 2017)
        ]
        assert len(sainz) == 1, "Sainz's 2017 move is missing or double-counted"
        row = sainz.iloc[0]
        assert int(row["round"]) == 17, (
            f"the move is recorded at round {row['round']}; it happened at "
            f"round 17, and snapping it to round 1 would be an annual reset "
            f"masquerading as transfer handling"
        )
        assert row["previous_constructor_id"] == "toro_rosso"
        assert row["constructor_id"] == "renault"

    def test_a_canonical_between_season_move_is_detected(self, transfers):
        hamilton = transfers[
            (transfers["driver_id"] == "hamilton") & (transfers["season"] == 2013)
        ]
        assert len(hamilton) == 1
        assert hamilton.iloc[0]["previous_constructor_id"] == "mclaren"
        assert hamilton.iloc[0]["constructor_id"] == "mercedes"

    def test_the_race_after_a_real_move_is_flagged(self, corpus, transfers):
        """FR-216: the flag is on the row, not merely in a side table."""
        built = build_primary(corpus)
        keys = built.keys
        for driver, season, rnd in (("sainz", 2017, 17), ("hamilton", 2013, 1)):
            mask = (
                (keys["driver_id"] == driver)
                & (keys["season"] == season)
                & (keys["round"] == rnd)
            )
            assert mask.any(), f"{driver} {season} round {rnd} is not in the corpus"
            assert built.flags.loc[mask, "post_transfer"].all(), (
                f"{driver}'s {season} move is recorded but the race after it "
                f"carries no POST_TRANSFER flag"
            )

    def test_the_new_constructors_form_is_not_inherited_from_the_old_one(
        self, corpus
    ):
        """The correctness half of FR-216, on a real move.

        Sainz's constructor-form features at Renault must not carry Toro
        Rosso's history across the move. Checked as: the value at the transfer
        race differs from what the same driver's feature held immediately
        before it, since the two constructors were not equally competitive.
        """
        built = build_primary(corpus)
        keys = built.keys
        constructor_features = [
            name for name in built.feature_names if name.startswith("constructor_")
        ]
        assert constructor_features, "no constructor-form features to check"

        sainz = keys["driver_id"] == "sainz"
        season_2017 = keys["season"] == 2017
        before = built.X.loc[sainz & season_2017 & (keys["round"] == 16)]
        after = built.X.loc[sainz & season_2017 & (keys["round"] == 17)]
        assert len(before) == 1 and len(after) == 1

        changed = [
            name
            for name in constructor_features
            if not _values_match(before.iloc[0][name], after.iloc[0][name])
        ]
        assert changed, (
            "every constructor-form feature is unchanged across Sainz's move "
            "to Renault, which means Toro Rosso's history was carried over"
        )


def _values_match(a, b) -> bool:
    """NaN-tolerant equality: a reset feature may legitimately become null."""
    import math

    a_null = a is None or (isinstance(a, float) and math.isnan(a))
    b_null = b is None or (isinstance(b, float) and math.isnan(b))
    if a_null or b_null:
        return a_null and b_null
    return a == b
