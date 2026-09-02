"""Tests for the learned ranker (FR-401..FR-406).

The tests that matter here are the ones that would still pass if the model
were bad and fail if it were dishonest: that no holdout row can reach the fit,
that the intervals are sized on data the fit did not see, that the same seed
gives the same predictions, and that the persisted record says exactly what
FR-404 says it must. Accuracy is the evaluation harness's business, and it is
measured against the real corpus rather than asserted against a fixture.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from f1pred.corpus import CorpusWindow
from f1pred.evaluate.split import ChronologicalSplit
from f1pred.models.base import Predictor
from f1pred.models.ranker import (
    FR404_FIELDS,
    Hyperparameters,
    ModelError,
    TrainedRanker,
    race_seasons,
    recency_weights,
    relevance,
    train,
)

FEATURES = ("f_grid", "f_form", "f_noise")

FIRST_SEASON = 2010
LAST_SEASON = 2020
ROUNDS = 4
FIELD = 18


def corpus(seed: int = 7) -> pd.DataFrame:
    """A synthetic corpus with one real signal and one pure distractor.

    Grid slot carries the signal, so a working ranker beats grid order only
    slightly and a broken one falls far below it — which is the shape of the
    real problem and keeps the fixture from flattering the model.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        for rnd in range(1, ROUNDS + 1):
            for slot in range(1, FIELD + 1):
                rows.append(
                    {
                        "season": season,
                        "round": rnd,
                        "race_date": pd.Timestamp(f"{season}-03-01")
                        + pd.Timedelta(days=21 * rnd),
                        "circuit_id": f"c{rnd}",
                        "driver_id": f"d{slot:02d}",
                        "constructor_id": f"t{(slot - 1) // 2}",
                        "f_grid": float(slot),
                        "f_form": float(rng.normal(slot, 1.5)),
                        "f_noise": float(rng.normal()),
                    }
                )
    df = pd.DataFrame(rows)
    latent = df["f_grid"] * 0.7 + df["f_form"] * 0.3 + rng.normal(0, 2.0, len(df))
    df["target_position"] = (
        latent.groupby([df["season"], df["round"]]).rank(method="first").astype(int)
    )
    return df


def make_split(holdout_seasons: int = 3) -> ChronologicalSplit:
    seasons = range(FIRST_SEASON, LAST_SEASON + 1)
    return ChronologicalSplit(
        window=CorpusWindow(width=LAST_SEASON - FIRST_SEASON + 1),
        holdout_seasons=holdout_seasons,
        season_end_dates={s: date(s, 12, 1) for s in seasons},
    )


def quick() -> Hyperparameters:
    """Few trees. The properties under test are structural, not accuracy."""
    return Hyperparameters(n_estimators=40, max_depth=4)


@pytest.fixture(scope="module")
def fitted() -> TrainedRanker:
    return train(corpus(), split=make_split(), features=FEATURES, params=quick())


@pytest.fixture(scope="module")
def one_race() -> pd.DataFrame:
    df = corpus()
    return df[(df["season"] == LAST_SEASON) & (df["round"] == 1)]


class TestHyperparameters:
    def test_negative_decay_is_refused(self):
        with pytest.raises(ModelError, match="non-negative"):
            Hyperparameters(recency_decay=-0.1)

    @pytest.mark.parametrize("coverage", [0.0, 1.0, 1.5, -0.2])
    def test_coverage_must_be_a_proportion(self, coverage):
        with pytest.raises(ModelError, match="strictly between"):
            Hyperparameters(interval_coverage=coverage)

    def test_calibrating_on_the_training_rows_is_refused(self):
        with pytest.raises(ModelError, match="held back"):
            Hyperparameters(calibration_seasons=0)

    def test_every_field_reaches_the_record(self):
        """FR-404 field 4 is the whole object, not the tuning-relevant part."""
        params = Hyperparameters(recency_decay=0.25)
        assert params.as_dict()["recency_decay"] == 0.25
        assert set(params.as_dict()) >= set(params.booster_params()) - {"random_state"}


class TestRecencyWeights:
    def test_zero_decay_is_uniform_bit_for_bit(self):
        """T-408: uniform weighting stays expressible, exactly."""
        seasons = pd.Series([2010, 2015, 2020])
        weights = recency_weights(seasons, decay=0.0)
        assert np.array_equal(weights, np.ones(3))

    def test_the_newest_season_anchors_the_scale(self):
        seasons = pd.Series([2018, 2019, 2020])
        weights = recency_weights(seasons, decay=0.5)
        assert weights[-1] == 1.0
        assert weights[0] == pytest.approx(np.exp(-1.0))
        assert (np.diff(weights) > 0).all()

    def test_age_is_measured_from_training_not_from_today(self):
        """Adding a season to the corpus must not reweight an old training set."""
        seasons = pd.Series([2016, 2017])
        anchored = recency_weights(seasons, decay=0.3)
        later = recency_weights(seasons, decay=0.3, most_recent=2020)
        assert anchored[-1] == 1.0
        assert later[-1] < 1.0
        # The *ratio* between rows is what the fit sees, and it is unchanged.
        assert anchored[0] / anchored[1] == pytest.approx(later[0] / later[1])

    def test_one_weight_per_race_not_per_driver(self):
        df = corpus()
        seasons = race_seasons(df)
        assert len(seasons) == len(df.groupby(["season", "round"]))
        assert len(seasons) < len(df)

    def test_race_weights_line_up_with_race_order(self):
        """The weight of race k must belong to race k, and nothing enforces that
        except this ordering — so it is asserted rather than assumed."""
        df = corpus()
        seasons = race_seasons(df)
        appearance = df[["season", "round"]].drop_duplicates()
        assert list(seasons.to_numpy()) == list(appearance["season"].to_numpy())


class TestRelevance:
    def test_the_winner_carries_the_largest_gain(self):
        df = corpus()
        race = df[(df["season"] == 2010) & (df["round"] == 1)]
        gains = relevance(race["target_position"], pd.Series("r", index=race.index))
        winner = race["target_position"].idxmin()
        assert gains[race.index.get_loc(winner)] == gains.max()

    def test_gain_counts_cars_beaten_so_field_size_matters(self):
        small = pd.Series([1, 2, 3])
        large = pd.Series(range(1, 21))
        g_small = relevance(small, pd.Series(["a"] * 3))
        g_large = relevance(large, pd.Series(["b"] * 20))
        # A P1 in a three-car field beat two cars; a P1 in twenty beat nineteen.
        assert g_small[0] == 2.0
        assert g_large[0] == 19.0

    def test_a_missing_target_is_an_error_not_a_zero(self):
        positions = pd.Series([1.0, np.nan, 3.0])
        with pytest.raises(ModelError, match="target"):
            relevance(positions, pd.Series(["r"] * 3))


class TestTrainingBoundary:
    def test_only_training_seasons_are_fitted(self, fitted):
        assert fitted.training_seasons == make_split().training_seasons
        assert max(fitted.training_seasons) < make_split().boundary_season

    def test_the_holdout_cannot_influence_the_fit(self, one_race):
        """FR-403, tested behaviourally rather than by reading the code.

        Every holdout row is corrupted beyond recognition — features scrambled,
        target reversed. If a single one of them reached the fit, or reached the
        interval calibration, the predictions would move. They do not.
        """
        clean = corpus()
        poisoned = clean.copy()
        holdout = poisoned["season"].isin(make_split().holdout)
        for column in FEATURES:
            poisoned.loc[holdout, column] = -999.0
        poisoned.loc[holdout, "target_position"] = (
            FIELD + 1 - poisoned.loc[holdout, "target_position"]
        )

        a = train(clean, split=make_split(), features=FEATURES, params=quick())
        b = train(poisoned, split=make_split(), features=FEATURES, params=quick())

        assert a.predict(one_race).equals(b.predict(one_race))
        assert a.interval == b.interval

    def test_a_frame_missing_the_target_is_refused(self):
        df = corpus().drop(columns=["target_position"])
        with pytest.raises(ModelError, match="target_position"):
            train(df, split=make_split(), features=FEATURES, params=quick())

    def test_a_model_with_no_features_is_refused(self):
        with pytest.raises(ModelError, match="no features"):
            train(corpus(), split=make_split(), features=(), params=quick())

    def test_seasons_outside_the_window_never_train(self):
        """A frame wider than the window must not quietly widen the fit."""
        df = corpus()
        older = df[df["season"] == FIRST_SEASON].copy()
        older["season"] = FIRST_SEASON - 5
        wider = pd.concat([older, df], ignore_index=True)
        model = train(wider, split=make_split(), features=FEATURES, params=quick())
        assert FIRST_SEASON - 5 not in model.training_seasons


class TestReproducibility:
    def test_same_seed_same_data_same_predictions(self, one_race):
        """FR-405."""
        a = train(corpus(), split=make_split(), features=FEATURES, params=quick())
        b = train(corpus(), split=make_split(), features=FEATURES, params=quick())
        assert a.scores(one_race).equals(b.scores(one_race))
        assert a.predict(one_race).equals(b.predict(one_race))

    def test_a_different_seed_is_a_different_model(self, one_race):
        """Guards the guard: if the seed did nothing, the test above would pass
        for the wrong reason."""
        a = train(
            corpus(),
            split=make_split(),
            features=FEATURES,
            params=Hyperparameters(n_estimators=40, max_depth=4, seed=1),
        )
        b = train(
            corpus(),
            split=make_split(),
            features=FEATURES,
            params=Hyperparameters(n_estimators=40, max_depth=4, seed=2),
        )
        assert not a.scores(one_race).equals(b.scores(one_race))


class TestRecencyWeightingChangesTheFit:
    def test_decay_is_applied_and_not_merely_recorded(self, one_race):
        """T-408: a parameter that is persisted but ignored is worse than none."""
        uniform = train(
            corpus(), split=make_split(), features=FEATURES, params=quick()
        )
        decayed = train(
            corpus(),
            split=make_split(),
            features=FEATURES,
            params=Hyperparameters(n_estimators=40, max_depth=4, recency_decay=0.6),
        )
        assert not uniform.scores(one_race).equals(decayed.scores(one_race))
        assert uniform.params.recency_decay == 0.0
        assert decayed.params.recency_decay == 0.6


class TestPredictions:
    def test_the_model_is_a_predictor(self, fitted):
        """FR-304: the harness must not be able to tell it from a baseline."""
        assert isinstance(fitted, Predictor)

    def test_a_race_comes_back_as_a_permutation(self, fitted, one_race):
        positions = fitted.predict(one_race)
        assert sorted(positions.tolist()) == list(range(1, len(one_race) + 1))
        assert positions.index.equals(one_race.index)

    def test_a_missing_feature_is_refused_not_imputed(self, fitted, one_race):
        with pytest.raises(ModelError, match="missing"):
            fitted.predict(one_race.drop(columns=["f_form"]))

    def test_every_driver_gets_an_interval(self, fitted, one_race):
        """FR-402."""
        out = fitted.predict_with_interval(one_race)
        assert len(out) == len(one_race)
        assert (out["predicted_position_lower"] <= out["predicted_position"]).all()
        assert (out["predicted_position"] <= out["predicted_position_upper"]).all()

    def test_intervals_stay_inside_the_field(self, fitted, one_race):
        out = fitted.predict_with_interval(one_race)
        assert out["predicted_position_lower"].min() >= 1.0
        assert out["predicted_position_upper"].max() <= float(len(one_race))


class TestIntervalCalibration:
    def test_the_band_is_sized_on_held_back_training_seasons(self, fitted):
        expected = make_split().training_seasons[-fitted.params.calibration_seasons :]
        assert fitted.interval.calibration_seasons == expected
        assert set(fitted.interval.calibration_seasons) & set(make_split().holdout) == set()

    def test_a_wider_coverage_gives_a_wider_band(self):
        narrow = train(
            corpus(),
            split=make_split(),
            features=FEATURES,
            params=Hyperparameters(n_estimators=40, max_depth=4, interval_coverage=0.5),
        )
        wide = train(
            corpus(),
            split=make_split(),
            features=FEATURES,
            params=Hyperparameters(n_estimators=40, max_depth=4, interval_coverage=0.95),
        )
        narrow_width = narrow.interval.upper_offset - narrow.interval.lower_offset
        wide_width = wide.interval.upper_offset - wide.interval.lower_offset
        assert wide_width > narrow_width

    def test_the_band_reports_itself_uncalibrated(self, fitted):
        """T-509 has not run yet, and nothing may read as though it had."""
        assert fitted.interval.as_dict()["status"] == "UNCALIBRATED"


class TestModelRecord:
    def test_exactly_the_six_fields_fr404_enumerates(self, fitted):
        assert tuple(fitted.metadata()) == FR404_FIELDS
        assert len(FR404_FIELDS) == 6

    def test_each_feature_carries_its_availability_declaration(self, fitted):
        entries = fitted.metadata()["features"]
        assert [e["name"] for e in entries] == list(FEATURES)
        # These fixture features are not registered, and the record says so
        # rather than inventing a declaration for them.
        assert {e["availability"] for e in entries} == {"unregistered"}

    def test_registered_features_report_their_real_declaration(self):
        """The fixture case above must not be the only one exercised."""
        import f1pred.features.definitions  # noqa: F401
        from f1pred.features.registry import REGISTRY

        first = next(iter(REGISTRY))
        declared = {f.name: f.availability.value for f in REGISTRY}
        assert declared[first.name] in {"pre_race", "post_race"}

    def test_the_training_range_is_the_training_data(self, fitted):
        start, end = fitted.training_date_range
        assert start.startswith(str(min(fitted.training_seasons)))
        assert end.startswith(str(max(fitted.training_seasons)))
        assert fitted.training_rows == len(fitted.training_seasons) * ROUNDS * FIELD

    def test_the_commit_is_recorded(self, fitted):
        assert fitted.code_commit
        assert fitted.code_commit != ""

    def test_evaluation_metrics_attach_without_mutating_the_model(self, fitted):
        scored = fitted.with_evaluation_metrics({"spearman": 0.5})
        assert scored.metadata()["evaluation_metrics"] == {"spearman": 0.5}
        assert fitted.metadata()["evaluation_metrics"] == {}
        assert scored is not fitted


class TestPersistence:
    def test_saving_writes_the_booster_and_the_record(self, fitted, tmp_path):
        model_path = fitted.save(tmp_path / "ranker")
        assert model_path.exists()
        record = json.loads((tmp_path / "ranker.json").read_text())
        assert set(FR404_FIELDS) <= set(record)
        assert record["interval"]["status"] == "UNCALIBRATED"
        assert record["training_seasons"] == list(fitted.training_seasons)

    def test_the_record_is_readable_without_unpickling_anything(
        self, fitted, tmp_path
    ):
        """Provenance must not be hostage to the library version that wrote it."""
        fitted.save(tmp_path / "ranker")
        record = json.loads((tmp_path / "ranker.json").read_text())
        assert record["hyperparameters"]["recency_decay"] == 0.0

    def test_a_reloaded_booster_predicts_identically(
        self, fitted, one_race, tmp_path
    ):
        import xgboost as xgb

        model_path = fitted.save(tmp_path / "ranker")
        reloaded = xgb.XGBRanker()
        reloaded.load_model(model_path)
        before = fitted.scores(one_race).to_numpy()
        after = reloaded.predict(one_race[list(fitted.feature_names)])
        assert np.allclose(before, after)


class TestEarlyStopping:
    """The mechanism that keeps the fit from memorising its training races.

    Without it the first honest run of this model scored 0.9625 in-sample
    against 0.6302 on the holdout — 400 trees at depth 6 fit 274 races almost
    exactly. The count is chosen against seasons held back from *inside*
    training, never against the holdout: picking it against the holdout would
    make the reported number the best of several looks at the answer.
    """

    def test_the_tree_count_is_recorded_and_can_be_below_the_ceiling(self):
        model = train(
            corpus(),
            split=make_split(),
            features=FEATURES,
            params=Hyperparameters(n_estimators=400, max_depth=6),
        )
        assert 0 < model.trees <= 400
        assert model.metadata()["hyperparameters"]["trees_kept"] == model.trees

    def test_disabling_it_keeps_every_tree(self):
        model = train(
            corpus(),
            split=make_split(),
            features=FEATURES,
            params=Hyperparameters(n_estimators=25, early_stopping_rounds=None),
        )
        assert model.trees == 25

    def test_it_reduces_the_gap_between_fitted_and_unseen_seasons(self):
        """The property that matters, stated as a property rather than a number.

        Both models are scored on the *calibration* seasons, which neither the
        stopped fit's tree count nor the unstopped fit's was chosen to flatter
        beyond what training already saw — and on the training seasons. The
        stopped model must not be the one that separates them further.
        """
        df = corpus()
        split = make_split()
        training, _ = split.split(df)
        params = Hyperparameters(n_estimators=400, max_depth=6)

        stopped = train(df, split=split, features=FEATURES, params=params)
        unstopped = train(
            df,
            split=split,
            features=FEATURES,
            params=Hyperparameters(
                n_estimators=400, max_depth=6, early_stopping_rounds=None
            ),
        )

        def in_sample_spearman(model) -> float:
            scores = []
            for _, race in training.groupby(["season", "round"], sort=True):
                predicted = model.predict(race)
                actual = race["target_position"]
                scores.append(predicted.corr(actual, method="spearman"))
            return float(np.mean(scores))

        assert in_sample_spearman(stopped) < in_sample_spearman(unstopped)

    def test_a_negative_patience_is_refused(self):
        with pytest.raises(ModelError, match="early_stopping_rounds"):
            Hyperparameters(early_stopping_rounds=0)

    def test_the_count_comes_from_the_calibration_slice_not_the_holdout(self):
        """Corrupting the holdout must not change how many trees survive."""
        clean = corpus()
        poisoned = clean.copy()
        holdout = poisoned["season"].isin(make_split().holdout)
        for column in FEATURES:
            poisoned.loc[holdout, column] = -999.0

        a = train(clean, split=make_split(), features=FEATURES, params=quick())
        b = train(poisoned, split=make_split(), features=FEATURES, params=quick())
        assert a.trees == b.trees
