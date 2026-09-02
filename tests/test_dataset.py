"""Assembling the experiment frame from checkpoints (FR-304, FR-506, FR-508).

The module under test is the one committed path from the parquet files on disk
to a frame the harness can score, so most of what is checked here is that it
refuses to produce a frame that would misdescribe itself: a corpus quietly
missing a season, a matrix built from different races, a feature wearing a
baseline's name.

Nothing here reads `data/raw/` or touches the network. Every corpus is
synthetic and every parquet file is written into `tmp_path`, because a test
that reads the real corpus passes or fails according to what somebody last
collected.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import f1pred.features.definitions  # noqa: F401
from f1pred.data.collect import write_checkpoint
from f1pred.data.target import build_target
from f1pred.data.telemetry import frame_from_rows, write_practice_checkpoint
from f1pred.data.weather import write_archive
from f1pred.dataset import (
    BASELINE_COLUMNS,
    DatasetError,
    Experiment,
    available_seasons,
    enrich,
    experiment,
    load_corpus,
    prepare,
)
from f1pred.features.build import (
    KEY_COLUMNS,
    TARGET_COLUMN,
    FeatureMatrix,
    build_oracle,
    build_primary,
)
from f1pred.features.priors import sort_canonically
from f1pred.models.baselines import GridOrderBaseline, StandingsOrderBaseline
from tests.fixtures.races import synthetic_corpus

SEASONS = (2018, 2019, 2020)
ROUNDS = 4
CIRCUITS = ("albert_park", "monaco", "monza", "spa")


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return synthetic_corpus(seasons=SEASONS, rounds=ROUNDS, transfer=(2019, 2))


@pytest.fixture(scope="module")
def root(tmp_path_factory, corpus) -> Path:
    """A bare corpus on disk: season checkpoints and nothing else."""
    directory = tmp_path_factory.mktemp("raw")
    for season, frame in corpus.groupby("season", sort=True):
        write_checkpoint(frame.reset_index(drop=True), directory, int(season))
    return directory


@pytest.fixture(scope="module")
def enriched_root(tmp_path_factory, corpus) -> Path:
    """The same corpus, plus a weather archive and practice checkpoints."""
    directory = tmp_path_factory.mktemp("raw_enriched")
    for season, frame in corpus.groupby("season", sort=True):
        write_checkpoint(frame.reset_index(drop=True), directory, int(season))
    write_archive(make_archive(), directory)
    for season in SEASONS:
        write_practice_checkpoint(make_practice(season), directory, season)
    return directory


def make_archive(first_year: int = 2005, last_year: int = 2020) -> pd.DataFrame:
    """A daily archive over every fixture circuit, deep enough for a prior.

    The values are flat on purpose. What the enrichment tests check is that the
    columns arrive attached to the right races, not what the numbers say.
    """
    frames = []
    for circuit in CIRCUITS:
        dates = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "circuit_id": circuit,
                    "date": dates,
                    "precipitation_sum": np.where(
                        np.arange(len(dates)) % 4 == 0, 3.0, 0.0
                    ),
                    "temperature_2m_max": 22.0,
                    "temperature_2m_mean": 17.0,
                    "wind_speed_10m_max": 11.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_practice(season: int, *, drivers: int = 10) -> pd.DataFrame:
    """One practice row per driver per round, in the declared practice schema."""
    rows = [
        {
            "season": season,
            "round": rnd,
            "driver_code": f"D{i:02d}",
            "practice_sessions": 1,
            "practice_laps": 20,
            "long_run_pace_s": 90.0 + i * 0.1,
            "long_run_laps": 12,
            "best_lap_s": 85.0 + i * 0.1,
            "stint_pace_trend_s_per_lap": 0.02,
            "longest_stint_laps": 12,
            "stints": 3,
        }
        for rnd in range(1, ROUNDS + 1)
        for i in range(drivers)
    ]
    return frame_from_rows(rows)


def colliding_builder(df: pd.DataFrame) -> FeatureMatrix:
    """A matrix whose single feature is named after a baseline's raw input."""
    work = sort_canonically(build_target(df))
    return FeatureMatrix(
        set_name="colliding",
        X=pd.DataFrame(
            {"grid_position": work["grid_position"].astype("float64")},
            index=work.index,
        ),
        keys=work[list(KEY_COLUMNS)].copy(),
        flags=pd.DataFrame(index=work.index),
        target=pd.to_numeric(work[TARGET_COLUMN], errors="coerce"),
        feature_names=("grid_position",),
    )


class TestLoadCorpus:
    def test_a_requested_season_without_a_checkpoint_raises(self, root):
        with pytest.raises(DatasetError, match="2021"):
            load_corpus([2018, 2021], root=root)

    def test_the_error_names_the_season_rather_than_returning_a_short_corpus(
        self, root
    ):
        try:
            load_corpus([2018, 2019, 2021], root=root)
        except DatasetError as exc:
            assert "2021" in str(exc)
        else:  # pragma: no cover - the assertion above is the test
            pytest.fail("a missing season was tolerated")

    def test_every_checkpoint_is_loaded_when_no_seasons_are_named(self, root, corpus):
        loaded = load_corpus(root=root)
        assert len(loaded) == len(corpus)
        assert sorted(loaded["season"].unique()) == list(SEASONS)

    def test_the_loaded_corpus_is_in_canonical_order(self, root):
        loaded = load_corpus(root=root)
        assert sort_canonically(loaded).equals(loaded)
        assert loaded["race_date"].is_monotonic_increasing

    def test_seasons_concatenate_rather_than_replace_one_another(self, root, corpus):
        one = load_corpus([2018], root=root)
        two = load_corpus([2018, 2019], root=root)
        assert len(two) == len(one) + int((corpus["season"] == 2019).sum())

    def test_an_explicit_subset_loads_only_those_seasons(self, root):
        loaded = load_corpus([2020, 2018], root=root)
        assert sorted(loaded["season"].unique()) == [2018, 2020]

    def test_a_season_asked_for_twice_is_read_once(self, root):
        assert len(load_corpus([2018, 2018], root=root)) == len(
            load_corpus([2018], root=root)
        )

    def test_an_empty_root_raises_rather_than_returning_nothing(self, tmp_path):
        with pytest.raises(DatasetError, match="no season checkpoints"):
            load_corpus(root=tmp_path)

    def test_available_seasons_ignores_a_partial_write(self, root, tmp_path):
        assert available_seasons(root) == SEASONS
        (tmp_path / "season_2031.parquet.partial").write_bytes(b"")
        assert available_seasons(tmp_path) == ()

    def test_available_seasons_of_a_missing_directory_is_empty(self, tmp_path):
        assert available_seasons(tmp_path / "nowhere") == ()


class TestEnrich:
    def test_absent_enrichment_is_tolerated(self, root):
        loaded = load_corpus(root=root)
        out = enrich(loaded, root=root)
        assert list(out.columns) == list(loaded.columns)
        assert len(out) == len(loaded)

    def test_absent_enrichment_is_logged_rather_than_silent(self, root, caplog):
        with caplog.at_level(logging.INFO, logger="f1pred.dataset"):
            enrich(load_corpus(root=root), root=root)
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "weather archive" in messages
        assert "practice checkpoints" in messages

    def test_weather_is_attached_when_the_archive_is_on_disk(self, enriched_root):
        out = enrich(load_corpus(root=enriched_root), root=enriched_root)
        assert "weather_prior_wet_probability" in out.columns
        assert out["weather_prior_wet_probability"].notna().any()

    def test_practice_is_attached_when_checkpoints_are_on_disk(self, enriched_root):
        out = enrich(load_corpus(root=enriched_root), root=enriched_root)
        assert "long_run_pace_s" in out.columns
        assert out["long_run_pace_s"].notna().all()

    def test_enrichment_does_not_change_the_row_count(self, enriched_root):
        loaded = load_corpus(root=enriched_root)
        assert len(enrich(loaded, root=enriched_root)) == len(loaded)

    def test_enriching_twice_does_not_suffix_the_columns(self, enriched_root):
        once = enrich(load_corpus(root=enriched_root), root=enriched_root)
        twice = enrich(once, root=enriched_root)
        assert list(twice.columns) == list(once.columns)

    def test_practice_on_disk_without_a_join_key_is_a_corpus_defect(
        self, enriched_root
    ):
        loaded = load_corpus(root=enriched_root).drop(columns=["driver_code"])
        with pytest.raises(DatasetError, match="driver_code"):
            enrich(loaded, root=enriched_root)


class TestExperimentFrame:
    @pytest.fixture(scope="class")
    def bare(self, root) -> Experiment:
        return experiment(load_corpus(root=root))

    def test_the_frame_carries_the_keys(self, bare):
        assert set(KEY_COLUMNS) <= set(bare.frame.columns)

    def test_the_frame_carries_the_segmentation_column(self, bare):
        # FR-505 segments by circuit; the harness reads it off this frame.
        assert "circuit_id" in bare.frame.columns

    def test_the_frame_carries_both_baseline_columns(self, bare):
        assert set(BASELINE_COLUMNS) <= set(bare.frame.columns)
        assert bare.frame["championship_points_prior"].notna().all()
        assert (bare.frame["championship_points_prior"] >= 0).all()

    def test_the_frame_carries_every_built_feature(self, bare):
        assert set(bare.feature_names) <= set(bare.frame.columns)
        assert bare.feature_names == bare.matrix.feature_names
        assert len(bare.feature_names) > 0

    def test_the_frame_carries_the_target(self, bare):
        assert TARGET_COLUMN in bare.frame.columns
        assert bare.frame[TARGET_COLUMN].notna().all()

    def test_everything_sits_on_one_aligned_index(self, bare, root):
        loaded = load_corpus(root=root)
        assert len(bare.frame) == len(loaded)
        assert bare.frame.index.equals(bare.matrix.X.index)
        assert bare.frame.index.equals(bare.matrix.keys.index)
        assert len(bare) == len(bare.frame)

    def test_the_frame_is_in_canonical_order(self, bare, root):
        canonical = sort_canonically(load_corpus(root=root))
        assert bare.frame["driver_id"].tolist() == canonical["driver_id"].tolist()

    def test_the_baseline_columns_are_the_corpus_values(self, bare, root):
        canonical = sort_canonically(load_corpus(root=root))
        assert bare.frame["grid_position"].tolist() == (
            canonical["grid_position"].tolist()
        )

    def test_a_stale_points_column_is_recomputed(self, root):
        loaded = load_corpus(root=root)
        loaded["championship_points_prior"] = -999.0
        assert (experiment(loaded).frame["championship_points_prior"] >= 0).all()

    def test_both_baselines_run_on_the_assembled_frame(self, bare):
        one_race = bare.frame[
            (bare.frame["season"] == 2019) & (bare.frame["round"] == 1)
        ]
        for baseline in (GridOrderBaseline(), StandingsOrderBaseline()):
            predicted = baseline.predict(one_race)
            assert sorted(predicted) == list(range(1, len(one_race) + 1))

    def test_absent_enrichment_is_reported_through_absent_sources(self, bare):
        assert "weather_prior_wet_probability" in bare.absent_sources
        assert "long_run_pace_s" in bare.absent_sources
        assert bare.absent_sources == bare.matrix.absent_sources
        assert bare.frame["weather_prior_wet"].isna().all()

    def test_enrichment_removes_a_source_from_the_absent_list(self, enriched_root):
        built = experiment(
            enrich(load_corpus(root=enriched_root), root=enriched_root)
        )
        assert "weather_prior_wet_probability" not in built.absent_sources
        assert "long_run_pace_s" not in built.absent_sources
        assert built.frame["weather_prior_wet"].notna().any()

    def test_a_feature_named_after_a_baseline_column_raises(self, root):
        with pytest.raises(DatasetError, match="grid_position"):
            experiment(load_corpus(root=root), builder=colliding_builder)

    def test_a_feature_named_after_a_key_raises(self, root):
        loaded = load_corpus(root=root)
        primary = build_primary(loaded)
        renamed = FeatureMatrix(
            set_name=primary.set_name,
            X=primary.X.rename(columns={"field_size": "driver_id"}),
            keys=primary.keys,
            flags=primary.flags,
            target=primary.target,
            feature_names=tuple(
                "driver_id" if n == "field_size" else n
                for n in primary.feature_names
            ),
        )
        with pytest.raises(DatasetError, match="driver_id"):
            experiment(loaded, matrix=renamed)

    def test_a_matrix_built_from_another_corpus_is_refused(self, root):
        with pytest.raises(DatasetError, match="different corpora"):
            experiment(
                load_corpus(root=root),
                matrix=build_primary(load_corpus([2018], root=root)),
            )

    def test_a_supplied_matrix_is_used_rather_than_rebuilt(self, root):
        loaded = load_corpus(root=root)
        oracle = build_oracle(loaded)
        built = experiment(loaded, matrix=oracle)
        assert built.matrix is oracle
        assert built.feature_names == oracle.feature_names
        assert set(build_primary(loaded).feature_names) < set(built.feature_names)

    def test_a_frame_with_no_target_and_no_grid_is_refused(self, root):
        loaded = load_corpus(root=root).drop(columns=["grid_position"])
        with pytest.raises(DatasetError, match="grid_position"):
            experiment(loaded, matrix=build_primary(load_corpus(root=root)))


class TestPrepare:
    def test_prepare_chains_load_enrich_and_assemble(self, enriched_root, corpus):
        built = prepare(root=enriched_root)
        assert isinstance(built, Experiment)
        assert len(built.frame) == len(corpus)
        assert "weather_prior_wet_probability" not in built.absent_sources
        assert set(BASELINE_COLUMNS) <= set(built.frame.columns)
        assert TARGET_COLUMN in built.frame.columns

    def test_prepare_honours_a_season_subset(self, enriched_root):
        built = prepare([2018, 2019], root=enriched_root)
        assert sorted(built.frame["season"].unique()) == [2018, 2019]

    def test_prepare_refuses_a_season_that_was_never_collected(self, enriched_root):
        with pytest.raises(DatasetError, match="2024"):
            prepare([2018, 2024], root=enriched_root)

    def test_prepare_is_deterministic(self, root):
        first, second = prepare(root=root), prepare(root=root)
        pd.testing.assert_frame_equal(first.frame, second.frame)


class TestNonStarters:
    """The rows FR-002 drops, which is where the index alignment is decided.

    `features.build` drops non-starters and re-indexes; a frame sorted outside
    it keeps every row and its own index. Both are `RangeIndex`, so pandas
    aligns them positionally and without complaint — the features of row 6,868
    would be handed the grid slot of a different driver, and no exception would
    be raised anywhere. The real corpus has 43 such rows against 6,911, so this
    is not a hypothetical: every one of these tests failed before the drop was
    moved ahead of the build.

    `synthetic_corpus` has no withdrawals, which is exactly why the defect
    survived a full test suite. These fixtures add them.
    """

    @staticmethod
    def with_withdrawals(corpus: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        from f1pred.data.schema import ClassificationOutcome

        out = corpus.copy().reset_index(drop=True)
        first = out[(out["season"] == 2018) & (out["round"] == 1)].index[:n]
        out.loc[first, "classification_outcome"] = (
            ClassificationOutcome.WITHDRAWN.value
        )
        return out

    @pytest.fixture
    def withdrawn_root(self, tmp_path, corpus) -> Path:
        directory = tmp_path / "raw_withdrawn"
        directory.mkdir()
        frame = self.with_withdrawals(corpus)
        for season, rows in frame.groupby("season", sort=True):
            write_checkpoint(rows.reset_index(drop=True), directory, int(season))
        return directory

    def test_a_corpus_with_withdrawals_still_assembles(self, withdrawn_root, corpus):
        built = prepare(root=withdrawn_root)
        assert len(built.frame) == len(corpus) - 3

    def test_the_dropped_rows_are_the_withdrawals(self, withdrawn_root, corpus):
        built = prepare(root=withdrawn_root)
        started = build_target(self.with_withdrawals(corpus))
        assert len(built.frame) == len(started)

    def test_grid_slots_follow_their_own_driver(self, withdrawn_root, corpus):
        """The assertion the silent positional alignment would have failed.

        Every assembled row is matched back to its source row by driver and
        race, and its grid slot must be that driver's. Off-by-one alignment
        after a drop puts a neighbour's grid slot on the row and nothing else
        in the pipeline notices.
        """
        built = prepare(root=withdrawn_root)
        source = build_target(self.with_withdrawals(corpus))
        merged = built.frame.merge(
            source[["season", "round", "driver_id", "grid_position", TARGET_COLUMN]],
            on=["season", "round", "driver_id"],
            suffixes=("", "_source"),
        )
        assert len(merged) == len(built.frame)
        assert (merged["grid_position"] == merged["grid_position_source"]).all()
        assert (merged[TARGET_COLUMN] == merged[f"{TARGET_COLUMN}_source"]).all()

    def test_a_baseline_scores_the_assembled_frame(self, withdrawn_root):
        """A misaligned grid column produces a valid-looking, wrong ordering,
        so the baseline is run rather than merely constructed."""
        built = prepare(root=withdrawn_root)
        race = built.frame[
            (built.frame["season"] == 2018) & (built.frame["round"] == 1)
        ]
        predicted = GridOrderBaseline().predict(race)
        assert sorted(predicted.tolist()) == list(range(1, len(race) + 1))
