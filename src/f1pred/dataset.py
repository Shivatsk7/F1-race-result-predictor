"""From the checkpoints on disk to the one frame an evaluation scores (FR-304, FR-506).

Every layer below this one already exists and is tested: collection writes a
per-season checkpoint (FR-102), the weather archive and the practice sessions
sit beside it, and `features.build` turns a race frame into a matrix. What did
not exist was a committed path *between* them. Each results file in `results/`
was produced by assembling that path by hand, which means no two of them can be
proved to have been assembled the same way, and FR-506's machine-readable
record describes a run nobody can reproduce.

Three decisions here are about that, and each of them is a refusal:

* An explicitly requested season with no checkpoint raises. Returning the
  seasons that happened to be present would let a results file claim a
  16-season corpus while resting on nine, and nothing downstream would notice —
  the frame would simply be shorter.
* The assembled frame carries the baselines' raw inputs and the model's
  features side by side, on one index. FR-304 requires the baselines to be
  scored on the identical holdout by the identical code, and the cheapest way
  to break that is to hand them a frame assembled separately.
* A feature name that collides with a key or a baseline column raises. A silent
  overwrite there means `grid_position` holds one thing for `GridOrderBaseline`
  and another for the model, and the comparison FR-301 exists to make is no
  longer between two predictors on one dataset.

Missing *enrichment* is the deliberate exception. Weather (FR-206) and practice
telemetry (FR-213, FR-214) are optional attachments; the builder already nulls
their columns and records `absent_sources`, so a corpus collected without them
is a thinner experiment rather than a failed one.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from f1pred.config import RAW_DIR
from f1pred.data import telemetry, weather
from f1pred.data.collect import checkpoint_path, read_checkpoint
from f1pred.data.target import build_target
from f1pred.features.build import (
    KEY_COLUMNS,
    TARGET_COLUMN,
    FeatureMatrix,
    build_primary,
)
from f1pred.features.priors import sort_canonically
from f1pred.models.baselines import championship_points_prior

logger = logging.getLogger(__name__)

#: The name `collect.checkpoint_path` writes. Matched rather than globbed so a
#: stray `season_2019.parquet.partial` from an interrupted write is not mistaken
#: for a completed season.
_CHECKPOINT_NAME: Final[re.Pattern[str]] = re.compile(r"^season_(\d{4})\.parquet$")

#: Columns the baselines read straight off the frame rather than from the
#: matrix. They are not features and must never become features: FR-302's
#: standings baseline stops being an independent comparison the moment it reads
#: something the feature pipeline produced.
BASELINE_COLUMNS: Final[tuple[str, ...]] = (
    "grid_position",
    "championship_points_prior",
)

#: What `championship_points_prior` needs, checked here so the failure names the
#: assembly step rather than surfacing from inside a baseline.
_BASELINE_INPUTS: Final[tuple[str, ...]] = ("grid_position", "points")


class DatasetError(ValueError):
    """Raised when an experiment frame cannot be assembled as asked for."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def available_seasons(root: Path = RAW_DIR) -> tuple[int, ...]:
    """Seasons with a completed checkpoint under `root`, ascending."""
    directory = Path(root)
    if not directory.is_dir():
        return ()
    found = (
        _CHECKPOINT_NAME.match(p.name) for p in directory.iterdir() if p.is_file()
    )
    return tuple(sorted(int(m.group(1)) for m in found if m is not None))


def load_corpus(
    seasons: Iterable[int] | None = None, *, root: Path = RAW_DIR
) -> pd.DataFrame:
    """Read the season checkpoints and return one canonically ordered corpus.

    `seasons=None` means every checkpoint on disk. A named season with no
    checkpoint is an error, not a season to skip — silently returning a short
    corpus is the failure mode that makes a results file lie about what it was
    trained on, and it leaves no trace anywhere in the output.

    The requested seasons are de-duplicated and sorted before reading, so the
    same request written two ways cannot produce two different corpora
    (FR-208); asking for a season twice would otherwise double its rows.
    """
    directory = Path(root)
    if seasons is None:
        wanted = available_seasons(directory)
        if not wanted:
            raise DatasetError(
                f"no season checkpoints under {directory}. Collect the corpus "
                f"first: an empty corpus is not a small experiment, it is no "
                f"experiment."
            )
    else:
        wanted = tuple(sorted({int(s) for s in seasons}))
        if not wanted:
            raise DatasetError("no seasons requested")

    frames: list[pd.DataFrame] = []
    for season in wanted:
        frame = read_checkpoint(directory, season)
        if frame is None:
            raise DatasetError(
                f"season {season} was requested but has no checkpoint at "
                f"{checkpoint_path(directory, season)}. Refusing to return the "
                f"other {len(wanted) - 1} season(s) under the name of a corpus "
                f"that includes it."
            )
        frames.append(frame)

    logger.info(
        "loaded %d season(s) from %s: %s",
        len(frames),
        directory,
        ", ".join(str(s) for s in wanted),
    )
    return sort_canonically(pd.concat(frames, ignore_index=True))


# --------------------------------------------------------------------------
# Enrichment (FR-206, FR-213, FR-214)
# --------------------------------------------------------------------------


def _attach_weather(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    archive = weather.read_archive(root)
    if archive is None or archive.empty:
        logger.info("no weather archive under %s; weather features stay null", root)
        return df
    if any(column in df.columns for column in weather.PRIOR_COLUMNS):
        # A second merge would suffix the existing columns `_x`/`_y`, leaving
        # neither under the name the registry reads — an enriched frame that
        # builds as though it were bare.
        logger.info("weather already attached; skipping")
        return df
    return weather.attach(df, weather.race_weather_table(archive, df))


def _attach_practice(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    seasons = sorted({int(s) for s in df["season"].unique()})
    frames = [
        frame
        for frame in (telemetry.read_practice_checkpoint(root, s) for s in seasons)
        if frame is not None and not frame.empty
    ]
    if not frames:
        logger.info(
            "no practice checkpoints under %s for seasons %s; telemetry features "
            "stay null, which FR-214 makes a fact rather than a gap",
            root,
            seasons,
        )
        return df
    if "long_run_pace_s" in df.columns:
        logger.info("practice data already attached; skipping")
        return df
    if "driver_code" not in df.columns:
        raise DatasetError(
            "practice data is on disk but the corpus carries no 'driver_code' "
            "to join it on. That is a malformed corpus, not an absent source."
        )
    return telemetry.attach(df, pd.concat(frames, ignore_index=True))


def enrich(df: pd.DataFrame, *, root: Path = RAW_DIR) -> pd.DataFrame:
    """Attach whatever enrichment `root` holds, and say what it did not hold.

    Absence is logged rather than raised. The two sources are optional by
    construction — the weather archive is one file for the whole corpus, and
    FR-214 makes practice telemetry null for half of it regardless — so a
    caller who has neither still gets a buildable frame, with the builder
    recording the shortfall in `absent_sources` rather than this function
    guessing at values.
    """
    directory = Path(root)
    return _attach_practice(_attach_weather(df, directory), directory)


# --------------------------------------------------------------------------
# The experiment frame
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Experiment:
    """One corpus, assembled once, for every predictor that will be scored on it.

    `frame` and `matrix` are two views of the same rows in the same order, not
    two datasets: the frame is what the harness hands a predictor race by race,
    the matrix is what a model trains on. Keeping them in one object is what
    stops a caller scoring a model on features built from a different frame
    than the one the baselines saw (FR-304).
    """

    frame: pd.DataFrame
    matrix: FeatureMatrix
    feature_names: tuple[str, ...]
    #: Carried up from the matrix so a results file can state which enrichment
    #: the run actually had. A run with no weather is a different experiment
    #: from one with it, and only this field distinguishes them afterwards.
    absent_sources: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.frame)


def experiment(
    df: pd.DataFrame,
    *,
    matrix: FeatureMatrix | None = None,
    builder: Callable[[pd.DataFrame], FeatureMatrix] = build_primary,
) -> Experiment:
    """Assemble keys, baseline inputs, features and target onto one index.

    `matrix` reuses an already-built matrix — the oracle and telemetry arms
    build their own, and rebuilding here would both cost a second pass and
    quietly discard the arm's feature set. It is checked against the frame
    rather than trusted: a matrix built from a different corpus would otherwise
    align by position and produce a frame whose features belong to other races.

    `championship_points_prior` is recomputed here even when the input frame
    already carries one, so the column FR-302 reads is always the one
    `models.baselines` defines rather than whatever a caller left behind.

    The frame is put through FR-002's non-starter drop and the canonical sort
    *before* the matrix is built, and that ordering is the whole correctness
    argument for this function. `features.build` does both internally and
    `sort_canonically` resets the index, so a frame sorted out here and a matrix
    built in there end up carrying two unrelated `RangeIndex`es of different
    lengths — 6,911 rows against 6,868 on the real corpus, because a withdrawal
    is not a race. Two `RangeIndex`es align silently and positionally, which
    would staple one driver's grid slot to another driver's features and raise
    nothing at all. Doing the drop and the sort first makes the two indexes the
    same object's, so the alignment below is real rather than coincidental.
    """
    missing = [c for c in _BASELINE_INPUTS if c not in df.columns]
    if missing:
        raise DatasetError(
            f"cannot assemble the baseline columns: the corpus is missing "
            f"{missing}. FR-301 and FR-302 read these off the same frame the "
            f"model does, so an absent one is a corpus defect, not a reason to "
            f"score the model unopposed."
        )

    work = sort_canonically(df if TARGET_COLUMN in df.columns else build_target(df))
    built = builder(work) if matrix is None else matrix

    if len(built.keys) != len(work) or not built.keys.index.equals(work.index):
        raise DatasetError(
            f"the supplied matrix has {len(built.keys)} row(s) and the frame "
            f"{len(work)}; they were built from different corpora. Aligning "
            f"them would attach one race's features to another race's result."
        )

    reserved = set(KEY_COLUMNS) | set(BASELINE_COLUMNS) | {TARGET_COLUMN}
    collisions = sorted(set(built.feature_names) & reserved)
    if collisions:
        raise DatasetError(
            f"feature name(s) {collisions} collide with a key, a baseline "
            f"column, or the target. One name would then mean two things: the "
            f"model would read the feature and the baseline the raw column, "
            f"and the comparison between them would be meaningless."
        )

    if built.target is not None:
        target = built.target
    elif TARGET_COLUMN in work.columns:
        target = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")
    else:
        raise DatasetError(
            f"neither the matrix nor the corpus carries {TARGET_COLUMN!r}; "
            f"there is nothing to score against"
        )

    # `keys` already carries `circuit_id`, which FR-505's segmentation reads.
    frame = built.keys.copy()
    frame["grid_position"] = work["grid_position"]
    frame["championship_points_prior"] = championship_points_prior(work)
    for name in built.feature_names:
        frame[name] = built.X[name]
    frame[TARGET_COLUMN] = target

    return Experiment(
        frame=frame,
        matrix=built,
        feature_names=built.feature_names,
        absent_sources=built.absent_sources,
    )


def prepare(
    seasons: Iterable[int] | None = None,
    *,
    root: Path = RAW_DIR,
    builder: Callable[[pd.DataFrame], FeatureMatrix] = build_primary,
) -> Experiment:
    """Disk to scorable experiment in one call — the reproducible path.

    Exists so that the answer to "how was this results file produced?" is a
    function name rather than a reconstruction of somebody's session.
    """
    return experiment(
        enrich(load_corpus(seasons, root=root), root=root), builder=builder
    )


__all__ = [
    "BASELINE_COLUMNS",
    "DatasetError",
    "Experiment",
    "available_seasons",
    "enrich",
    "experiment",
    "load_corpus",
    "prepare",
]
