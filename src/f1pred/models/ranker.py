"""The learned ranker (FR-401..FR-406). Ported from v0's `RacePredictor`.

Three things changed in the port, and each of them is the point of the port.

**The split is not the model's business.** v0's trainer took the whole frame
and cut its own test set with `TimeSeriesSplit`, so the boundary the model
trained against and the boundary it was scored against were set in different
files by different code. Here training takes a split (`TrainingBoundary`: a
`ChronologicalSplit` or a one-season `SeasonFold`) and calls
`assert_no_leakage` on what it is about to fit (FR-403) — the model never
chooses its own holdout, and cannot be handed one that overlaps.

**Intervals are calibrated on data the model did not train on.** v0 fitted its
residual bands on the test set, which is the same mistake as tuning on the
holdout: the intervals looked well-sized because they had seen the answers. The
bands here come from a chronological slice held back from *inside* training
(FR-402), so the holdout stays untouched. They are deliberately conservative —
the shadow fit that produces them sees fewer seasons than the shipped model, so
its residuals are, if anything, wider than the shipped model's will be. Erring
wide is the honest direction for an interval; erring narrow claims a precision
that was never measured. Nothing here asserts the bands are *correct*: that is
P5's job, and until P5 measures coverage every interval this model emits is
UNCALIBRATED.

**Recency weighting is a parameter, not a rewrite** (FR-406). Zero decay is
uniform weighting exactly, so the weighting's effect can be measured against
its own absence rather than argued about.

The model implements `Predictor` (FR-304), so the evaluation harness cannot
tell it apart from the three baselines and cannot score it on different terms.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import xgboost as xgb

from f1pred.config import SEED
from f1pred.evaluate.split import TrainingBoundary
from f1pred.features.build import TARGET_COLUMN
from f1pred.features.registry import REGISTRY, Registry
from f1pred.models.base import assert_carries_uncertainty, rank_to_positions

logger = logging.getLogger(__name__)

#: Race identity, and the order the ranker's groups must arrive in.
RACE_KEYS: Final[tuple[str, str]] = ("season", "round")

#: How many of the latest training seasons are held back to size the intervals
#: of FR-402. Three, matching the holdout floor: a slice thin enough to be
#: dominated by one freak season would give bands that say more about that
#: season than about the model.
DEFAULT_CALIBRATION_SEASONS: Final[int] = 3

#: FR-402's interval, as a central coverage level. Stated here rather than at
#: the call site so that two results files cannot quote intervals of different
#: widths under the same name.
DEFAULT_INTERVAL_COVERAGE: Final[float] = 0.80


class ModelError(ValueError):
    """Raised when a model cannot be trained or asked to predict as specified."""


# --------------------------------------------------------------------------
# Hyperparameters (FR-404 field 4, FR-406)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hyperparameters:
    """Everything that changes what the fit produces, in one recorded object.

    A frozen dataclass rather than a dict because FR-404 requires the
    hyperparameters to be persisted with the model: a dict invites a call site
    to pass one extra key that never reaches the record, and then the record
    describes a model that was not the one trained.

    The tree parameters are v0's, carried over unchanged and deliberately not
    retuned. Tuning them against the holdout would spend the holdout, and the
    project's question is whether a model of this kind beats the grid, not
    whether a tuned one does.
    """

    objective: str = "rank:pairwise"
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    seed: int = SEED

    #: FR-406. Weight of a training row decays by `exp(-decay * seasons_old)`,
    #: counting the most recent training season as zero seasons old. Zero is
    #: uniform weighting exactly — not approximately — so the benefit of
    #: recency can be measured against its own absence (T-408).
    recency_decay: float = 0.0

    #: FR-402. Central coverage of the emitted interval.
    interval_coverage: float = DEFAULT_INTERVAL_COVERAGE

    #: Seasons held back from training to size the intervals.
    calibration_seasons: int = DEFAULT_CALIBRATION_SEASONS

    #: Stop adding trees once the held-back seasons have not improved for this
    #: many rounds, and fit the shipped model to the count that survived.
    #: Without it the first honest run of this model scored 0.9625 in-sample
    #: against 0.6302 on the holdout: 400 trees at depth 6 memorise 274 races
    #: comfortably. `None` disables it and keeps every tree, which is what the
    #: tests of the mechanism itself use.
    #:
    #: This is model selection, and it is deliberately confined to seasons
    #: *inside* training. Choosing the tree count against the holdout would
    #: spend the holdout — the number it then produced would be the best of
    #: several looks at the answer, reported as though it were the first.
    early_stopping_rounds: int | None = 40

    #: Single-threaded on purpose. XGBoost's histogram builder sums gradients
    #: in thread-completion order, so a multi-threaded fit is reproducible only
    #: for a fixed thread count — and FR-405 promises reproducibility from the
    #: seed and the data, not from the machine. The corpus is ~7,000 rows; the
    #: fit costs seconds either way.
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.recency_decay < 0.0:
            raise ModelError(
                f"recency_decay must be non-negative, got {self.recency_decay}; "
                f"a negative decay would weight the distant past above last season"
            )
        if not 0.0 < self.interval_coverage < 1.0:
            raise ModelError(
                f"interval_coverage must lie strictly between 0 and 1, got "
                f"{self.interval_coverage}"
            )
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ModelError(
                f"early_stopping_rounds must be positive or None, got "
                f"{self.early_stopping_rounds}"
            )
        if self.calibration_seasons < 1:
            raise ModelError(
                "at least one season must be held back to size the intervals; "
                "calibrating them on the rows the model trained on would "
                "reproduce v0's mistake in a different place"
            )

    def as_dict(self) -> dict[str, Any]:
        """FR-404 field 4. Every field, including the ones left at default."""
        return asdict(self)

    def booster_params(self) -> dict[str, Any]:
        """The subset xgboost itself accepts."""
        return {
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
        }


# --------------------------------------------------------------------------
# Training inputs
# --------------------------------------------------------------------------


def recency_weights(
    seasons: pd.Series, *, decay: float, most_recent: int | None = None
) -> np.ndarray:
    """FR-406: weight a race by how recent its season is.

    One weight per *race*, not per driver-row. That is what a ranking objective
    accepts — it weights query groups, since a pairwise loss is defined between
    two cars in the same race and a per-row weight would have no consistent
    meaning for the pair. It is also what recency means here: a season is
    stale or current as a whole, never one car at a time.

    Age is counted from the most recent *training* season rather than from
    today, so the weights of a given training set do not shift when the corpus
    window advances. Zero decay returns exact ones — `exp(-0 * age)` is 1.0 in
    IEEE arithmetic for every finite age, so uniform weighting is reproduced
    bit-for-bit rather than merely closely (T-408).
    """
    values = pd.to_numeric(seasons, errors="coerce").to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise ModelError("training races carry a non-numeric season")
    latest = float(most_recent if most_recent is not None else values.max())
    return np.exp(-decay * (latest - values))


def race_seasons(frame: pd.DataFrame) -> pd.Series:
    """The season of each race, in the order the races appear in `frame`.

    `sort=False` so the result lines up with the query-group ids, which are
    numbered by order of appearance. Sorting here would pair a race's weight
    with a different race's rows without any error being raised.
    """
    return frame.groupby(list(RACE_KEYS), sort=False)["season"].first()


def relevance(positions: pd.Series, race_ids: pd.Series) -> np.ndarray:
    """Turn finishing positions into the gains a ranking objective wants.

    XGBoost ranks by *descending* label, so the winner must carry the largest
    number. The gain is the count of cars this one finished ahead of, computed
    per race: a P5 in a 20-car field beat more cars than a P5 in a field of
    twelve, and a label that ignored field size would call those two results
    identical.
    """
    pos = pd.to_numeric(positions, errors="coerce")
    if pos.isna().any():
        raise ModelError(
            f"{int(pos.isna().sum())} training row(s) have no {TARGET_COLUMN}; "
            f"the FR-001 target must be built before training"
        )
    field_size = pos.groupby(race_ids).transform("size")
    return (field_size - pos).to_numpy(dtype="float64")


def _race_ids(keys: pd.DataFrame) -> pd.Series:
    missing = [k for k in RACE_KEYS if k not in keys.columns]
    if missing:
        raise ModelError(f"frame is missing race key column(s) {missing}")
    return (
        keys[list(RACE_KEYS)].astype("int64").astype(str).agg("_".join, axis=1)
    )


def _query_ids(race_ids: pd.Series) -> np.ndarray:
    """Contiguous integer group ids, as xgboost's `qid` requires.

    The rows must already be grouped by race; `factorize` numbers them in order
    of appearance, so a frame that is *not* grouped produces a non-monotonic
    qid and xgboost raises rather than silently training on scrambled groups.
    """
    codes, _ = pd.factorize(race_ids, sort=False)
    return codes.astype("int64")


# --------------------------------------------------------------------------
# The trained model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """FR-402's band, as an offset either side of the point prediction.

    Offsets rather than absolute positions because the band is fitted once and
    applied to every driver: what is calibrated is how far the truth tends to
    sit from where the model put a car, not where the truth tends to be.
    """

    lower_offset: float
    upper_offset: float
    coverage: float
    calibration_rows: int
    calibration_seasons: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "lower_offset": self.lower_offset,
            "upper_offset": self.upper_offset,
            "calibration_rows": self.calibration_rows,
            "calibration_seasons": list(self.calibration_seasons),
            # T-509 will make this label unskippable at the render layer. Until
            # P5 measures coverage, the number above is a fitted width and not
            # a verified one, and the results files must say so.
            "status": "UNCALIBRATED",
        }


@dataclass(frozen=True, slots=True)
class TrainedRanker:
    """A fitted ranker that satisfies `Predictor` (FR-304, FR-401).

    Frozen: a trained model that can be re-fitted in place is a model whose
    recorded metadata can stop describing it, which is precisely what FR-404
    exists to prevent. Re-training returns a new object.
    """

    booster: xgb.XGBRanker
    feature_names: tuple[str, ...]
    params: Hyperparameters
    interval: Interval
    training_seasons: tuple[int, ...]
    training_rows: int
    training_date_range: tuple[str, str]
    code_commit: str
    #: Trees actually kept. Differs from `params.n_estimators` whenever early
    #: stopping fired, and it is the count that describes the shipped model.
    trees: int = 0
    name: str = "xgboost_ranker"
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)

    # -- prediction ---------------------------------------------------------

    def scores(self, race: pd.DataFrame) -> pd.Series:
        """Raw ranking scores, higher meaning further up the order."""
        missing = [f for f in self.feature_names if f not in race.columns]
        if missing:
            raise ModelError(
                f"{self.name} was trained on {len(self.feature_names)} features "
                f"and the frame is missing {len(missing)} of them: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}. Predicting on a subset "
                f"would silently score a different model than the one recorded."
            )
        raw = self.booster.predict(race[list(self.feature_names)])
        return pd.Series(raw, index=race.index, name="rank_score")

    def predict(self, race: pd.DataFrame) -> pd.Series:
        """One race in, positions 1..n out. The `Predictor` contract."""
        return rank_to_positions(self.scores(race), ascending=False)

    def predict_with_interval(self, race: pd.DataFrame) -> pd.DataFrame:
        """FR-402: a point prediction and an interval, per driver.

        The interval is centred on the point prediction and clipped to the
        field, so it can never propose a position nobody can finish in. Clipping
        makes the band narrower for the cars at either end of the grid, which is
        a real statement — P1 cannot finish better than first — and not a
        precision claim.
        """
        positions = self.predict(race)
        field_size = float(len(positions))
        lower = (positions + self.interval.lower_offset).clip(1.0, field_size)
        upper = (positions + self.interval.upper_offset).clip(1.0, field_size)
        result = pd.DataFrame(
            {
                "predicted_position": positions,
                "predicted_position_lower": lower,
                "predicted_position_upper": upper,
                "interval_coverage": self.interval.coverage,
            },
            index=race.index,
        )
        # FR-752 (T-514): checked at the one place this project turns a point
        # prediction into something published, not trusted from how the frame
        # above was built.
        assert_carries_uncertainty(result)
        return result

    # -- the record (FR-404) ------------------------------------------------

    def with_evaluation_metrics(self, metrics: dict[str, Any]) -> TrainedRanker:
        """Attach FR-404's sixth field once the harness has produced it."""
        return replace(self, evaluation_metrics=dict(metrics))

    def metadata(self, *, registry: Registry = REGISTRY) -> dict[str, Any]:
        """FR-404's six fields, and exactly those six.

        Enumerated as a tuple of keys and asserted against, because the
        requirement says *exactly* these six: a seventh field added in passing
        would go unnoticed, and a missing one is the kind of gap that only
        surfaces when somebody tries to reproduce a number a year later.
        """
        declared = {f.name: f.availability.value for f in registry}
        record = {
            "training_date_range": list(self.training_date_range),
            "training_rows": self.training_rows,
            "features": [
                {"name": n, "availability": declared.get(n, "unregistered")}
                for n in self.feature_names
            ],
            # `trees_kept` sits with the hyperparameters because that is what
            # it is: early stopping makes `n_estimators` a ceiling rather than
            # a setting, and the count that survived is the one that describes
            # this model.
            "hyperparameters": {**self.params.as_dict(), "trees_kept": self.trees},
            "code_commit": self.code_commit,
            "evaluation_metrics": self.evaluation_metrics,
        }
        if tuple(record) != FR404_FIELDS:
            raise ModelError(
                f"model record carries {tuple(record)}; FR-404 enumerates "
                f"exactly {FR404_FIELDS}"
            )
        return record

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Write the booster beside its FR-404 record, under one stem.

        Two files rather than a pickle: a pickled estimator is readable only by
        the library version that wrote it, and a model whose metadata cannot be
        read without xgboost installed is a model whose provenance is hostage
        to a dependency.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_path = path.with_suffix(".ubj")
        self.booster.save_model(model_path)
        record = self.metadata()
        record["interval"] = self.interval.as_dict()
        record["name"] = self.name
        record["training_seasons"] = list(self.training_seasons)
        record["model_file"] = model_path.name
        path.with_suffix(".json").write_text(
            json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        return model_path

    @classmethod
    def load(cls, path: Path) -> TrainedRanker:
        """Read back what `save` wrote, under the same stem.

        The feature list comes from the record rather than from the booster,
        because it is the record that FR-404 makes authoritative and the record
        that a reader can check. A booster whose feature count disagrees with
        the record is a corrupted pair, and it raises here rather than
        predicting from whichever of the two the library happened to prefer.
        """
        path = Path(path)
        record_path, model_path = path.with_suffix(".json"), path.with_suffix(".ubj")
        if not record_path.exists() or not model_path.exists():
            raise ModelError(
                f"a saved model is two files under one stem, {path}.ubj and "
                f"{path}.json; found "
                f"{[p.name for p in (model_path, record_path) if p.exists()] or 'neither'}"
            )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        booster = xgb.XGBRanker()
        booster.load_model(model_path)

        # FR-404 records each feature as a name and its declared availability;
        # the model itself needs the names, in the recorded order.
        features = tuple(f["name"] for f in record["features"])
        loaded_count = getattr(booster, "n_features_in_", len(features))
        if loaded_count != len(features):
            raise ModelError(
                f"{model_path.name} takes {loaded_count} features but "
                f"{record_path.name} records {len(features)}; the pair does not "
                f"describe one model"
            )

        interval_record = record["interval"]
        # `trees_kept` rides with the hyperparameters in the record but is not
        # one of them — it is what early stopping left, and it is restored onto
        # the model rather than passed back into the parameter object.
        hyperparameters = dict(record["hyperparameters"])
        trees = int(hyperparameters.pop("trees_kept", 0))
        params = Hyperparameters(**hyperparameters)
        return cls(
            booster=booster,
            feature_names=features,
            params=params,
            interval=Interval(
                lower_offset=float(interval_record["lower_offset"]),
                upper_offset=float(interval_record["upper_offset"]),
                coverage=float(interval_record["coverage"]),
                calibration_rows=int(interval_record["calibration_rows"]),
                calibration_seasons=tuple(interval_record["calibration_seasons"]),
            ),
            training_seasons=tuple(record["training_seasons"]),
            training_rows=int(record["training_rows"]),
            training_date_range=tuple(record["training_date_range"]),
            code_commit=record["code_commit"],
            trees=trees,
            name=record.get("name", "xgboost_ranker"),
            evaluation_metrics=dict(record.get("evaluation_metrics") or {}),
        )


#: FR-404 enumerates six fields. The tuple is the enumeration, and
#: `metadata()` fails against it rather than trusting the writer to count.
FR404_FIELDS: Final[tuple[str, ...]] = (
    "training_date_range",
    "training_rows",
    "features",
    "hyperparameters",
    "code_commit",
    "evaluation_metrics",
)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def code_commit(repo: Path | None = None) -> str:
    """FR-404 field 5. The commit the training code was at, dirt included.

    A dirty tree is recorded as such rather than rejected: refusing to train
    would push people to commit noise to get a number, and a hash that silently
    describes different code than was run is worse than an honest `-dirty`.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def _fit(
    X: pd.DataFrame,
    y: np.ndarray,
    qid: np.ndarray,
    weights: np.ndarray,
    params: Hyperparameters,
    *,
    n_estimators: int | None = None,
    watch: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
) -> xgb.XGBRanker:
    """One fit. `watch` turns on early stopping against a held-back slice."""
    settings = params.booster_params()
    if n_estimators is not None:
        settings["n_estimators"] = n_estimators
    if watch is not None and params.early_stopping_rounds is not None:
        settings["early_stopping_rounds"] = params.early_stopping_rounds
        settings["eval_metric"] = "ndcg"
    booster = xgb.XGBRanker(**settings)
    kwargs: dict[str, Any] = {}
    if watch is not None and params.early_stopping_rounds is not None:
        watch_X, watch_y, watch_qid = watch
        kwargs = {"eval_set": [(watch_X, watch_y)], "eval_qid": [watch_qid]}
    booster.fit(X, y, qid=qid, sample_weight=weights, verbose=False, **kwargs)
    return booster


def _calibrate(
    frame: pd.DataFrame,
    features: Sequence[str],
    params: Hyperparameters,
    calibration_seasons: tuple[int, ...],
) -> tuple[Interval, int]:
    """Size FR-402's band, and pick the tree count, from one shadow fit.

    Both jobs need the same thing — a model scored on seasons it did not train
    on — so they share one fit rather than paying for two. The returned count
    is the number of trees that were still helping when the held-back seasons
    stopped improving; the shipped model is then fitted to exactly that many.

    The shadow model is trained on the training seasons *before* the
    calibration slice and scored on the slice, so the residuals measure a model
    predicting seasons it has not seen — which is the situation the shipped
    model will be in. The shipped model is then fitted on all training seasons,
    including the slice, because throwing away three seasons of signal to keep
    the two fits identical would cost more accuracy than the slight
    conservatism it removes.
    """
    proper = frame[~frame["season"].isin(calibration_seasons)]
    holdback = frame[frame["season"].isin(calibration_seasons)]
    if proper.empty or holdback.empty:
        raise ModelError(
            f"calibration slice {list(calibration_seasons)} leaves "
            f"{len(proper)} training and {len(holdback)} calibration rows; "
            f"FR-402's interval cannot be sized from that"
        )

    shadow = _fit(
        proper[list(features)],
        relevance(proper[TARGET_COLUMN], _race_ids(proper)),
        _query_ids(_race_ids(proper)),
        recency_weights(race_seasons(proper), decay=params.recency_decay),
        params,
        watch=(
            holdback[list(features)],
            relevance(holdback[TARGET_COLUMN], _race_ids(holdback)),
            _query_ids(_race_ids(holdback)),
        ),
    )
    # `best_iteration` is an index; the count of trees kept is one more. Absent
    # when early stopping is off, in which case every tree was kept.
    kept = (
        int(shadow.best_iteration) + 1
        if params.early_stopping_rounds is not None
        and getattr(shadow, "best_iteration", None) is not None
        else params.n_estimators
    )

    residuals: list[np.ndarray] = []
    for _, race in holdback.groupby(list(RACE_KEYS), sort=True):
        scores = pd.Series(
            shadow.predict(race[list(features)]), index=race.index
        )
        predicted = rank_to_positions(scores, ascending=False)
        actual = pd.to_numeric(race[TARGET_COLUMN], errors="coerce")
        residuals.append((actual - predicted).to_numpy(dtype="float64"))

    pooled = np.concatenate(residuals)
    pooled = pooled[~np.isnan(pooled)]
    if pooled.size == 0:
        raise ModelError("calibration slice produced no usable residuals")

    tail = (1.0 - params.interval_coverage) / 2.0
    return (
        Interval(
            lower_offset=float(np.quantile(pooled, tail)),
            upper_offset=float(np.quantile(pooled, 1.0 - tail)),
            coverage=params.interval_coverage,
            calibration_rows=int(pooled.size),
            calibration_seasons=calibration_seasons,
        ),
        kept,
    )


def train(
    frame: pd.DataFrame,
    *,
    split: TrainingBoundary,
    features: Sequence[str],
    params: Hyperparameters | None = None,
    name: str = "xgboost_ranker",
    repo: Path | None = None,
) -> TrainedRanker:
    """Fit the ranker on the training side of `split`, and only that side.

    `frame` is the whole corpus. The split is applied here rather than by the
    caller for the same reason the harness takes the whole frame: a trainer that
    accepts a pre-cut training set has no way to check that the cut respected
    the boundary, and FR-403 is not a property one can check after the fact.
    """
    params = params or Hyperparameters()
    features = tuple(features)
    if not features:
        raise ModelError("a model with no features is not a model")

    missing = [c for c in (*RACE_KEYS, TARGET_COLUMN, "race_date", *features)
               if c not in frame.columns]
    if missing:
        raise ModelError(f"frame is missing required column(s) {missing}")

    training, _ = split.split(frame)
    if training.empty:
        raise ModelError(
            f"training seasons {list(split.training_seasons)} select no rows "
            f"from this frame"
        )
    # FR-403. Checked, not assumed — the split checks its own boundary...
    split.assert_no_leakage(training)
    # ...and the trainer checks the split's cut against what it declared, so a
    # split whose `split()` and `training_seasons` disagree cannot widen the fit.
    declared = {int(s) for s in split.training_seasons}
    undeclared = sorted(
        {int(s) for s in pd.to_numeric(training["season"]).dropna()} - declared
    )
    if undeclared:
        raise ModelError(
            f"the split returned training rows from season(s) {undeclared} that "
            f"are not among its declared training seasons {sorted(declared)}"
        )

    training = training.sort_values(list(RACE_KEYS), kind="stable")
    # Plain ints, not numpy scalars: these reach a JSON results file, and
    # `json` refuses an np.int64 with an error that names the type and not
    # the field it came from.
    seasons = tuple(int(s) for s in sorted(training["season"].unique()))

    # The calibration slice is the tail of training, so it stays chronological:
    # sizing intervals on a random slice would let the band see seasons the
    # shadow model was trained on.
    calibration = seasons[-params.calibration_seasons:]
    interval, trees = _calibrate(training, features, params, calibration)

    race_ids = _race_ids(training)
    booster = _fit(
        training[list(features)],
        relevance(training[TARGET_COLUMN], race_ids),
        _query_ids(race_ids),
        recency_weights(race_seasons(training), decay=params.recency_decay),
        params,
        n_estimators=trees,
    )

    dates = pd.to_datetime(training["race_date"])
    logger.info(
        "trained %s on %d rows across %d seasons (%s..%s), %d trees, decay=%.3g",
        name,
        len(training),
        len(seasons),
        seasons[0],
        seasons[-1],
        trees,
        params.recency_decay,
    )
    return TrainedRanker(
        booster=booster,
        feature_names=features,
        params=params,
        interval=interval,
        training_seasons=seasons,
        training_rows=int(len(training)),
        training_date_range=(
            dates.min().date().isoformat(),
            dates.max().date().isoformat(),
        ),
        code_commit=code_commit(repo),
        trees=trees,
        name=name,
    )


__all__ = [
    "DEFAULT_CALIBRATION_SEASONS",
    "DEFAULT_INTERVAL_COVERAGE",
    "FR404_FIELDS",
    "Hyperparameters",
    "Interval",
    "ModelError",
    "TrainedRanker",
    "code_commit",
    "race_seasons",
    "recency_weights",
    "relevance",
    "train",
]
