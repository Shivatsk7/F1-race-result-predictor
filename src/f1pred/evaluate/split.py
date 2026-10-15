"""The chronological split (FR-403, FR-501, FR-501a, plan D-4).

The split is an object, not an argument. Nothing here accepts a `test_size`
float, a shuffle flag, or a random state, so random splitting is not
discouraged — it is unexpressible. That matters more than a warning would: a
random split over race data leaks the future into training through every
rolling feature, and it produces a number that looks wonderful.

The object also does not receive a season list. It receives a `CorpusWindow`
(D-9), which resolves itself from what is on disk, so the boundary advances at
a season rollover with no code or config edit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Protocol, runtime_checkable

import pandas as pd

from f1pred.corpus import CorpusWindow, default_window

#: Constitution IV.2, restated by FR-501. These are floors, not defaults, and
#: FR-501a forbids lowering them to fit a thin corpus.
MIN_TRAINING_SEASONS: Final[int] = 5
MIN_HOLDOUT_SEASONS: Final[int] = 3
MIN_TOTAL_SEASONS: Final[int] = 8


#: FR-511a's label for a fold that falls short of the FR-501 training minimum.
#: It is a string in the results file rather than an absent field because a
#: reader scanning for compliance must find something, not nothing.
DIAGNOSTIC_LABEL: Final[str] = "DIAGNOSTIC"


class SplitError(ValueError):
    """Raised when a split would violate the constitutional minima."""


class FoldError(SplitError):
    """Raised when a one-season fold cannot be built or applied as specified."""


@runtime_checkable
class TrainingBoundary(Protocol):
    """What `ranker.train` needs from a split, and all it needs.

    Named so that both split types are accepted on purpose rather than because
    neither happens to be isinstance-checked. Anything that can say which
    seasons train, cut a frame along that line, and check a training set
    against it can train a model; nothing else can.
    """

    @property
    def training_seasons(self) -> tuple[int, ...]: ...

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def assert_no_leakage(self, training: pd.DataFrame) -> None: ...


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Train on the earlier seasons, evaluate on the later ones. Never mixed.

    `holdout_seasons` is how many of the window's *most recent* seasons are
    held out. It is a count rather than a fraction because a fraction would
    have to be rounded, and a rounding rule is one refactor away from becoming
    a `test_size`.
    """

    window: CorpusWindow
    holdout_seasons: int = MIN_HOLDOUT_SEASONS
    season_end_dates: dict[int, date] | None = None

    def __post_init__(self) -> None:
        if self.holdout_seasons < MIN_HOLDOUT_SEASONS:
            raise SplitError(
                f"holdout of {self.holdout_seasons} seasons is below the "
                f"Constitution IV.2 floor of {MIN_HOLDOUT_SEASONS}. FR-501a is "
                f"explicit that the minima are not lowered to fit."
            )
        seasons = self.window.resolve(self.season_end_dates)
        if len(seasons) < MIN_TOTAL_SEASONS:
            raise SplitError(
                f"{len(seasons)} seasons resolved; FR-501 requires at least "
                f"{MIN_TOTAL_SEASONS}."
            )
        if len(seasons) - self.holdout_seasons < MIN_TRAINING_SEASONS:
            raise SplitError(
                f"a {self.holdout_seasons}-season holdout leaves "
                f"{len(seasons) - self.holdout_seasons} training seasons, below "
                f"the floor of {MIN_TRAINING_SEASONS}."
            )

    # -- the boundary -------------------------------------------------------

    @property
    def seasons(self) -> tuple[int, ...]:
        return self.window.resolve(self.season_end_dates)

    @property
    def training_seasons(self) -> tuple[int, ...]:
        return self.seasons[: -self.holdout_seasons]

    @property
    def holdout(self) -> tuple[int, ...]:
        return self.seasons[-self.holdout_seasons :]

    @property
    def boundary_season(self) -> int:
        """The first held-out season. Nothing at or after it may train."""
        return self.holdout[0]

    # -- applying it --------------------------------------------------------

    def _seasons_of(self, df: pd.DataFrame) -> pd.Series:
        if "season" not in df.columns:
            raise SplitError("frame has no `season` column to split on")
        return pd.to_numeric(df["season"], errors="coerce")

    def train_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.training_seasons)

    def holdout_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.holdout)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Training rows and holdout rows, in that order.

        Rows from seasons outside the window belong to neither and are dropped
        — silently including them in training would widen the corpus past what
        the results file says it used.
        """
        return df[self.train_mask(df)].copy(), df[self.holdout_mask(df)].copy()

    def assert_no_leakage(self, training: pd.DataFrame) -> None:
        """FR-403: nothing at or after the boundary may appear in training.

        Called by the trainer rather than trusted, because the expensive
        version of this mistake is one that trains happily and reports a
        number nobody can reproduce.
        """
        seasons = set(self._seasons_of(training).dropna().astype(int))
        offending = sorted(s for s in seasons if s >= self.boundary_season)
        if offending:
            raise SplitError(
                f"training data contains season(s) {offending} at or after the "
                f"holdout boundary {self.boundary_season} (FR-403)"
            )

    # -- reporting ----------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """FR-506/FR-510: what every results file records about the split."""
        seasons = self.seasons
        shortfall = len(seasons) < MIN_TOTAL_SEASONS
        return {
            "window": self.window.describe(self.season_end_dates),
            "training_seasons": list(self.training_seasons),
            "holdout_seasons": list(self.holdout),
            "boundary_season": self.boundary_season,
            "n_training_seasons": len(self.training_seasons),
            "n_holdout_seasons": len(self.holdout),
            # FR-501a: recorded in every results file, never absorbed.
            "meets_constitutional_minimum": not shortfall,
            "minimum_required": {
                "total": MIN_TOTAL_SEASONS,
                "training": MIN_TRAINING_SEASONS,
                "holdout": MIN_HOLDOUT_SEASONS,
            },
        }


@dataclass(frozen=True, slots=True)
class SeasonFold:
    """Train on the seasons before one scored season, score that season.

    **Why this is not a `ChronologicalSplit`.** That type enforces the
    Constitution IV.2 minima structurally — at least eight seasons, at least
    five training, at least three held out — and refuses anything else at
    construction. A fold scores a *single* season, which it refuses, and that
    refusal stays: `ChronologicalSplit` is the one place a thin test set must
    be impossible rather than merely discouraged.

    One season is still the right unit in two places. The regulation-transfer
    experiment (FR-511) scores one reset season per arm, and some arms train
    on fewer than five seasons; FR-511a asks for those to be **expressible
    while labelled**, so `describe()` labels them `DIAGNOSTIC`, judged against
    `MIN_TRAINING_SEASONS`. The practice exam (Constitution IV.4c) predicts
    each of its seasons with a model trained only on earlier seasons, then
    pools three folds — the pooling, not any one fold, meets the three-season
    minimum.

    The surface is the one `ranker.train` and `harness.evaluate` consume
    (`TrainingBoundary`), so a fold and a `ChronologicalSplit` score models the
    same way even though they cannot share a type.
    """

    training_seasons: tuple[int, ...]
    evaluated_season: int

    def __post_init__(self) -> None:
        # Normalised to a tuple of plain ints: these reach a JSON results file,
        # and `json` refuses an np.int64 with an error naming the type rather
        # than the field it came from.
        seasons = tuple(int(s) for s in self.training_seasons)
        object.__setattr__(self, "training_seasons", seasons)
        object.__setattr__(self, "evaluated_season", int(self.evaluated_season))

        if not seasons:
            raise FoldError(
                f"a fold scoring {self.evaluated_season} with no training seasons "
                f"has nothing before it to train on"
            )
        if len(set(seasons)) != len(seasons):
            raise FoldError(f"training seasons {list(seasons)} contain a duplicate")
        if list(seasons) != sorted(seasons):
            raise FoldError(
                f"training seasons {list(seasons)} are not in chronological order"
            )
        offending = [s for s in seasons if s >= self.evaluated_season]
        if offending:
            raise FoldError(
                f"training season(s) {offending} are at or after the evaluated "
                f"season {self.evaluated_season}. A fold trains on seasons "
                f"*strictly* prior to the one being scored (FR-403)."
            )

    @classmethod
    def from_archive(cls, archive: Sequence[int], evaluated_season: int) -> SeasonFold:
        """Every archive season strictly before `evaluated_season`, as training.

        The archive is passed in rather than resolved here so that the same
        construction serves the real corpus and a fixture, and so that no fold
        can quietly widen its own training set.
        """
        return cls(
            training_seasons=tuple(s for s in sorted(archive) if s < evaluated_season),
            evaluated_season=evaluated_season,
        )

    # -- the boundary -------------------------------------------------------

    @property
    def seasons(self) -> tuple[int, ...]:
        return (*self.training_seasons, self.evaluated_season)

    @property
    def holdout(self) -> tuple[int, ...]:
        """One season. The whole point of a fold is that it is one."""
        return (self.evaluated_season,)

    @property
    def boundary_season(self) -> int:
        """The scored season. Nothing at or after it may train."""
        return self.evaluated_season

    @property
    def n_training_seasons(self) -> int:
        """FR-511a: the count that every fold reports and is judged on."""
        return len(self.training_seasons)

    @property
    def meets_constitutional_minimum(self) -> bool:
        """FR-511a, judged per fold against the FR-501 training floor.

        Only the training minimum applies. The three-season holdout floor is
        met by pooling folds (IV.4c) or, for a reset season, cannot be met
        without ceasing to measure a reset.
        """
        return self.n_training_seasons >= MIN_TRAINING_SEASONS

    @property
    def label(self) -> str | None:
        """`DIAGNOSTIC` when this fold falls short, `None` when it complies."""
        return None if self.meets_constitutional_minimum else DIAGNOSTIC_LABEL

    # -- applying it --------------------------------------------------------

    def _seasons_of(self, df: pd.DataFrame) -> pd.Series:
        if "season" not in df.columns:
            raise FoldError("frame has no `season` column to split on")
        return pd.to_numeric(df["season"], errors="coerce")

    def train_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.training_seasons)

    def holdout_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._seasons_of(df).isin(self.holdout)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Training rows and holdout rows, in that order.

        Seasons *after* the evaluated one belong to neither and are dropped.
        The frame handed in is usually the whole archive, so without this a
        fold scoring 2014 would train on 2015 onwards — the future of the
        season it is supposed to be meeting for the first time.
        """
        return df[self.train_mask(df)].copy(), df[self.holdout_mask(df)].copy()

    def assert_no_leakage(self, training: pd.DataFrame) -> None:
        """FR-403: nothing at or after the scored season may appear in training.

        Called by the trainer rather than trusted. A fold that leaked its own
        season into training would report a flattering score, which reads as
        good news and is the finding nobody would think to question.
        """
        seasons = set(self._seasons_of(training).dropna().astype(int))
        offending = sorted(s for s in seasons if s >= self.boundary_season)
        if offending:
            raise FoldError(
                f"training data contains season(s) {offending} at or after the "
                f"evaluated season {self.boundary_season} (FR-403)"
            )

    # -- reporting ----------------------------------------------------------

    def describe(self) -> dict[str, object]:
        """What FR-511a requires of every fold in `results/`.

        `meets_constitutional_minimum` is computed against `MIN_TRAINING_SEASONS`
        rather than a literal five, so that a change to the constitutional floor
        reaches every fold instead of leaving one asserting an old value.
        """
        return {
            "training_seasons": list(self.training_seasons),
            "n_training_seasons": self.n_training_seasons,
            "evaluated_season": self.evaluated_season,
            # Named as `ChronologicalSplit` names it, so the two records can be
            # read side by side and `harness.render_verdict` can read either.
            "holdout_seasons": list(self.holdout),
            "boundary_season": self.boundary_season,
            "meets_constitutional_minimum": self.meets_constitutional_minimum,
            "minimum_required": {"training": MIN_TRAINING_SEASONS},
            "label": self.label,
        }


def default_split(
    *, holdout_seasons: int = MIN_HOLDOUT_SEASONS, as_of: date | None = None
) -> ChronologicalSplit:
    """The project's standard split: the default window, most recent N held out."""
    return ChronologicalSplit(window=default_window(as_of), holdout_seasons=holdout_seasons)


__all__ = [
    "DIAGNOSTIC_LABEL",
    "MIN_HOLDOUT_SEASONS",
    "MIN_TOTAL_SEASONS",
    "MIN_TRAINING_SEASONS",
    "ChronologicalSplit",
    "FoldError",
    "SeasonFold",
    "SplitError",
    "TrainingBoundary",
    "default_split",
]
