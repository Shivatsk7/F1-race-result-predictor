"""Project-wide constants and declared data.

Nothing here is inferred at run time. Regime boundaries and the corpus start are
owner decisions recorded in the spec (FR-210, OQ-6); a table is the honest way to
carry them, because F1's regulation cycle is irregular and no formula reproduces
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = PACKAGE_ROOT.parent.parent

DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
FASTF1_CACHE_DIR: Final[Path] = DATA_DIR / "fastf1_cache"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

# --------------------------------------------------------------------------
# Determinism (FR-208, FR-405, FR-604)
# --------------------------------------------------------------------------

SEED: Final[int] = 20260815

# --------------------------------------------------------------------------
# Corpus window (FR-508, FR-510, OQ-6)
# --------------------------------------------------------------------------

#: Owner decision, 2026-08-25: the corpus starts in 2010, the first season after
#: the refuelling ban, so every grid in the corpus was set on low fuel. See
#: spec §7 OQ-6.
CORPUS_START_SEASON: Final[int] = 2010

#: Default rolling-window width. Resolves to 2010-2025 as of the 2025 season.
DEFAULT_WINDOW_WIDTH: Final[int] = 16

#: FR-510: no reported result may rest on fewer than eight seasons.
MIN_WINDOW_WIDTH: Final[int] = 8

# --------------------------------------------------------------------------
# Exam plan (Constitution IV.4, change evaluation-rules-rework)
# --------------------------------------------------------------------------
#
# Every season has one fixed role. The roles are a committed record, not a
# function of the calendar or of what is on disk: the rolling window above
# moved the test seasons by itself at each rollover, and a test set that moves
# or is reused for every decision stops being a test. Only a dated amendment
# to Constitution IV.4 may change this table, and it bumps `version`.


class SeasonRole(Enum):
    """What a season may be used for (Constitution IV.4)."""

    #: Training only; never scored as a test.
    LEARNING = "learning_only"
    #: Every decision about the model, including tuning (IV.4a).
    PRACTICE = "practice_exam"
    #: Opened only when a stage of work is finished, and logged (IV.4b).
    FINAL = "final_exam"
    #: Sealed; opened once, after the rework is complete (IV.4d).
    BONUS = "bonus_exam"


@dataclass(frozen=True, slots=True)
class SeasonSpan:
    role: SeasonRole
    first: int
    last: int


@dataclass(frozen=True, slots=True)
class ExamPlan:
    version: int
    amended: str
    authority: str
    spans: tuple[SeasonSpan, ...]
    #: Set by hand once the rework is finished; the bonus exam refuses until then.
    rework_complete: bool

    def role_for_season(self, season: int) -> SeasonRole:
        """Return `season`'s role.

        Takes no date, deliberately: a role cannot depend on when it is asked.
        Raises for a season the plan does not cover rather than guessing one;
        an unplanned season is missing data, not a default (cf. FR-109).
        """
        for span in self.spans:
            if span.first <= season <= span.last:
                return span.role
        covered = f"{self.spans[0].first}-{self.spans[-1].last}"
        raise ValueError(
            f"season {season} has no role in exam plan v{self.version} "
            f"(covers {covered}); amend the plan rather than inferring one"
        )

    def seasons(self, role: SeasonRole) -> tuple[int, ...]:
        """Every season holding `role`, in order."""
        return tuple(
            season
            for span in self.spans
            if span.role is role
            for season in range(span.first, span.last + 1)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "amended": self.amended,
            "authority": self.authority,
            "rework_complete": self.rework_complete,
            "roles": {
                role.value: list(self.seasons(role)) for role in SeasonRole
            },
        }


EXAM_PLAN: Final[ExamPlan] = ExamPlan(
    version=1,
    amended="2026-10-13",
    authority="Constitution IV.4 (v1.4.0), ratified 2026-10-13",
    spans=(
        SeasonSpan(SeasonRole.LEARNING, 2010, 2019),
        SeasonSpan(SeasonRole.PRACTICE, 2020, 2022),
        SeasonSpan(SeasonRole.FINAL, 2023, 2025),
        SeasonSpan(SeasonRole.BONUS, 2026, 2026),
    ),
    rework_complete=False,
)

EXAM_PLAN_PATH: Final[Path] = RESULTS_DIR / "exam_plan.json"


def role_for_season(season: int) -> SeasonRole:
    """The committed role of `season` under `EXAM_PLAN`."""
    return EXAM_PLAN.role_for_season(season)


def write_exam_plan(path: Path | None = None) -> Path:
    """Write `EXAM_PLAN` to `results/exam_plan.json`.

    The constant is the source of truth; the file is its committed, readable
    copy, and a test fails if the two disagree. Like every results file it
    carries the corpus stamp (AC-11, AC-21); a plan is not a measurement of
    race data, so the stamp says so.
    """
    # Imported here: provenance imports this module.
    from f1pred.provenance import Scope, stamp

    record = stamp(
        EXAM_PLAN.as_dict(),
        scope=Scope.NOT_CORPUS,
        reason=(
            "the exam plan assigns roles to seasons; it is a committed decision "
            "(Constitution IV.4), not a result computed from race data"
        ),
    )
    target = path if path is not None else EXAM_PLAN_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n")
    return target


# --------------------------------------------------------------------------
# Collection tolerances (FR-104c, FR-106)
# --------------------------------------------------------------------------

#: FR-106: warn when a season's row count deviates from expectation by more.
ROW_COUNT_DEVIATION_TOLERANCE: Final[float] = 0.10

#: FR-104c: abort rather than continue past this share of failed races.
MAX_FAILURE_RATE: Final[float] = 0.10

#: FR-111: transient network failures are retried this many times.
MAX_RETRIES: Final[int] = 4

#: FR-111: the upstream default timeout is tight enough to trip intermittently.
REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0


# --------------------------------------------------------------------------
# Regulation regimes (FR-210, plan D-10)
# --------------------------------------------------------------------------


class RegimeSeverity(Enum):
    """How much of the competitive order a rule change is expected to reset."""

    MAJOR = "major"
    MODERATE = "moderate"


@dataclass(frozen=True, slots=True)
class Regime:
    start_season: int
    severity: RegimeSeverity
    description: str


#: Declared, never inferred (FR-210). Extends past the corpus deliberately: when
#: 2026 completes, the FR-511 transfer experiment gains an observation with no
#: code change.
REGIMES: Final[tuple[Regime, ...]] = (
    Regime(2010, RegimeSeverity.MAJOR, "refuelling banned; low-fuel Q3"),
    Regime(2011, RegimeSeverity.MODERATE, "DRS introduced; control tyre supplier change"),
    Regime(2014, RegimeSeverity.MAJOR, "V6 hybrid power units"),
    Regime(2017, RegimeSeverity.MAJOR, "wider cars and tyres"),
    Regime(2019, RegimeSeverity.MODERATE, "front-wing simplification"),
    Regime(2022, RegimeSeverity.MAJOR, "ground effect, full chassis reset"),
    Regime(2026, RegimeSeverity.MAJOR, "new power units, active aero, smaller cars"),
)


def regime_for_season(season: int) -> Regime:
    """Return the regime in force during `season`.

    Raises for a season before the first declared regime rather than guessing:
    an undeclared regime is missing data, not a default (cf. FR-109).
    """
    applicable = [r for r in REGIMES if r.start_season <= season]
    if not applicable:
        raise ValueError(
            f"season {season} precedes the first declared regime "
            f"({REGIMES[0].start_season}); add it to REGIMES rather than "
            f"inferring one"
        )
    return max(applicable, key=lambda r: r.start_season)


def seasons_since_regime_start(season: int) -> int:
    """Seasons elapsed since the current regime began. Zero in a reset year."""
    return season - regime_for_season(season).start_season


# --------------------------------------------------------------------------
# Cold-start fallbacks (FR-212)
# --------------------------------------------------------------------------
#
# FR-212 forbids the field mean, and rightly: filling a rookie's form features
# with 0.5 asserts average competitiveness on no evidence at all. These values
# are instead *measured* — the mean within-race finishing percentile actually
# achieved by drivers and constructors on their corpus debut.
#
# Measured over 2011-2014 only, deliberately. Those seasons sit inside any
# chronological training split (FR-501), so the constant cannot carry holdout
# information back into a training feature. 2010 is excluded because every
# entrant is a debutant at the corpus start, which would measure the corpus
# boundary rather than debut performance.
#
# The samples are small — 20 driver debuts and 6 constructor debuts — so these
# are weak estimates, and that is exactly why every row they touch is flagged
# LOW_INFORMATION (FR-212) rather than being presented as an ordinary
# prediction. For reference the contemporaneous field mean is 0.52.

#: Mean finishing percentile of a driver's corpus debut, 2011-2014 (n=20).
COLD_START_DRIVER_PERCENTILE: Final[float] = 0.35

#: Mean finishing percentile of a constructor's corpus debut, 2011-2014 (n=6).
COLD_START_CONSTRUCTOR_PERCENTILE: Final[float] = 0.33


# --------------------------------------------------------------------------
# Shrinkage (FR-212a)
# --------------------------------------------------------------------------
#
# The number of prior races at which an observation and the declared prior
# carry equal weight in `priors.shrink_towards`.
#
# Derived, not tuned. A driver's within-race finishing percentile after one
# race correlates 0.41 with where they settle over ten or more (measured
# 2026-08-31 across 76 drivers). Treating that correlation as the reliability
# of a one-race sample gives a weight of r^2 = 0.17 on the observation, so
# 1 / (1 + k) = 0.17 and k is very close to five.
#
# Tuning this against holdout results would be fitting a feature to the
# outcome, so it stays where the measurement put it.
SHRINKAGE_STRENGTH: Final[float] = 5.0
