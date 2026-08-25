"""Project-wide constants and declared data.

Nothing here is inferred at run time. Regime boundaries and the corpus start are
owner decisions recorded in the spec (FR-210, OQ-6); a table is the honest way to
carry them, because F1's regulation cycle is irregular and no formula reproduces
it.
"""

from __future__ import annotations

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
