"""Declared circuit attributes (FR-109).

Every circuit that appears in the 2010-2025 corpus has an entry. A missing
entry raises rather than defaulting: FR-109 makes absence an error precisely
because a silently defaulted circuit type would flow into FR-505's segmentation
and FR-602's overtaking model without anyone noticing.

**These are author judgements, not measurements.** Circuit type is close to
objective; the overtaking tier is not. It is declared here so that FR-505 has
something to segment by from the first run, and so the assignment is visible and
arguable rather than buried in a model. Once a corpus exists, the tier can be
checked against observed position changes per race — a disagreement would be a
finding about this table, not about the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class CircuitType(Enum):
    """How the venue is constituted."""

    PERMANENT = "permanent"
    STREET = "street"
    #: Public roads closed for the event but forming a semi-permanent facility.
    HYBRID = "hybrid"


class OvertakingTier(Enum):
    """How hard it is to pass, coarsely. Ordered worst-to-best by `rank`."""

    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    VERY_HARD = "very_hard"

    @property
    def rank(self) -> int:
        return _TIER_RANK[self]


_TIER_RANK: Final[dict[OvertakingTier, int]] = {
    OvertakingTier.EASY: 0,
    OvertakingTier.MODERATE: 1,
    OvertakingTier.HARD: 2,
    OvertakingTier.VERY_HARD: 3,
}


class UnknownCircuitError(KeyError):
    """Raised when a circuit has no declared entry (FR-109)."""


@dataclass(frozen=True, slots=True)
class Circuit:
    circuit_id: str
    name: str
    country: str
    circuit_type: CircuitType
    overtaking_tier: OvertakingTier
    #: Decimal degrees, from the same upstream source as the race results
    #: (retrieved 2026-08-31). Typed out by hand they would be a source of
    #: silent error; fetched from the source the corpus already trusts, they
    #: are the same provenance as everything else. Used to place a race in the
    #: ERA5 grid for FR-206's climatological prior.
    latitude: float = 0.0
    longitude: float = 0.0


_P = CircuitType.PERMANENT
_S = CircuitType.STREET
_H = CircuitType.HYBRID
_EASY = OvertakingTier.EASY
_MOD = OvertakingTier.MODERATE
_HARD = OvertakingTier.HARD
_VHARD = OvertakingTier.VERY_HARD

#: All 35 circuits raced between 2010 and 2025 inclusive, enumerated from the
#: source rather than from memory.
CIRCUITS: Final[dict[str, Circuit]] = {
    c.circuit_id: c
    for c in (
        Circuit("albert_park", "Albert Park Grand Prix Circuit", "Australia", _H, _MOD, -37.8497, 144.968),
        Circuit("americas", "Circuit of the Americas", "USA", _P, _MOD, 30.1328, -97.6411),
        Circuit("bahrain", "Bahrain International Circuit", "Bahrain", _P, _EASY, 26.0325, 50.5106),
        Circuit("baku", "Baku City Circuit", "Azerbaijan", _S, _EASY, 40.3725, 49.8533),
        Circuit("buddh", "Buddh International Circuit", "India", _P, _MOD, 28.3487, 77.5331),
        Circuit("catalunya", "Circuit de Barcelona-Catalunya", "Spain", _P, _HARD, 41.57, 2.26111),
        Circuit("hockenheimring", "Hockenheimring", "Germany", _P, _MOD, 49.3278, 8.56583),
        Circuit("hungaroring", "Hungaroring", "Hungary", _P, _HARD, 47.5789, 19.2486),
        Circuit("imola", "Autodromo Enzo e Dino Ferrari", "Italy", _P, _HARD, 44.3439, 11.7167),
        Circuit("interlagos", "Autodromo Jose Carlos Pace", "Brazil", _P, _EASY, -23.7036, -46.6997),
        Circuit("istanbul", "Istanbul Park", "Turkey", _P, _MOD, 40.9517, 29.405),
        Circuit("jeddah", "Jeddah Corniche Circuit", "Saudi Arabia", _S, _MOD, 21.6319, 39.1044),
        Circuit("losail", "Losail International Circuit", "Qatar", _P, _HARD, 25.49, 51.4542),
        Circuit("marina_bay", "Marina Bay Street Circuit", "Singapore", _S, _HARD, 1.2914, 103.864),
        Circuit("miami", "Miami International Autodrome", "USA", _S, _MOD, 25.9581, -80.2389),
        Circuit("monaco", "Circuit de Monaco", "Monaco", _S, _VHARD, 43.7347, 7.42056),
        Circuit("monza", "Autodromo Nazionale di Monza", "Italy", _P, _EASY, 45.6156, 9.28111),
        Circuit("mugello", "Autodromo Internazionale del Mugello", "Italy", _P, _HARD, 43.9975, 11.3719),
        Circuit("nurburgring", "Nurburgring", "Germany", _P, _MOD, 50.3356, 6.9475),
        Circuit("portimao", "Autodromo Internacional do Algarve", "Portugal", _P, _MOD, 37.227, -8.6267),
        Circuit("red_bull_ring", "Red Bull Ring", "Austria", _P, _EASY, 47.2197, 14.7647),
        Circuit("ricard", "Circuit Paul Ricard", "France", _P, _MOD, 43.2506, 5.79167),
        Circuit("rodriguez", "Autodromo Hermanos Rodriguez", "Mexico", _P, _MOD, 19.4042, -99.0907),
        Circuit("sepang", "Sepang International Circuit", "Malaysia", _P, _EASY, 2.76083, 101.738),
        Circuit("shanghai", "Shanghai International Circuit", "China", _P, _EASY, 31.3389, 121.22),
        Circuit("silverstone", "Silverstone Circuit", "UK", _P, _MOD, 52.0786, -1.01694),
        Circuit("sochi", "Sochi Autodrom", "Russia", _H, _HARD, 43.4057, 39.9578),
        Circuit("spa", "Circuit de Spa-Francorchamps", "Belgium", _P, _EASY, 50.4372, 5.97139),
        Circuit("suzuka", "Suzuka Circuit", "Japan", _P, _HARD, 34.8431, 136.541),
        Circuit("valencia", "Valencia Street Circuit", "Spain", _S, _HARD, 39.4589, -0.331667),
        Circuit("vegas", "Las Vegas Strip Street Circuit", "USA", _S, _EASY, 36.1147, -115.173),
        Circuit("villeneuve", "Circuit Gilles Villeneuve", "Canada", _H, _MOD, 45.5, -73.5228),
        Circuit("yas_marina", "Yas Marina Circuit", "UAE", _P, _MOD, 24.4672, 54.6031),
        Circuit("yeongam", "Korean International Circuit", "Korea", _P, _MOD, 34.7333, 126.417),
        Circuit("zandvoort", "Circuit Park Zandvoort", "Netherlands", _P, _HARD, 52.3888, 4.54092),
    )
}


def get(circuit_id: str) -> Circuit:
    """Look up a circuit. Raises `UnknownCircuitError` if undeclared (FR-109)."""
    try:
        return CIRCUITS[circuit_id]
    except KeyError:
        raise UnknownCircuitError(
            f"circuit {circuit_id!r} has no declared entry. Add it to CIRCUITS "
            f"with a deliberate type and overtaking tier — FR-109 makes a "
            f"missing entry an error, not a default."
        ) from None


def assert_all_declared(circuit_ids: object) -> None:
    """Fail loudly if a collected corpus contains an undeclared circuit."""
    missing = sorted({str(c) for c in circuit_ids} - set(CIRCUITS))
    if missing:
        raise UnknownCircuitError(
            f"{len(missing)} circuit(s) in the corpus have no declared entry: "
            f"{missing}. FR-109 requires every circuit be declared."
        )


__all__ = [
    "CIRCUITS",
    "Circuit",
    "CircuitType",
    "OvertakingTier",
    "UnknownCircuitError",
    "assert_all_declared",
    "get",
]
