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
        Circuit("albert_park", "Albert Park Grand Prix Circuit", "Australia", _H, _MOD),
        Circuit("americas", "Circuit of the Americas", "USA", _P, _MOD),
        Circuit("bahrain", "Bahrain International Circuit", "Bahrain", _P, _EASY),
        Circuit("baku", "Baku City Circuit", "Azerbaijan", _S, _EASY),
        Circuit("buddh", "Buddh International Circuit", "India", _P, _MOD),
        Circuit("catalunya", "Circuit de Barcelona-Catalunya", "Spain", _P, _HARD),
        Circuit("hockenheimring", "Hockenheimring", "Germany", _P, _MOD),
        Circuit("hungaroring", "Hungaroring", "Hungary", _P, _HARD),
        Circuit("imola", "Autodromo Enzo e Dino Ferrari", "Italy", _P, _HARD),
        Circuit("interlagos", "Autodromo Jose Carlos Pace", "Brazil", _P, _EASY),
        Circuit("istanbul", "Istanbul Park", "Turkey", _P, _MOD),
        Circuit("jeddah", "Jeddah Corniche Circuit", "Saudi Arabia", _S, _MOD),
        Circuit("losail", "Losail International Circuit", "Qatar", _P, _HARD),
        Circuit("marina_bay", "Marina Bay Street Circuit", "Singapore", _S, _HARD),
        Circuit("miami", "Miami International Autodrome", "USA", _S, _MOD),
        Circuit("monaco", "Circuit de Monaco", "Monaco", _S, _VHARD),
        Circuit("monza", "Autodromo Nazionale di Monza", "Italy", _P, _EASY),
        Circuit("mugello", "Autodromo Internazionale del Mugello", "Italy", _P, _HARD),
        Circuit("nurburgring", "Nurburgring", "Germany", _P, _MOD),
        Circuit("portimao", "Autodromo Internacional do Algarve", "Portugal", _P, _MOD),
        Circuit("red_bull_ring", "Red Bull Ring", "Austria", _P, _EASY),
        Circuit("ricard", "Circuit Paul Ricard", "France", _P, _MOD),
        Circuit("rodriguez", "Autodromo Hermanos Rodriguez", "Mexico", _P, _MOD),
        Circuit("sepang", "Sepang International Circuit", "Malaysia", _P, _EASY),
        Circuit("shanghai", "Shanghai International Circuit", "China", _P, _EASY),
        Circuit("silverstone", "Silverstone Circuit", "UK", _P, _MOD),
        Circuit("sochi", "Sochi Autodrom", "Russia", _H, _HARD),
        Circuit("spa", "Circuit de Spa-Francorchamps", "Belgium", _P, _EASY),
        Circuit("suzuka", "Suzuka Circuit", "Japan", _P, _HARD),
        Circuit("valencia", "Valencia Street Circuit", "Spain", _S, _HARD),
        Circuit("vegas", "Las Vegas Strip Street Circuit", "USA", _S, _EASY),
        Circuit("villeneuve", "Circuit Gilles Villeneuve", "Canada", _H, _MOD),
        Circuit("yas_marina", "Yas Marina Circuit", "UAE", _P, _MOD),
        Circuit("yeongam", "Korean International Circuit", "Korea", _P, _MOD),
        Circuit("zandvoort", "Circuit Park Zandvoort", "Netherlands", _P, _HARD),
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
