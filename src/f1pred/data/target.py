"""The prediction target (FR-001, FR-001a, FR-002, FR-003a).

FR-001 orders a race as: classified finishers in official order, then
retirements by laps completed descending, then disqualified entrants last, ties
broken by starting grid position.

The T-099 spike found that the source's own `position` field already encodes
this rule. FR-001a therefore makes this module a *validator* rather than a
constructor: we compute the ordering independently, compare, and report
disagreements. A source ordering that agrees is used; one that disagrees is a
defect to investigate, not something to silently override.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from f1pred.data.schema import (
    NON_STARTER_OUTCOMES,
    ClassificationOutcome,
    SchemaError,
)

#: Sort rank by outcome. Lower sorts earlier (better finishing position).
_OUTCOME_RANK: dict[ClassificationOutcome, int] = {
    ClassificationOutcome.CLASSIFIED: 0,
    ClassificationOutcome.NOT_CLASSIFIED: 1,
    ClassificationOutcome.DISQUALIFIED: 2,
}


@dataclass(frozen=True, slots=True)
class OrderingDisagreement:
    """One race where the source order and the FR-001 rule differ."""

    season: int
    round: int
    driver_id: str
    source_position: int
    computed_position: int

    def __str__(self) -> str:
        return (
            f"{self.season} r{self.round} {self.driver_id}: "
            f"source={self.source_position} computed={self.computed_position}"
        )


@dataclass
class ValidationReport:
    """FR-001a: what the comparison found, for the data-quality record."""

    races_checked: int = 0
    races_in_agreement: int = 0
    disagreements: list[OrderingDisagreement] = field(default_factory=list)

    @property
    def races_in_disagreement(self) -> int:
        return self.races_checked - self.races_in_agreement

    @property
    def agrees(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        if self.agrees:
            return (
                f"FR-001a: source ordering agrees with the FR-001 rule on all "
                f"{self.races_checked} races checked."
            )
        return (
            f"FR-001a: {self.races_in_disagreement} of {self.races_checked} races "
            f"disagree with the FR-001 rule ({len(self.disagreements)} rows). "
            f"First: {self.disagreements[0]}"
        )


def drop_non_starters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FR-002: split a frame into starters and recorded exclusions.

    Non-start is decided by the classification outcome, never by lap count. A
    lap-1 retirement completes zero laps and *is* a race; a withdrawal completes
    zero laps and is not.
    """
    outcomes = df["classification_outcome"].map(
        lambda v: ClassificationOutcome(v) if isinstance(v, str) else v
    )
    is_non_starter = outcomes.isin(NON_STARTER_OUTCOMES)
    return df.loc[~is_non_starter].copy(), df.loc[is_non_starter].copy()


def compute_order(race: pd.DataFrame) -> pd.Series:
    """Return FR-001 classified position for one race, indexed like `race`.

    `race` must already have non-starters removed (FR-002).
    """
    if race.empty:
        return pd.Series(dtype="int64", index=race.index)

    outcomes = race["classification_outcome"].map(ClassificationOutcome)
    unknown = outcomes[~outcomes.isin(_OUTCOME_RANK)]
    if not unknown.empty:
        raise SchemaError(
            f"cannot order outcome(s) {sorted(set(unknown.map(lambda o: o.value)))}; "
            f"non-starters must be removed before ordering (FR-002)"
        )

    keys = pd.DataFrame(
        {
            "outcome_rank": outcomes.map(_OUTCOME_RANK),
            # Classified entries keep the official order; retirements are ranked
            # by distance, so negate laps to sort descending within the group.
            "within_group": race["classified_position"].where(
                outcomes == ClassificationOutcome.CLASSIFIED,
                -race["laps_completed"],
            ),
            "grid": race["grid_position"],
        },
        index=race.index,
    )

    ordered_index = keys.sort_values(
        ["outcome_rank", "within_group", "grid"], kind="mergesort"
    ).index
    return pd.Series(range(1, len(ordered_index) + 1), index=ordered_index, name="target_position").reindex(race.index)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `target_position` column computed per race under FR-001."""
    starters, _ = drop_non_starters(df)
    out = starters.copy()
    positions = pd.Series(index=out.index, dtype="int64")
    for _, race in out.groupby(["season", "round"], sort=False):
        positions.loc[race.index] = compute_order(race)
    out["target_position"] = positions
    return out


def validate_against_source(df: pd.DataFrame) -> ValidationReport:
    """FR-001a: compare the source's ordering to the FR-001 rule.

    Disagreements are reported, never silently corrected.
    """
    starters, _ = drop_non_starters(df)
    report = ValidationReport()

    for (season, rnd), race in starters.groupby(["season", "round"], sort=True):
        report.races_checked += 1
        computed = compute_order(race)
        source = race["classified_position"]
        mismatched = computed[computed != source]
        if mismatched.empty:
            report.races_in_agreement += 1
            continue
        for idx in mismatched.index:
            report.disagreements.append(
                OrderingDisagreement(
                    season=int(season),
                    round=int(rnd),
                    driver_id=str(race.at[idx, "driver_id"]),
                    source_position=int(source.at[idx]),
                    computed_position=int(computed.at[idx]),
                )
            )
    return report


__all__ = [
    "OrderingDisagreement",
    "ValidationReport",
    "build_target",
    "compute_order",
    "drop_non_starters",
    "validate_against_source",
]
