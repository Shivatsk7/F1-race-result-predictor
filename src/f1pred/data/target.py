"""The prediction target (FR-001, FR-001a, FR-001b, FR-002, FR-003a).

FR-001 orders a race as: classified finishers in official order, then
retirements by laps completed descending, then disqualified entrants last.
FR-001b breaks remaining ties by track position on the final completed lap,
falling back to starting grid position only where that signal is missing.

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

#: Optional column carrying track position on the final completed lap (FR-001b).
#: Absent for races where laps completed left no tie, since fetching it costs
#: roughly eleven extra requests per race.
FINAL_LAP_POSITION_COLUMN: str = "final_lap_position"

#: Sorts behind any real position, so rows lacking the signal fall through to
#: the grid tie-break rather than jumping the queue.
_NO_FINAL_LAP_POSITION: int = 10_000

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

    # FR-001b: retirements tied on laps completed are separated by track
    # position on their final completed lap. Grid position is the last resort
    # only — it is unrelated to how the race had actually unfolded, and it
    # reverses the official order in real cases (Australia 2016).
    if FINAL_LAP_POSITION_COLUMN in race.columns:
        final_lap_pos = pd.to_numeric(
            race[FINAL_LAP_POSITION_COLUMN], errors="coerce"
        ).fillna(_NO_FINAL_LAP_POSITION)
    else:
        final_lap_pos = pd.Series(_NO_FINAL_LAP_POSITION, index=race.index)

    keys = pd.DataFrame(
        {
            "outcome_rank": outcomes.map(_OUTCOME_RANK),
            # Classified entries keep the official order; retirements are ranked
            # by distance, so negate laps to sort descending within the group.
            "within_group": race["classified_position"].where(
                outcomes == ClassificationOutcome.CLASSIFIED,
                -race["laps_completed"],
            ),
            "final_lap_position": final_lap_pos,
            "grid": race["grid_position"],
        },
        index=race.index,
    )

    ordered_index = keys.sort_values(
        ["outcome_rank", "within_group", "final_lap_position", "grid"],
        kind="mergesort",
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


def races_needing_final_lap_position(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Races where laps completed leaves retirements tied (FR-001b).

    Collection uses this to fetch the lap-times detail only where it changes an
    answer. Fetching it for every race would cost roughly eleven extra requests
    per race across the whole corpus, to break ties that mostly do not exist.
    """
    starters, _ = drop_non_starters(df)
    retired = starters[
        starters["classification_outcome"]
        == ClassificationOutcome.NOT_CLASSIFIED.value
    ]
    if retired.empty:
        return []

    needed: list[tuple[int, int]] = []
    for (season, rnd), race in retired.groupby(["season", "round"], sort=True):
        counts = race["laps_completed"].value_counts()
        if (counts > 1).any():
            needed.append((int(season), int(rnd)))
    return needed


def grid_resolved_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose ordering fell through to the grid tie-break (FR-001b).

    FR-001b requires these be flagged: they are ordered by a signal unrelated to
    how the race had unfolded, so their position is the weakest in the corpus.
    """
    starters, _ = drop_non_starters(df)
    retired = starters[
        starters["classification_outcome"]
        == ClassificationOutcome.NOT_CLASSIFIED.value
    ]
    if retired.empty:
        return retired

    flagged = []
    for _, race in retired.groupby(["season", "round"], sort=True):
        tied_laps = race["laps_completed"].value_counts()
        tied_laps = set(tied_laps[tied_laps > 1].index)
        if not tied_laps:
            continue
        candidates = race[race["laps_completed"].isin(tied_laps)]
        if FINAL_LAP_POSITION_COLUMN in candidates.columns:
            candidates = candidates[candidates[FINAL_LAP_POSITION_COLUMN].isna()]
        flagged.append(candidates)
    return pd.concat(flagged) if flagged else retired.iloc[0:0]


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
    "FINAL_LAP_POSITION_COLUMN",
    "OrderingDisagreement",
    "ValidationReport",
    "build_target",
    "compute_order",
    "drop_non_starters",
    "grid_resolved_rows",
    "races_needing_final_lap_position",
    "validate_against_source",
]
