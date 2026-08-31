"""The registered features (FR-202, FR-203, FR-209, FR-210, FR-211).

Every feature the project computes is declared here. Importing this module is
what populates `registry.REGISTRY`; `build` and `checks` read the result. There
is no list of feature names anywhere else in the codebase, which is the point
of FR-202 — v0's hand-maintained list is what let a post-race weather flag sit
in the model for four months.

**Shape of this module.** Historical performance is computed twice: once as an
absolute quantity in the INTERMEDIATE group, and once as that quantity's rank
against the cars that actually turned up. Only the second reaches a model,
because FR-209 says a pace number that means one thing in 2013 and another in
2023 is not a feature, it is a date in disguise.

The disposition of every v0 feature — ported here, or dropped and why — is
recorded in `disposition.py` and tested, so "we ported the features" is a
checkable claim rather than a memory.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from f1pred.config import RegimeSeverity, regime_for_season
from f1pred.config import seasons_since_regime_start as _regime_age
from f1pred.data.schema import ClassificationOutcome, WeekendFormat
from f1pred.features.priors import (
    RACE_KEYS,
    prior_race_count,
    prior_rate,
    prior_stat,
    within_race_percentile,
)
from f1pred.features.ratings import sequential_elo
from f1pred.features.registry import Availability, FeatureGroup, feature
from f1pred.reference import circuits

PRE = Availability.PRE_RACE
POST = Availability.POST_RACE
INTERMEDIATE = FeatureGroup.INTERMEDIATE

#: The FR-001 total order, added by `data.target.build_target` before any
#: feature runs. Named once so no feature spells it as a literal.
FINISH = "target_position"


def _relative(
    base: str,
    *,
    higher_is_better: bool,
    description: str,
    evidence: str,
    name: str | None = None,
) -> None:
    """Register `base`'s field-relative form as a primary feature (FR-209).

    Generated rather than hand-written because these nineteen declarations are
    identical but for two words, and nineteen copies of the same function is
    nineteen chances to rank one of them the wrong way round.
    """

    def compute(df: pd.DataFrame, _base: str = base, _hib: bool = higher_is_better):
        return within_race_percentile(df, df[_base], higher_is_better=_hib)

    compute.__name__ = name or f"{base}_pct"
    compute.__doc__ = description
    feature(
        PRE,
        depends_on=[base, evidence],
        summarises_history=True,
        relative=True,
        evidence=evidence,
        name=name or f"{base}_pct",
    )(compute)


# --------------------------------------------------------------------------
# Starting position and race context — knowable at lights out by definition
# --------------------------------------------------------------------------


@feature(PRE, depends_on=["grid_position"])
def grid_slot(df: pd.DataFrame) -> pd.Series:
    """Where the car starts. The single strongest signal in the sport.

    Named `grid_slot` rather than `grid_position` because a feature may not
    share a name with the raw column it reads: `depends_on` resolves against
    both, and a shadowed name is an unresolvable dependency.
    """
    return df["grid_position"].astype("float64")


@feature(PRE, depends_on=["grid_position"])
def grid_position_sq(df: pd.DataFrame) -> pd.Series:
    """Grid position squared: the cost of a place lost grows down the order."""
    return df["grid_position"].astype("float64") ** 2


@feature(PRE, depends_on=["grid_position", "constructor_id", *RACE_KEYS])
def grid_rank_vs_teammate(df: pd.DataFrame) -> pd.Series:
    """Rank within the constructor's own cars on this grid — 1 is ahead.

    v0 called this `quali_vs_teammate`. It is the cleanest available read on a
    driver's pace with the car held constant, which is why it survives even
    though the qualifying times it would ideally use are not in the corpus.
    """
    return (
        df.groupby([*RACE_KEYS, "constructor_id"], dropna=False, sort=False)[
            "grid_position"
        ]
        .rank(method="dense")
        .astype("float64")
    )


@feature(PRE, depends_on=[*RACE_KEYS])
def field_size(df: pd.DataFrame) -> pd.Series:
    """How many cars started. Ranges from 20 to 26 across the corpus."""
    return (
        df.groupby(list(RACE_KEYS), dropna=False, sort=False)["driver_id"]
        .transform("size")
        .astype("float64")
    )


@feature(PRE, depends_on=["weekend_format"])
def is_sprint_weekend(df: pd.DataFrame) -> pd.Series:
    """Whether a sprint race preceded the Grand Prix (FR-108, FR-211).

    Knowable months in advance — the calendar declares it. The sprint's
    *result* is a further pre-race signal that FR-211 admits; it is not yet
    collected, so this flag carries only the weekend's structure.
    """
    return (df["weekend_format"] != WeekendFormat.CONVENTIONAL.value).astype("float64")


# --------------------------------------------------------------------------
# The sprint race (FR-211, T-212)
# --------------------------------------------------------------------------
#
# A sprint runs on Saturday and the Grand Prix on Sunday, so its result is
# knowable before lights out and is admissible pre-race under Constitution
# III.1. Twenty-four weekends in the corpus have one; the other 305 do not, and
# for those the value is null and stays null. FR-211 is explicit that a neutral
# fill here would be a claim about a race that never happened, so every feature
# in this section is declared `never_impute` and the FR-212 cold-start
# fallbacks skip it.


@feature(
    PRE,
    depends_on=["sprint_position", *RACE_KEYS],
    summarises_history=True,
    relative=True,
    never_impute=True,
)
def sprint_finish_pct(df: pd.DataFrame) -> pd.Series:
    """Sprint finishing position ranked against the same weekend's field.

    Relative for the same reason every other form feature is (FR-209), and
    conveniently so: a sprint field is exactly the field that lines up for the
    Grand Prix the next day, which makes the comparison unusually clean.
    """
    return within_race_percentile(
        df, pd.to_numeric(df["sprint_position"], errors="coerce"), higher_is_better=False
    )


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["sprint_position", "sprint_grid_position"],
    summarises_history=True,
    never_impute=True,
)
def sprint_places_gained(df: pd.DataFrame) -> pd.Series:
    """Places gained in the sprint, from its own grid. Positive is forward."""
    return pd.to_numeric(df["sprint_grid_position"], errors="coerce") - pd.to_numeric(
        df["sprint_position"], errors="coerce"
    )


@feature(
    PRE,
    depends_on=["sprint_places_gained", *RACE_KEYS],
    summarises_history=True,
    relative=True,
    never_impute=True,
)
def sprint_places_gained_pct(df: pd.DataFrame) -> pd.Series:
    """Sprint race-day progress ranked against the same weekend's field.

    Carries something the finishing position alone does not: a car that started
    the sprint last and finished tenth is a different proposition on Sunday
    from one that qualified tenth and stayed there.
    """
    return within_race_percentile(
        df, df["sprint_places_gained"], higher_is_better=True
    )


# --------------------------------------------------------------------------
# Circuit attributes — declared data (FR-109, T-101b)
# --------------------------------------------------------------------------


def _circuit_attr(df: pd.DataFrame, fn: Callable[[circuits.Circuit], float]) -> pd.Series:
    """Map each row's circuit to a declared attribute. Unknown ids raise."""
    unique = {cid: fn(circuits.get(cid)) for cid in df["circuit_id"].unique()}
    return df["circuit_id"].map(unique).astype("float64")


@feature(PRE, depends_on=["circuit_id"])
def is_street_circuit(df: pd.DataFrame) -> pd.Series:
    """Street venue: overtaking is scarce and the grid holds."""
    return _circuit_attr(df, lambda c: float(c.circuit_type is circuits.CircuitType.STREET))


@feature(PRE, depends_on=["circuit_id"])
def is_permanent_circuit(df: pd.DataFrame) -> pd.Series:
    """Purpose-built permanent facility."""
    return _circuit_attr(
        df, lambda c: float(c.circuit_type is circuits.CircuitType.PERMANENT)
    )


@feature(PRE, depends_on=["circuit_id"])
def is_hybrid_circuit(df: pd.DataFrame) -> pd.Series:
    """Public roads forming a semi-permanent facility."""
    return _circuit_attr(
        df, lambda c: float(c.circuit_type is circuits.CircuitType.HYBRID)
    )


@feature(PRE, depends_on=["circuit_id"])
def overtaking_difficulty(df: pd.DataFrame) -> pd.Series:
    """Declared overtaking tier, 0 (easy) to 3 (very hard)."""
    return _circuit_attr(df, lambda c: float(c.overtaking_tier.rank))


@feature(PRE, depends_on=["grid_position", "overtaking_difficulty"])
def grid_x_overtaking(df: pd.DataFrame) -> pd.Series:
    """Grid position weighted by how hard the circuit is to pass at.

    Replaces v0's `weighted_grid_position`, which multiplied grid position by a
    logistic of five undeclared track indices. This uses the one circuit
    attribute the project actually declares (FR-109) and leaves the shape of
    the interaction to the model.
    """
    return df["grid_position"].astype("float64") * (1.0 + df["overtaking_difficulty"])


@feature(PRE, depends_on=["grid_position", "is_street_circuit"])
def grid_x_street(df: pd.DataFrame) -> pd.Series:
    """Grid position on a street circuit, zero elsewhere (v0's street pressure)."""
    return df["grid_position"].astype("float64") * df["is_street_circuit"]


# --------------------------------------------------------------------------
# Regulation regime — declared data (FR-210, plan D-10)
# --------------------------------------------------------------------------


@feature(PRE, depends_on=["season"])
def regime_id(df: pd.DataFrame) -> pd.Series:
    """Start season of the regulation regime in force (FR-210).

    Knowable before a season begins, so Constitution III.1 is satisfied
    trivially. Carried as the start season rather than an index, so the value
    stays meaningful when a regime is inserted into the declared table.
    """
    unique = {s: float(regime_for_season(int(s)).start_season) for s in df["season"].unique()}
    return df["season"].map(unique).astype("float64")


@feature(PRE, depends_on=["season"])
def seasons_since_regime_start(df: pd.DataFrame) -> pd.Series:
    """Seasons elapsed since the current regime began; zero in a reset year."""
    unique = {s: float(_regime_age(int(s))) for s in df["season"].unique()}
    return df["season"].map(unique).astype("float64")


@feature(PRE, depends_on=["season"])
def regime_is_major(df: pd.DataFrame) -> pd.Series:
    """Whether the regime in force began with a major regulation reset."""
    unique = {
        s: float(regime_for_season(int(s)).severity is RegimeSeverity.MAJOR)
        for s in df["season"].unique()
    }
    return df["season"].map(unique).astype("float64")


# --------------------------------------------------------------------------
# Experience — participation counts, not performance (so FR-209 does not bind)
# --------------------------------------------------------------------------


@feature(PRE, depends_on=["driver_id"])
def driver_prior_races(df: pd.DataFrame) -> pd.Series:
    """Races this driver has already contested in the corpus. Zero = rookie."""
    return prior_race_count(df, entity="driver_id")


@feature(PRE, depends_on=["constructor_id"])
def constructor_prior_races(df: pd.DataFrame) -> pd.Series:
    """Races this constructor has already contested. Zero = new team."""
    return prior_race_count(df, entity="constructor_id")


@feature(PRE, group=INTERMEDIATE, depends_on=["driver_id", "season"])
def driver_prior_races_season(df: pd.DataFrame) -> pd.Series:
    """Races this driver has run *this season*. Zero every January.

    A separate counter from the career one because the season-scoped features
    reset annually, and 26% of corpus rows carry a season rate built on four
    races or fewer. Trusting those as if they were career-length samples is
    the specific error shrinkage exists to correct.
    """
    return prior_race_count(df, entity=["season", "driver_id"])


@feature(PRE, group=INTERMEDIATE, depends_on=["constructor_id", "season"])
def constructor_prior_races_season(df: pd.DataFrame) -> pd.Series:
    """Races this constructor has run this season."""
    return prior_race_count(df, entity=["season", "constructor_id"])


@feature(PRE, group=INTERMEDIATE, depends_on=["driver_id", "circuit_id"])
def driver_prior_visits(df: pd.DataFrame) -> pd.Series:
    """Previous visits this driver has made to this circuit.

    Usually a very small number — a driver sees a circuit once a year — which
    is why the circuit-history features are the thinnest in the set and the
    ones shrinkage moves most.
    """
    return prior_race_count(df, entity=["driver_id", "circuit_id"])


@feature(PRE, group=INTERMEDIATE, depends_on=["constructor_id", "circuit_id"])
def constructor_prior_visits(df: pd.DataFrame) -> pd.Series:
    """Previous visits this constructor has made to this circuit."""
    return prior_race_count(df, entity=["constructor_id", "circuit_id"])


# --------------------------------------------------------------------------
# Driver form — absolute intermediates (FR-203)
# --------------------------------------------------------------------------


@feature(PRE, group=INTERMEDIATE, depends_on=["driver_id", FINISH], summarises_history=True)
def driver_finish_last3(df: pd.DataFrame) -> pd.Series:
    """Mean finishing position over the driver's three previous races."""
    return prior_stat(df, entity="driver_id", value=FINISH, window=3)


@feature(PRE, group=INTERMEDIATE, depends_on=["driver_id", FINISH], summarises_history=True)
def driver_finish_last5(df: pd.DataFrame) -> pd.Series:
    """Mean finishing position over the driver's five previous races."""
    return prior_stat(df, entity="driver_id", value=FINISH, window=5)


@feature(
    PRE, group=INTERMEDIATE, depends_on=["driver_id", "grid_position"], summarises_history=True
)
def driver_grid_last3(df: pd.DataFrame) -> pd.Series:
    """Mean grid position over the driver's three previous races."""
    return prior_stat(df, entity="driver_id", value="grid_position", window=3)


@feature(PRE, group=INTERMEDIATE, depends_on=["driver_id", "points"], summarises_history=True)
def driver_points_last3(df: pd.DataFrame) -> pd.Series:
    """Mean points scored over the driver's three previous races."""
    return prior_stat(df, entity="driver_id", value="points", window=3)


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["driver_id", "grid_position", FINISH],
    summarises_history=True,
)
def driver_gain_last5(df: pd.DataFrame) -> pd.Series:
    """Mean places gained from the grid over the driver's five previous races."""
    work = df.copy()
    work["_gain"] = df["grid_position"].astype("float64") - pd.to_numeric(
        df[FINISH], errors="coerce"
    )
    return prior_stat(work, entity="driver_id", value="_gain", window=5)


@feature(
    PRE, group=INTERMEDIATE, depends_on=["driver_id", "season", FINISH], summarises_history=True
)
def driver_win_rate_season(df: pd.DataFrame) -> pd.Series:
    """Share of this season's earlier races the driver won.

    Scoped to the season, so the rate speaks about the current car rather than
    about a career. Resets at each season boundary, which is correct: a driver
    who won six races in a title-winning car is not thereby a favourite in a
    midfield one.
    """
    return prior_rate(
        df, entity=["season", "driver_id"], mask=pd.to_numeric(df[FINISH]) == 1
    )


@feature(
    PRE, group=INTERMEDIATE, depends_on=["driver_id", "season", FINISH], summarises_history=True
)
def driver_podium_rate_season(df: pd.DataFrame) -> pd.Series:
    """Share of this season's earlier races the driver finished in the top three."""
    return prior_rate(
        df, entity=["season", "driver_id"], mask=pd.to_numeric(df[FINISH]) <= 3
    )


@feature(
    PRE, group=INTERMEDIATE, depends_on=["driver_id", "season", "points"], summarises_history=True
)
def driver_points_per_race_season(df: pd.DataFrame) -> pd.Series:
    """Mean points per race so far this season, excluding this race."""
    return prior_stat(df, entity=["season", "driver_id"], value="points")


@feature(PRE, group=INTERMEDIATE, depends_on=["driver_id", FINISH], summarises_history=True)
def driver_elo_race(df: pd.DataFrame) -> pd.Series:
    """Pairwise race-execution rating as it stood before this race."""
    return sequential_elo(df, entity="driver_id", outcome=FINISH, k_factor=24.0)


@feature(
    PRE, group=INTERMEDIATE, depends_on=["driver_id", "grid_position"], summarises_history=True
)
def driver_elo_quali(df: pd.DataFrame) -> pd.Series:
    """Pairwise qualifying-pace rating as it stood before this race."""
    return sequential_elo(df, entity="driver_id", outcome="grid_position", k_factor=20.0)


# --------------------------------------------------------------------------
# Constructor form — aggregated to the team-race grain before any shift
# --------------------------------------------------------------------------


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["constructor_id", "season", "points"],
    summarises_history=True,
)
def constructor_points_per_race_season(df: pd.DataFrame) -> pd.Series:
    """Mean points the team scored per earlier race this season, both cars."""
    return prior_stat(
        df, entity=["season", "constructor_id"], value="points", within_race="sum"
    )


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["constructor_id", "season", "classification_outcome"],
    summarises_history=True,
)
def constructor_reliability_season(df: pd.DataFrame) -> pd.Series:
    """Share of the team's earlier car-starts this season that were classified.

    Read from the classification outcome, not from a regex over the status
    string. v0 matched `Finished|Lap`, which counted "+1 Lap" as a finish and
    "Lapped" as one too, but silently scored a classified retirement — Bahrain
    2022, both Red Bulls — as a failure.
    """
    classified = (
        df["classification_outcome"] == ClassificationOutcome.CLASSIFIED.value
    )
    return prior_rate(df, entity=["season", "constructor_id"], mask=classified)


@feature(
    PRE, group=INTERMEDIATE, depends_on=["constructor_id", FINISH], summarises_history=True
)
def constructor_finish_last3(df: pd.DataFrame) -> pd.Series:
    """Mean finishing position of the team's cars over its three previous races."""
    return prior_stat(df, entity="constructor_id", value=FINISH, window=3)


# --------------------------------------------------------------------------
# Circuit history — a driver's record at this venue, before today
# --------------------------------------------------------------------------


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["driver_id", "circuit_id", FINISH],
    summarises_history=True,
)
def driver_circuit_finish_mean(df: pd.DataFrame) -> pd.Series:
    """Mean finish at this circuit in the driver's earlier visits."""
    return prior_stat(df, entity=["driver_id", "circuit_id"], value=FINISH)


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["driver_id", "circuit_id", "grid_position"],
    summarises_history=True,
)
def driver_circuit_grid_mean(df: pd.DataFrame) -> pd.Series:
    """Mean grid slot at this circuit in the driver's earlier visits."""
    return prior_stat(df, entity=["driver_id", "circuit_id"], value="grid_position")


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["driver_id", "circuit_id", FINISH],
    summarises_history=True,
)
def driver_circuit_best_finish(df: pd.DataFrame) -> pd.Series:
    """Best finish at this circuit in the driver's earlier visits."""
    return prior_stat(df, entity=["driver_id", "circuit_id"], value=FINISH, how="min")


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=["constructor_id", "circuit_id", FINISH],
    summarises_history=True,
)
def constructor_circuit_finish_mean(df: pd.DataFrame) -> pd.Series:
    """Mean finish of the team's cars at this circuit in earlier visits."""
    return prior_stat(df, entity=["constructor_id", "circuit_id"], value=FINISH)


# --------------------------------------------------------------------------
# Field-relative forms — the primary set (FR-209)
# --------------------------------------------------------------------------

#: base feature, higher-is-better, description, evidence counter.
#: The fourth column is what makes shrinkage correct rather than approximate:
#: a season rate backed by two races and a career rating backed by two hundred
#: must not be trusted equally, and only a per-feature declaration can say so.
_RELATIVE: tuple[tuple[str, bool, str, str], ...] = (
    ("driver_finish_last3", False, "Recent finishing form, ranked against today's field.", "driver_prior_races"),
    ("driver_finish_last5", False, "Five-race finishing form, ranked against today's field.", "driver_prior_races"),
    ("driver_grid_last3", False, "Recent qualifying form, ranked against today's field.", "driver_prior_races"),
    ("driver_points_last3", True, "Recent points scoring, ranked against today's field.", "driver_prior_races"),
    ("driver_gain_last5", True, "Recent race-day places gained, ranked against today's field.", "driver_prior_races"),
    ("driver_win_rate_season", True, "Season win rate, ranked against today's field.", "driver_prior_races_season"),
    ("driver_podium_rate_season", True, "Season podium rate, ranked against today's field.", "driver_prior_races_season"),
    ("driver_points_per_race_season", True, "Season points rate, ranked against today's field.", "driver_prior_races_season"),
    ("driver_elo_race", True, "Race-execution rating, ranked against today's field.", "driver_prior_races"),
    ("driver_elo_quali", True, "Qualifying-pace rating, ranked against today's field.", "driver_prior_races"),
    ("constructor_points_per_race_season", True, "Team season points rate, ranked against today's field.", "constructor_prior_races_season"),
    ("constructor_reliability_season", True, "Team season reliability, ranked against today's field.", "constructor_prior_races_season"),
    ("constructor_finish_last3", False, "Team recent finishing form, ranked against today's field.", "constructor_prior_races"),
    ("driver_circuit_finish_mean", False, "Record at this circuit, ranked against today's field.", "driver_prior_visits"),
    ("driver_circuit_grid_mean", False, "Qualifying record at this circuit, ranked against today's field.", "driver_prior_visits"),
    ("driver_circuit_best_finish", False, "Best result at this circuit, ranked against today's field.", "driver_prior_visits"),
    ("constructor_circuit_finish_mean", False, "Team record at this circuit, ranked against today's field.", "constructor_prior_visits"),
)

for _base, _hib, _doc, _evidence in _RELATIVE:
    _relative(_base, higher_is_better=_hib, description=_doc, evidence=_evidence)


# --------------------------------------------------------------------------
# Composite prior strength — v0's `_add_strength_context`, made relative
# --------------------------------------------------------------------------

#: v0 blended these four with weights 0.35/0.20/0.20/0.25 chosen by nobody in
#: particular. The weights are kept because they are at least declared and the
#: blend's only job is to produce a ranking, but the components are registered
#: separately so the model can disagree with them.
_STRENGTH_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("driver_elo_race_pct", 0.35),
    ("driver_elo_quali_pct", 0.20),
    ("driver_points_last3_pct", 0.20),
    ("constructor_points_per_race_season_pct", 0.25),
)


@feature(
    PRE,
    group=INTERMEDIATE,
    depends_on=[name for name, _ in _STRENGTH_WEIGHTS],
    summarises_history=True,
)
def prior_strength_score(df: pd.DataFrame) -> pd.Series:
    """Weighted blend of driver rating, qualifying rating, form, and team pace."""
    total = pd.Series(0.0, index=df.index, dtype="float64")
    for name, weight in _STRENGTH_WEIGHTS:
        total = total + weight * df[name].fillna(0.5)
    return total


@feature(PRE, depends_on=["prior_strength_score"], summarises_history=True, relative=True)
def strength_rank_prior(df: pd.DataFrame) -> pd.Series:
    """Where the field's prior strength ranks this car, 1 = strongest."""
    return (
        df.groupby(list(RACE_KEYS), dropna=False, sort=False)["prior_strength_score"]
        .rank(method="dense", ascending=False)
        .astype("float64")
    )


@feature(
    PRE,
    depends_on=["strength_rank_prior", "grid_position"],
    summarises_history=True,
    relative=True,
)
def overqualified_by_grid(df: pd.DataFrame) -> pd.Series:
    """Prior strength rank minus grid slot: positive means it out-qualified form.

    v0 registered this *and* its exact negation, `underqualified_by_grid`. One
    of the two was pure noise in every importance ranking it appeared in.
    """
    return df["strength_rank_prior"] - df["grid_position"].astype("float64")


# --------------------------------------------------------------------------
# Telemetry group — a separate arm, never the primary set (FR-213..215)
# --------------------------------------------------------------------------
#
# Friday practice is the only place in a weekend where long-run pace is visible
# before the race, so these are admissible pre-race under Constitution III.1.
# They are still kept out of the primary matrix, because the source begins in
# 2018 and the corpus begins in 2010: 3,453 of 6,911 rows have no practice data
# at all. FR-214 forbids imputing that away and FR-215 makes the group a
# separate arm measured on covered seasons only.
#
# Every one of these reads a column that `data.telemetry.attach` puts on the
# frame. Build the group with `build_group(attach(corpus, practice),
# FeatureGroup.TELEMETRY)`; without the attach, dependency resolution fails
# loudly rather than quietly producing a matrix of nulls.

TELEMETRY = FeatureGroup.TELEMETRY


@feature(
    PRE,
    group=TELEMETRY,
    depends_on=["long_run_pace_s", *RACE_KEYS],
    summarises_history=True,
    relative=True,
    never_impute=True,
)
def practice_long_run_pace_pct(df: pd.DataFrame) -> pd.Series:
    """Long-run practice pace ranked against the same weekend's field.

    Relative for a reason beyond FR-209 here: a lap time in seconds is a fact
    about the circuit, not the car, so the absolute value carries no signal at
    all across weekends. The ranking within one Friday is the whole content.
    """
    return within_race_percentile(
        df, pd.to_numeric(df["long_run_pace_s"], errors="coerce"), higher_is_better=False
    )


@feature(
    PRE,
    group=TELEMETRY,
    depends_on=["stint_pace_trend_s_per_lap", *RACE_KEYS],
    summarises_history=True,
    relative=True,
    never_impute=True,
)
def practice_stint_trend_pct(df: pd.DataFrame) -> pd.Series:
    """How well pace held up over a stint, ranked against the weekend's field.

    The underlying slope confounds tyre wear with fuel burn and track
    evolution, so its level means little — see the note on the collected
    column. Ranking it within a weekend cancels most of that: every car on
    Friday burns fuel on the same rubbering-in track.
    """
    return within_race_percentile(
        df,
        pd.to_numeric(df["stint_pace_trend_s_per_lap"], errors="coerce"),
        higher_is_better=False,
    )


@feature(
    PRE,
    group=TELEMETRY,
    depends_on=["best_lap_s", *RACE_KEYS],
    summarises_history=True,
    relative=True,
    never_impute=True,
)
def practice_best_lap_pct(df: pd.DataFrame) -> pd.Series:
    """Best practice lap ranked against the weekend's field.

    Overlaps with qualifying and is expected to add little once the grid is
    known. It is registered anyway so the separate-arm comparison of FR-215 is
    measuring the whole telemetry group rather than a hand-picked part of it.
    """
    return within_race_percentile(
        df, pd.to_numeric(df["best_lap_s"], errors="coerce"), higher_is_better=False
    )


@feature(
    PRE,
    group=TELEMETRY,
    depends_on=["long_run_laps"],
    never_impute=True,
)
def practice_long_run_laps(df: pd.DataFrame) -> pd.Series:
    """How many long-run laps the pace figure rests on.

    Not a performance measure — it is the sample size behind the three above,
    and a model is entitled to trust a pace built on twenty laps more than one
    built on five.
    """
    return pd.to_numeric(df["long_run_laps"], errors="coerce").astype("float64")


# --------------------------------------------------------------------------
# Weather — the same source in two roles, kept apart (FR-206, FR-751a)
# --------------------------------------------------------------------------
#
# One endpoint serves both. Ask ERA5 for the years before a race and you get a
# climatological prior, which Constitution III.1 admits. Ask for the race day
# and you get what happened, which III.2 forbids and which is the shape of the
# defect that inflated every v0 number. Nothing in the response distinguishes
# them, so `data.weather.climatological_prior` filters to seasons strictly
# before the race's own — a cut-off derived from the race date, which no caller
# can widen — and the race-day columns are registered POST_RACE below.
#
# Attach with `data.weather.attach(corpus, race_weather_table(archive, corpus))`.


@feature(PRE, depends_on=["weather_prior_wet_probability"], never_impute=True)
def weather_prior_wet(df: pd.DataFrame) -> pd.Series:
    """How often this circuit is wet at this point in the calendar, historically.

    Built only from years preceding the race. Sepang tops the corpus at 0.87,
    which is the right answer and a useful sanity check on the whole
    construction.
    """
    return pd.to_numeric(df["weather_prior_wet_probability"], errors="coerce")


@feature(PRE, depends_on=["weather_prior_temp_max_c"], never_impute=True)
def weather_prior_temp(df: pd.DataFrame) -> pd.Series:
    """Typical maximum temperature at this circuit at this time of year."""
    return pd.to_numeric(df["weather_prior_temp_max_c"], errors="coerce")


@feature(PRE, depends_on=["weather_prior_wind_kmh"], never_impute=True)
def weather_prior_wind(df: pd.DataFrame) -> pd.Series:
    """Typical maximum wind speed at this circuit at this time of year."""
    return pd.to_numeric(df["weather_prior_wind_kmh"], errors="coerce")


# --------------------------------------------------------------------------
# Post-race features — oracle set only (FR-207, Constitution III.2)
# --------------------------------------------------------------------------
#
# These exist so FR-207's upper bound can be measured and so the leakage checks
# of FR-204 and FR-205 have something real to catch. Nothing here may reach the
# primary matrix, and the static check fails the build if it does.


@feature(POST, depends_on=["weather_actual_precip_mm"])
def weather_actual_precip(df: pd.DataFrame) -> pd.Series:
    """Rain that actually fell on race day. **Post-race. Oracle set only.**

    Registered so FR-207 can put a number on what a weather leak is worth —
    the question v0 answered by accident, over four months, in the wrong
    direction. The static check keeps it out of the primary matrix and the
    behavioural check catches any pre-race feature that reads it.
    """
    return pd.to_numeric(df["weather_actual_precip_mm"], errors="coerce")


@feature(POST, depends_on=["weather_actual_temp_max_c"])
def weather_actual_temp(df: pd.DataFrame) -> pd.Series:
    """Race-day maximum temperature. **Post-race. Oracle set only.**"""
    return pd.to_numeric(df["weather_actual_temp_max_c"], errors="coerce")


@feature(POST, depends_on=["classification_outcome"])
def finished_race(df: pd.DataFrame) -> pd.Series:
    """Whether the car was classified. Known only once the race has run."""
    return (
        df["classification_outcome"] == ClassificationOutcome.CLASSIFIED.value
    ).astype("float64")


@feature(POST, depends_on=["laps_completed", *RACE_KEYS])
def race_distance_share(df: pd.DataFrame) -> pd.Series:
    """Laps completed as a share of the winner's. Post-race by construction."""
    leader = df.groupby(list(RACE_KEYS), dropna=False, sort=False)[
        "laps_completed"
    ].transform("max")
    return (df["laps_completed"] / leader.replace(0, np.nan)).astype("float64")


__all__ = ["FINISH"]
