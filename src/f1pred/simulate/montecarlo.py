"""Monte Carlo race simulation (FR-601..605, FR-751, FR-751a, plan D-7).

v0's `race_simulator.py` (tag `v0-vibe-coded`) had the right structure — pace
plus noise, a stochastic DNF draw, an iterative adjacent-swap overtaking pass —
and that structure is ported here largely intact, per D-7. What does not
survive the port is v0's parameters: `1.30 * pace_advantage + 1.15 * skill -
2.25 * difficulty - 0.35` and its dozen siblings were typed by hand and never
checked against anything. FR-751 requires every distribution's parameters to
be *estimated from prior races*, so this module separates fitting
(`fit_simulation_parameters`, reading a corpus of already-run races) from
simulating (`simulate_race`, reading only what is knowable before the race).

**The safety-car rate is a measured proxy, not an upstream field.** Real
per-lap track-status data exists in FastF1's race telemetry, but only from
2018 — the same floor T-214 already found for practice pace, and repeating it
here would tie the simulator's safety-car factor to 8 of the corpus's 16
seasons for no reason specific to safety cars. Instead this fits, from the
`classification_outcome` and `laps_completed` columns every season already
carries, the rate at which a circuit produces two or more retirements within
the first few laps — an accident-cluster proxy, not a confirmed deployment
count. It is named `_proxy` in code and documented as a proxy everywhere it is
reported, the same way T-101e's zero-lap retirements and T-213a's declined
supplement are labelled rather than smoothed over.

**Weather's two roles stay separated (FR-751a).** Fitting how much *more*
variable a wet race is than a dry one uses `weather_actual_precip_mm` — an
observed, post-race column — but only in aggregate, over many already-decided
races, to learn a structural fact about the sport ("wet races are messier").
That is a different act from handing the simulator today's actual rainfall,
which FR-751a forbids and which this module never does: `simulate_race` reads
only a pre-race wet *probability* for the specific race being simulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from f1pred.config import SEED
from f1pred.data.schema import ClassificationOutcome
from f1pred.reference import circuits
from f1pred.reference.circuits import OvertakingTier

POSITION_PROB_PREFIX: Final[str] = "prob_p"

#: F1's points-paying positions. A rule of the sport, not a fitted parameter.
POINTS_POSITIONS: Final[int] = 10

#: A retirement at or before this lap counts toward the safety-car proxy.
EARLY_RETIREMENT_LAP: Final[int] = 5

#: Above this, a race counts as "wet" for the purpose of fitting the weather
#: volatility multiplier. Matches the threshold used to report Sepang's prior.
WET_RACE_PRECIP_MM: Final[float] = 1.0

#: Two-sided normal z-scores for the coverage levels this project emits.
#: `pace_noise_std_from_interval` refuses any other coverage rather than
#: approximating one silently.
_Z_SCORE_BY_COVERAGE: Final[dict[float, float]] = {
    0.80: 1.2816,
    0.90: 1.6449,
    0.95: 1.9600,
}

REQUIRED_FIT_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "round",
    "circuit_id",
    "constructor_id",
    "classification_outcome",
    "laps_completed",
    "grid_position",
    "classified_position",
)

REQUIRED_RACE_COLUMNS: Final[tuple[str, ...]] = (
    "driver_id",
    "constructor_id",
    "circuit_id",
    "grid_position",
    "predicted_position",
)


class SimulationError(ValueError):
    """Raised when parameters cannot be fitted or a simulation cannot be run."""


# --------------------------------------------------------------------------
# Fitted parameters (FR-751, FR-751a)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimulationParameters:
    """Every stochastic factor's magnitude, measured from a corpus of races.

    `fit_simulation_parameters` builds this from races the caller has already
    decided are admissible history for the race about to be simulated — the
    same "prior races only" discipline `features.priors` enforces for model
    features (X.1, X.3). This object does not know or check which races that
    was; the caller's choice of `corpus` is the whole of that guarantee.
    """

    #: Std of (actual position - point prediction), in position units, from
    #: FR-402's own out-of-sample interval. Shared across drivers: the interval
    #: is one band for the whole model, not a per-driver quantity.
    pace_noise_std: float

    corpus_retirement_rate: float
    constructor_retirement_rate: dict[str, float]
    circuit_retirement_multiplier: dict[str, float]

    #: Share of races carrying the safety-car proxy incident, corpus-wide and
    #: per circuit (already shrunk toward the corpus rate).
    corpus_safety_car_rate: float
    circuit_safety_car_rate: dict[str, float]

    #: Share of classified finishers whose finishing position differs from
    #: their grid slot, per overtaking tier — a measured proxy for how much a
    #: circuit's tier actually lets the field reshuffle.
    tier_swap_rate: dict[OvertakingTier, float]

    #: How much wider the pace-noise draw gets in a wet race. 1.0 if no
    #: observed-weather column was available to fit it, which is reported via
    #: `weather_fitted` rather than silently defaulting to "no effect".
    wet_pace_multiplier: float
    weather_fitted: bool

    #: Share of races the fit corpus itself calls wet, for `simulate_race` to
    #: fall back on when a specific race carries no pre-race wet probability.
    corpus_wet_rate: float

    fit_races: int
    fit_seasons: tuple[int, ...]

    def retirement_probability(self, constructor_id: str, circuit_id: str) -> float:
        base = self.constructor_retirement_rate.get(
            constructor_id, self.corpus_retirement_rate
        )
        multiplier = self.circuit_retirement_multiplier.get(circuit_id, 1.0)
        return float(np.clip(base * multiplier, 0.01, 0.60))

    def safety_car_probability(self, circuit_id: str) -> float:
        return self.circuit_safety_car_rate.get(circuit_id, self.corpus_safety_car_rate)

    def swap_rate(self, circuit_id: str) -> float:
        tier = circuits.get(circuit_id).overtaking_tier
        return self.tier_swap_rate.get(tier, 0.30)


def _shrink(rate: pd.Series, n: pd.Series, prior: float, k: float) -> pd.Series:
    """The same shrunk-mean formula as `priors.shrink_towards`, on aggregates.

    `shrink_towards` is row-aligned to a corpus of prediction rows; here the
    inputs are already one row per group (per constructor, per circuit), so
    the formula is repeated rather than the row-aligned function reused.
    """
    return (n * rate + k * prior) / (n + k)


def fit_simulation_parameters(
    corpus: pd.DataFrame,
    *,
    pace_noise_std: float,
    shrinkage_strength: float = 5.0,
) -> SimulationParameters:
    """Measure every FR-602 factor's magnitude from `corpus`.

    `corpus` is the raw enriched frame (`dataset.enrich(dataset.load_corpus(...))`),
    not a built feature matrix — this reads `classification_outcome` and
    `laps_completed` directly, neither of which survives into the primary
    feature set. `pace_noise_std` is supplied rather than fitted here, because
    the honest out-of-sample estimate of it is FR-402's own calibrated
    interval (`pace_noise_std_from_interval`), and refitting a second one from
    raw residuals would give two numbers claiming to answer the same question.
    """
    missing = [c for c in REQUIRED_FIT_COLUMNS if c not in corpus.columns]
    if missing:
        raise SimulationError(
            f"cannot fit simulation parameters: corpus is missing {missing}"
        )
    if pace_noise_std <= 0:
        raise SimulationError(
            f"pace_noise_std must be positive, got {pace_noise_std}"
        )
    if shrinkage_strength <= 0:
        raise SimulationError(
            f"shrinkage_strength must be positive, got {shrinkage_strength}"
        )
    if corpus.empty:
        raise SimulationError("cannot fit simulation parameters from an empty corpus")

    work = corpus.copy()
    work["_retired"] = (
        work["classification_outcome"] == ClassificationOutcome.NOT_CLASSIFIED.value
    )
    corpus_retirement_rate = float(work["_retired"].mean())

    # -- retirement, by constructor and by circuit --------------------------
    by_constructor = work.groupby("constructor_id")["_retired"].agg(["mean", "count"])
    constructor_retirement_rate = _shrink(
        by_constructor["mean"], by_constructor["count"], corpus_retirement_rate,
        shrinkage_strength,
    ).to_dict()

    by_circuit = work.groupby("circuit_id")["_retired"].agg(["mean", "count"])
    circuit_rate = _shrink(
        by_circuit["mean"], by_circuit["count"], corpus_retirement_rate,
        shrinkage_strength,
    )
    circuit_retirement_multiplier = (
        (circuit_rate / corpus_retirement_rate) if corpus_retirement_rate > 0 else circuit_rate * 0
    ).to_dict()

    # -- safety-car proxy: >=2 retirements by EARLY_RETIREMENT_LAP -----------
    work["_early_retirement"] = work["_retired"] & (
        pd.to_numeric(work["laps_completed"], errors="coerce") <= EARLY_RETIREMENT_LAP
    )
    race_keys = ["season", "round"]
    incident_count = work.groupby(race_keys)["_early_retirement"].transform("sum")
    work["_incident_race"] = incident_count >= 2
    races = work.drop_duplicates(race_keys)[[*race_keys, "circuit_id", "_incident_race"]]
    corpus_safety_car_rate = float(races["_incident_race"].mean())
    by_circuit_sc = races.groupby("circuit_id")["_incident_race"].agg(["mean", "count"])
    circuit_safety_car_rate = _shrink(
        by_circuit_sc["mean"], by_circuit_sc["count"], corpus_safety_car_rate,
        shrinkage_strength,
    ).to_dict()

    # -- position-change dynamics by overtaking tier (FR-602) ----------------
    classified = work[
        work["classification_outcome"] == ClassificationOutcome.CLASSIFIED.value
    ].copy()
    classified["_moved"] = (
        classified["classified_position"] != classified["grid_position"]
    )
    # Grouped by the tier's *value* rather than the enum member: pandas'
    # groupby sorts its keys, and `OvertakingTier` declares no ordering.
    classified["_tier"] = classified["circuit_id"].map(
        lambda c: circuits.get(c).overtaking_tier.value
    )
    tier_swap_rate = {
        OvertakingTier(tier): float(rate)
        for tier, rate in classified.groupby("_tier")["_moved"].mean().items()
    }

    # -- weather volatility (FR-751a): fit only from observed columns, and
    #    say plainly when there was nothing to fit from. -------------------
    wet_pace_multiplier = 1.0
    weather_fitted = False
    corpus_wet_rate = 0.0
    if "weather_actual_precip_mm" in work.columns and work["weather_actual_precip_mm"].notna().any():
        precip = pd.to_numeric(work["weather_actual_precip_mm"], errors="coerce")
        race_wet = (
            work.assign(_precip=precip)
            .groupby(race_keys)["_precip"]
            .max()
            > WET_RACE_PRECIP_MM
        )
        corpus_wet_rate = float(race_wet.mean())
        wet_races = set(race_wet[race_wet].index)
        classified["_wet"] = list(
            zip(classified["season"], classified["round"])
        )
        classified["_wet"] = classified["_wet"].isin(wet_races)
        classified["_movement"] = (
            classified["classified_position"] - classified["grid_position"]
        ).abs()
        wet_std = classified.loc[classified["_wet"], "_movement"].std()
        dry_std = classified.loc[~classified["_wet"], "_movement"].std()
        if pd.notna(wet_std) and pd.notna(dry_std) and dry_std > 0:
            wet_pace_multiplier = float(np.clip(wet_std / dry_std, 1.0, 4.0))
            weather_fitted = True

    return SimulationParameters(
        pace_noise_std=float(pace_noise_std),
        corpus_retirement_rate=corpus_retirement_rate,
        constructor_retirement_rate=constructor_retirement_rate,
        circuit_retirement_multiplier=circuit_retirement_multiplier,
        corpus_safety_car_rate=corpus_safety_car_rate,
        circuit_safety_car_rate=circuit_safety_car_rate,
        tier_swap_rate=tier_swap_rate,
        wet_pace_multiplier=wet_pace_multiplier,
        weather_fitted=weather_fitted,
        corpus_wet_rate=corpus_wet_rate,
        fit_races=int(work.drop_duplicates(race_keys).shape[0]),
        fit_seasons=tuple(sorted(int(s) for s in work["season"].unique())),
    )


def pace_noise_std_from_interval(*, half_width: float, coverage: float) -> float:
    """Convert FR-402's symmetric interval half-width into a normal std.

    `half_width` is `(upper_offset - lower_offset) / 2` from the model's own
    `Interval` — already an out-of-sample estimate, held back from training
    per FR-402. Refused for a coverage this project does not emit, rather than
    approximating a z-score for it silently.
    """
    try:
        z = _Z_SCORE_BY_COVERAGE[round(coverage, 2)]
    except KeyError:
        raise SimulationError(
            f"no declared z-score for coverage {coverage}; add it to "
            f"_Z_SCORE_BY_COVERAGE deliberately rather than approximating"
        ) from None
    if half_width <= 0:
        raise SimulationError(f"interval half-width must be positive, got {half_width}")
    return float(half_width / z)


# --------------------------------------------------------------------------
# Simulation (FR-601, FR-602, FR-604)
# --------------------------------------------------------------------------


def _overtake_probability(pace_advantage: float, pace_scale: float, swap_rate: float) -> float:
    """Probability the trailing car passes an adjacent slower one this pass.

    At zero pace advantage this returns `swap_rate` itself — the tier's
    measured base rate of finishing out of grid order — rather than an
    arbitrary constant; a real pace advantage moves it up from there.
    """
    scale = pace_scale if pace_scale > 0 else 1.0
    sigmoid = 1.0 / (1.0 + np.exp(-pace_advantage / scale))
    return float(np.clip(swap_rate * sigmoid * 2.0, 0.01, 0.95))


def _simulate_overtaking_order(
    *,
    grid: np.ndarray,
    pace: np.ndarray,
    dnf_mask: np.ndarray,
    swap_rate: float,
    pace_scale: float,
    n_passes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One race's finishing order from grid order, pace, and DNFs.

    Ported from v0's adjacent-swap mechanism (D-7): repeatedly walk the order
    back-to-front and let a faster trailing car pass with `_overtake_probability`.
    What changed from v0 is only where that probability's magnitude comes
    from.
    """
    order = np.argsort(grid, kind="stable").astype(int)
    for _ in range(n_passes):
        for position in range(len(order) - 2, -1, -1):
            ahead, behind = order[position], order[position + 1]
            if dnf_mask[ahead] or dnf_mask[behind]:
                continue
            advantage = pace[ahead] - pace[behind]  # positive: behind car faster
            if advantage <= 0:
                continue
            probability = _overtake_probability(advantage, pace_scale, swap_rate)
            if rng.random() < probability:
                order[position], order[position + 1] = order[position + 1], order[position]

    if dnf_mask.any():
        finishers = [i for i in order if not dnf_mask[i]]
        dnfs = [i for i in order if dnf_mask[i]]
        dnfs.sort(key=lambda i: pace[i], reverse=True)
        order = np.array(finishers + dnfs, dtype=int)
    return order


def simulate_race(
    race: pd.DataFrame,
    params: SimulationParameters,
    *,
    n_simulations: int = 1000,
    seed: int = SEED,
    points_positions: int = POINTS_POSITIONS,
) -> pd.DataFrame:
    """Run `n_simulations` draws of one race and aggregate the outcomes.

    `race` is one row per driver, carrying `driver_id`, `constructor_id`,
    `circuit_id` (a single value for the whole frame), `grid_position`, a
    `predicted_position` (the model's point prediction — lower is better),
    and optionally `weather_prior_wet` (the pre-race climatological
    probability; FR-751a bars anything but a prior here). Deterministic given
    `seed` (FR-604); every driver's per-position probabilities sum to 1.0
    within tolerance by construction, and `assert_probabilities_sum_to_one`
    checks it rather than trusting it (FR-605).
    """
    missing = [c for c in REQUIRED_RACE_COLUMNS if c not in race.columns]
    if missing:
        raise SimulationError(f"cannot simulate: race frame is missing {missing}")
    if n_simulations < 1:
        raise SimulationError(f"n_simulations must be at least 1, got {n_simulations}")

    frame = race.sort_values("driver_id", kind="stable").reset_index(drop=True)
    n_drivers = len(frame)
    if n_drivers < 2:
        raise SimulationError("cannot simulate a race with fewer than two drivers")

    circuit_id = frame["circuit_id"].iloc[0]
    if not (frame["circuit_id"] == circuit_id).all():
        raise SimulationError("a simulated race must carry a single circuit_id")

    rng = np.random.default_rng(seed)

    base_pace = frame["predicted_position"].to_numpy(dtype=float)
    grid = frame["grid_position"].to_numpy(dtype=float)

    if "weather_prior_wet" in frame.columns and pd.notna(frame["weather_prior_wet"].iloc[0]):
        wet_probability = float(np.clip(frame["weather_prior_wet"].iloc[0], 0.0, 1.0))
    else:
        wet_probability = params.corpus_wet_rate

    pace_noise = rng.normal(0.0, params.pace_noise_std, size=(n_simulations, n_drivers))
    wet_draw = rng.random(n_simulations) < wet_probability
    if wet_draw.any():
        pace_noise[wet_draw] *= params.wet_pace_multiplier

    safety_car_rate = params.safety_car_probability(circuit_id)
    safety_car_draw = rng.random(n_simulations) < safety_car_rate
    if safety_car_draw.any():
        chaos_std = params.pace_noise_std * (1.0 + safety_car_rate)
        pace_noise[safety_car_draw] += rng.normal(
            0.0, chaos_std, size=(int(safety_car_draw.sum()), n_drivers)
        )

    simulated_pace = base_pace.reshape(1, -1) + pace_noise

    retirement_prob = np.array(
        [params.retirement_probability(c, circuit_id) for c in frame["constructor_id"]]
    )
    dnf_matrix = rng.random((n_simulations, n_drivers)) < retirement_prob.reshape(1, -1)

    swap_rate = params.swap_rate(circuit_id)
    n_passes = max(3, round(n_drivers * (0.5 + swap_rate)))

    position_matrix = np.empty((n_simulations, n_drivers), dtype=np.int16)
    for sim in range(n_simulations):
        order = _simulate_overtaking_order(
            grid=grid,
            pace=simulated_pace[sim],
            dnf_mask=dnf_matrix[sim],
            swap_rate=swap_rate,
            pace_scale=params.pace_noise_std,
            n_passes=n_passes,
            rng=rng,
        )
        position_matrix[sim, order] = np.arange(1, n_drivers + 1)

    return aggregate_positions(
        frame, position_matrix, dnf_matrix, points_positions=points_positions
    )


def aggregate_positions(
    frame: pd.DataFrame,
    position_matrix: np.ndarray,
    dnf_matrix: np.ndarray,
    *,
    points_positions: int = POINTS_POSITIONS,
) -> pd.DataFrame:
    """Turn a `(n_simulations, n_drivers)` position matrix into FR-603's table."""
    n_drivers = len(frame)
    output = frame[["driver_id", "constructor_id", "grid_position"]].copy()
    output["expected_finish"] = position_matrix.mean(axis=0)
    output["win_probability"] = (position_matrix == 1).mean(axis=0)
    output["podium_probability"] = (position_matrix <= 3).mean(axis=0)
    output["points_probability"] = (position_matrix <= points_positions).mean(axis=0)
    output["dnf_probability"] = dnf_matrix.mean(axis=0)
    for position in range(1, n_drivers + 1):
        output[f"{POSITION_PROB_PREFIX}{position:02d}"] = (
            position_matrix == position
        ).mean(axis=0)

    output = output.sort_values(
        ["expected_finish", "grid_position"], kind="stable"
    ).reset_index(drop=True)
    assert_probabilities_sum_to_one(output, n_drivers=n_drivers)
    return output


def assert_probabilities_sum_to_one(
    result: pd.DataFrame, *, n_drivers: int | None = None, tolerance: float = 1e-6
) -> None:
    """FR-605: every driver's per-position probabilities sum to 1.0 ± `tolerance`.

    Checked rather than trusted, the same discipline T-207 and T-221 apply to
    the leakage checks: a property that follows from the construction is still
    worth asserting, because the construction is what changes when someone
    edits this file later.
    """
    n = n_drivers if n_drivers is not None else sum(
        1 for c in result.columns if c.startswith(POSITION_PROB_PREFIX)
    )
    columns = [f"{POSITION_PROB_PREFIX}{p:02d}" for p in range(1, n + 1)]
    missing = [c for c in columns if c not in result.columns]
    if missing:
        raise SimulationError(f"cannot check FR-605: missing columns {missing}")
    totals = result[columns].sum(axis=1)
    bad = (totals - 1.0).abs() > tolerance
    if bad.any():
        worst = float((totals - 1.0).abs().max())
        raise SimulationError(
            f"FR-605 violated: {int(bad.sum())} driver row(s) have per-position "
            f"probabilities summing outside 1.0 +/- {tolerance} (worst "
            f"deviation {worst:.2e})"
        )


__all__ = [
    "EARLY_RETIREMENT_LAP",
    "POINTS_POSITIONS",
    "POSITION_PROB_PREFIX",
    "REQUIRED_FIT_COLUMNS",
    "REQUIRED_RACE_COLUMNS",
    "WET_RACE_PRECIP_MM",
    "SimulationError",
    "SimulationParameters",
    "aggregate_positions",
    "assert_probabilities_sum_to_one",
    "fit_simulation_parameters",
    "pace_noise_std_from_interval",
    "simulate_race",
]
