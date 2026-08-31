"""Hand-built race fixtures for target-construction tests.

Values are taken from the T-099 spike's live probe of the two races T-101c
names. They exercise all four FR-001 cases: classified finishers, lapped
finishers, retirements ordered by distance, and a disqualification.
"""

from __future__ import annotations

import pandas as pd

from f1pred.data.schema import RACE_ROW_SCHEMA, ClassificationOutcome, WeekendFormat


def _row(
    season: int,
    rnd: int,
    date: str,
    circuit: str,
    driver: str,
    grid: int,
    position: int,
    position_text: str,
    outcome: ClassificationOutcome,
    status: str,
    laps: int,
    final_lap_position: int | None = None,
    sprint_position: int | None = None,
    sprint_grid_position: int | None = None,
) -> dict[str, object]:
    return {
        "season": season,
        "round": rnd,
        "race_date": pd.Timestamp(date),
        "circuit_id": circuit,
        "weekend_format": WeekendFormat.CONVENTIONAL.value,
        "weekend_format_raw": "conventional",
        "driver_id": driver,
        "driver_code": driver[:3].upper(),
        "constructor_id": f"team_{driver}",
        "grid_position": grid,
        "classified_position": position,
        "position_text": position_text,
        "classification_outcome": outcome.value,
        "status_raw": status,
        "laps_completed": laps,
        "points": 0.0,
        "final_lap_position": final_lap_position,
        "sprint_position": sprint_position,
        "sprint_grid_position": sprint_grid_position,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col, dtype in RACE_ROW_SCHEMA.items():
        df[col] = df[col].astype(dtype)
    return df[list(RACE_ROW_SCHEMA)]


def australia_2014() -> pd.DataFrame:
    """22 entries: 10 finished, 4 lapped, 7 retired, 1 disqualified.

    Ricciardo was disqualified having completed the full distance, which is why
    a status-based rule would place him wrongly. Massa and Kobayashi both
    completed zero laps with no final-lap position to separate them, so this is
    the case that exercises FR-001b's grid fallback.
    """
    rows: list[dict[str, object]] = []
    add = rows.append
    common = (2014, 1, "2014-03-16", "albert_park")

    for pos in range(1, 11):
        add(_row(*common, f"finisher_{pos:02d}", pos, pos, str(pos),
                 ClassificationOutcome.CLASSIFIED, "Finished", 57))
    for offset, pos in enumerate(range(11, 15), start=1):
        add(_row(*common, f"lapped_{pos:02d}", pos, pos, str(pos),
                 ClassificationOutcome.CLASSIFIED, "+1 Lap", 56))

    retirements = [
        ("grosjean", 22, 15, "ERS", 43),
        ("maldonado", 21, 16, "ERS", 29),
        ("ericsson", 19, 17, "Oil pressure", 27),
        ("vettel", 12, 18, "Engine", 3),
        ("hamilton", 1, 19, "Engine", 2),
        ("massa", 9, 20, "Collision", 0),
        ("kobayashi", 14, 21, "Collision", 0),
    ]
    for driver, grid, pos, status, laps in retirements:
        add(_row(*common, driver, grid, pos, "R",
                 ClassificationOutcome.NOT_CLASSIFIED, status, laps))

    add(_row(*common, "ricciardo", 2, 22, "D",
             ClassificationOutcome.DISQUALIFIED, "Disqualified", 57))
    return _frame(rows)


def australia_2016() -> pd.DataFrame:
    """22 entries including one withdrawal (Kvyat, `W`, zero laps).

    Gutierrez and Alonso both retired on lap 16 after colliding, holding P12 and
    P13 respectively on that lap. Laps completed cannot separate them; FR-001b's
    final-lap position reproduces the official order, where a grid tie-break
    would reverse it.
    """
    rows: list[dict[str, object]] = []
    add = rows.append
    common = (2016, 1, "2016-03-20", "albert_park")

    for pos in range(1, 14):
        add(_row(*common, f"finisher_{pos:02d}", pos, pos, str(pos),
                 ClassificationOutcome.CLASSIFIED, "Finished", 57))
    for pos in range(14, 17):
        add(_row(*common, f"lapped_{pos:02d}", pos, pos, str(pos),
                 ClassificationOutcome.CLASSIFIED, "+1 Lap", 56))

    retirements = [
        ("ericsson", 15, 17, "Engine", 38),
        ("raikkonen", 4, 18, "Engine", 21),
        ("haryanto", 22, 19, "Mechanical", 17),
        ("gutierrez", 20, 20, "Collision", 16, 12),
        ("alonso", 11, 21, "Collision", 16, 13),
    ]
    for driver, grid, pos, status, laps, *detail in retirements:
        add(_row(*common, driver, grid, pos, "R",
                 ClassificationOutcome.NOT_CLASSIFIED, status, laps,
                 detail[0] if detail else None))

    # Never started: engine failure on the way to the grid.
    add(_row(*common, "kvyat", 18, 22, "W",
             ClassificationOutcome.WITHDRAWN, "Withdrew", 0))
    return _frame(rows)


def bahrain_2022_tail() -> pd.DataFrame:
    """The three cars that stopped late at Bahrain 2022.

    Perez and Verstappen retired with fuel-pressure failures and were still
    *classified* 18th and 19th; Gasly was not classified. A rule keyed on the
    status string rather than the position sentinel gets all three wrong.
    """
    common = (2022, 1, "2022-03-20", "bahrain")
    rows = [
        _row(*common, "leclerc", 1, 1, "1", ClassificationOutcome.CLASSIFIED,
             "Finished", 57),
        _row(*common, "perez", 4, 18, "18", ClassificationOutcome.CLASSIFIED,
             "Fuel pressure", 56),
        _row(*common, "verstappen", 2, 19, "19", ClassificationOutcome.CLASSIFIED,
             "Fuel pressure", 54),
        _row(*common, "gasly", 10, 20, "R", ClassificationOutcome.NOT_CLASSIFIED,
             "Power Unit", 44),
    ]
    return _frame(rows)


def synthetic_corpus(
    *,
    seasons: tuple[int, ...] = (2013, 2014, 2015),
    rounds: int = 6,
    teams: int = 5,
    circuits: tuple[str, ...] = ("albert_park", "monaco", "monza", "spa"),
    transfer: tuple[int, int] | None = (2014, 4),
    sprint_rounds: tuple[tuple[int, int], ...] = (),
    seed: int = 20260830,
) -> pd.DataFrame:
    """A small multi-season corpus for feature and leakage tests.

    Real races are the wrong tool here. The leakage checks need many races over
    several seasons with a stable roster, a mid-corpus driver transfer, and at
    least one retirement per race; a hand-built pair of real Grands Prix gives
    two races and no history. The values are meaningless — what matters is that
    the *structure* is faithful: two cars per team, drivers persisting across
    seasons, and one driver changing team mid-corpus so FR-216 has something to
    detect.

    `sprint_rounds` marks (season, round) pairs as sprint weekends by their
    declared format, leaving the sprint result itself null — which is the state
    the corpus is in before the FR-211 collection pass runs.

    Deterministic under `seed`, so a failing leakage test reproduces exactly.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    drivers = [f"driver_{i:02d}" for i in range(teams * 2)]
    team_of = {d: f"team_{i // 2:02d}" for i, d in enumerate(drivers)}
    # One mid-corpus move, so `detect_transfers` and POST_TRANSFER are exercised.
    mover, new_team = drivers[0], f"team_{teams - 1:02d}"

    rows: list[dict[str, object]] = []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            circuit = circuits[(rnd - 1) % len(circuits)]
            sprint = (season, rnd) in sprint_rounds
            date = f"{season}-{(rnd % 12) + 1:02d}-07"
            order = list(rng.permutation(drivers))
            # The last two cars retire, so retirement handling and the
            # reliability feature see real variation.
            retired = set(order[-2:])
            for slot, driver in enumerate(order, start=1):
                team = team_of[driver]
                if (
                    driver == mover
                    and transfer is not None
                    and (season, rnd) >= transfer
                ):
                    team = new_team
                is_retired = driver in retired
                add_points = max(0.0, 26.0 - 2.0 * slot) if not is_retired else 0.0
                rows.append(
                    _row(
                        season,
                        rnd,
                        date,
                        circuit,
                        driver,
                        grid=int(rng.integers(1, len(drivers) + 1)),
                        position=slot,
                        position_text="R" if is_retired else str(slot),
                        outcome=(
                            ClassificationOutcome.NOT_CLASSIFIED
                            if is_retired
                            else ClassificationOutcome.CLASSIFIED
                        ),
                        status="Engine" if is_retired else "Finished",
                        laps=int(rng.integers(5, 40)) if is_retired else 50,
                    )
                    | {
                        "constructor_id": team,
                        "points": add_points,
                        "weekend_format": (
                            WeekendFormat.SPRINT_STANDALONE.value
                            if sprint
                            else WeekendFormat.CONVENTIONAL.value
                        ),
                        "weekend_format_raw": (
                            "sprint_qualifying" if sprint else "conventional"
                        ),
                    }
                )
    return _frame(rows)
