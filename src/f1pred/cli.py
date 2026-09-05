"""The command surface (FR-801..804).

Six commands, in the order the pipeline runs them:

    f1pred collect     the corpus, from the network. The only networked stage.
    f1pred features    build the feature matrix and report on it
    f1pred train       fit the ranker and save it with its provenance record
    f1pred evaluate    score the model against all three baselines
    f1pred simulate    Monte Carlo the holdout and measure calibration
    f1pred predict     rank a starting grid supplied as CSV

Every stage that reads an earlier stage's output routes its absence through
`artifacts.require`, so a missing file always comes back naming the command
that makes it (FR-802). Every number leaves through `render`, so precision
(FR-804) and the `UNCALIBRATED` label (FR-705) are properties of the exit and
not of the caller's diligence.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import click
import pandas as pd

from f1pred import artifacts, render
from f1pred.artifacts import MissingArtifactError
from f1pred.config import RAW_DIR, RESULTS_DIR, SEED
from f1pred.data.grid_csv import GridCsvError, SCHEMA_DOCUMENTATION, read_grid_csv

__all__ = ["cli", "main"]

_LOG_FORMAT = "%(levelname)s %(message)s"


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format=_LOG_FORMAT)


class _Timer:
    """Wall-clock around a stage, for NFR-01's recorded measurement.

    NFR-01 asks for the number to be *in the results file*, not printed and
    forgotten: a budget nobody can check afterwards is a budget nobody is
    keeping.
    """

    def __init__(self) -> None:
        self.seconds = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


def _require_corpus() -> None:
    artifacts.require("corpus", RAW_DIR, exists=artifacts.corpus_is_present(RAW_DIR))


def _echo_table(headings: list[str], rows: list[list[str]]) -> None:
    """A plain aligned table. Deliberately not `rich`: the output of these
    commands gets pasted into issues and results notes, and box-drawing
    characters survive that badly."""
    widths = [len(h) for h in headings]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    click.echo("  ".join(h.ljust(w) for h, w in zip(headings, widths)).rstrip())
    click.echo("  ".join("-" * w for w in widths))
    for row in rows:
        click.echo("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())


def _calibration_statuses() -> tuple[dict[str, str], Path | None]:
    """What the last committed calibration run measured, by event.

    Unmeasured is `UNCALIBRATED`, not absent (FR-705). A run that has never
    been calibrated and one that failed calibration are different facts, but
    they carry the same obligation at the render layer, and conflating them
    there is safer than a third state nobody reads.
    """
    path = artifacts.latest_result("calibration_*.json")
    if path is None:
        return (
            {
                "win probability": render.UNCALIBRATED,
                "podium probability": render.UNCALIBRATED,
                "points probability": render.UNCALIBRATED,
                "position intervals": render.UNCALIBRATED,
            },
            None,
        )
    record = json.loads(path.read_text(encoding="utf-8"))["calibration"]
    statuses = {
        f"{name} probability": event["status"] for name, event in record["events"].items()
    }
    statuses["position intervals"] = record["interval_coverage"]["status"]
    return statuses, path


# --------------------------------------------------------------------------
# The group
# --------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Log what each stage is doing.")
@click.version_option(package_name="f1pred")
def cli(verbose: bool) -> None:
    """Pre-race Formula 1 finishing-order prediction, scored against the grid.

    The headline result is in the README, and every figure there resolves to a
    file under results/.
    """
    _configure_logging(verbose)


# --------------------------------------------------------------------------
# 1. collect
# --------------------------------------------------------------------------


@cli.command()
@click.option("--seasons", default=None, help="Season range, e.g. 2010-2025. Default: the corpus window.")
@click.option("--weather/--no-weather", default=False, help="Also collect the climatological weather archive.")
@click.option("--telemetry/--no-telemetry", default=False, help="Also collect practice telemetry (2018+).")
def collect(seasons: str | None, weather: bool, telemetry: bool) -> None:
    """Collect the race corpus from the timing source.

    The only stage that touches the network (NFR-05). Checkpoints per season
    and resumes: an interrupted run is restarted by running this again.
    """
    from f1pred.corpus import default_window
    from f1pred.data.collect import (
        ErgastClient,
        check_season_counts,
        collect_corpus,
        read_checkpoint,
        write_failure_manifest,
    )
    from f1pred.data.schema import empty_frame

    window = default_window()
    wanted = _parse_seasons(seasons) if seasons else window.resolve()
    click.echo(f"corpus: {wanted[0]}-{wanted[-1]} ({len(wanted)} seasons)")

    with _Timer() as timer:
        client = ErgastClient()
        reports, manifest = collect_corpus(wanted, root=RAW_DIR, client=client, log=click.echo)
        frames = [f for f in (read_checkpoint(RAW_DIR, s) for s in wanted) if f is not None]
        corpus = pd.concat(frames, ignore_index=True) if frames else empty_frame()

    for count in check_season_counts(corpus):
        if count.warning():
            click.echo(f"FR-106 warning: {count.warning()}")
    click.echo(manifest.summary())
    click.echo(f"manifest: {write_failure_manifest(manifest, RAW_DIR)}")
    click.echo(f"{len(corpus)} rows across {len(reports)} seasons in {timer.seconds:.0f}s")

    if weather:
        _collect_weather(corpus)
    if telemetry:
        _collect_telemetry(wanted)


def _parse_seasons(text: str) -> tuple[int, ...]:
    if "-" in text:
        first, last = (int(part) for part in text.split("-", 1))
        if last < first:
            raise click.BadParameter(f"{text!r} runs backwards")
        return tuple(range(first, last + 1))
    return (int(text),)


def _collect_weather(corpus: pd.DataFrame) -> None:
    from f1pred.data.weather import OpenMeteoClient, collect_archive, write_archive

    circuit_ids = sorted(corpus["circuit_id"].dropna().unique().tolist())
    click.echo(f"weather: {len(circuit_ids)} circuits")
    archive, failed = collect_archive(circuit_ids, client=OpenMeteoClient(), log=click.echo)
    click.echo(f"weather archive: {write_archive(archive, RAW_DIR)}")
    if failed:
        click.echo(f"weather: {len(failed)} circuit(s) failed: {', '.join(failed)}")


def _collect_telemetry(seasons: tuple[int, ...]) -> None:
    from f1pred.data.telemetry import (
        FastF1PracticeClient,
        TELEMETRY_START_SEASON,
        collect_practice_corpus,
    )

    covered = tuple(s for s in seasons if s >= TELEMETRY_START_SEASON)
    if not covered:
        click.echo(
            f"telemetry: no requested season reaches {TELEMETRY_START_SEASON}; "
            f"nothing to collect. FR-214 makes those seasons null, not missing."
        )
        return
    click.echo(f"telemetry: {covered[0]}-{covered[-1]}")
    _, reports = collect_practice_corpus(
        covered, client=FastF1PracticeClient(), root=RAW_DIR, log=click.echo
    )
    click.echo(f"telemetry: {len(reports)} season(s) on disk")


# --------------------------------------------------------------------------
# 2. features
# --------------------------------------------------------------------------


@cli.command()
@click.option("--seasons", default=None, help="Season range. Default: every collected season.")
@click.option("--check-leakage/--no-check-leakage", default=True, help="Run the two-layer leakage firewall.")
def features(seasons: str | None, check_leakage: bool) -> None:
    """Build the feature matrix and report what went into it.

    Reads the corpus; no network. The leakage firewall runs by default because
    a control you have to remember to switch on is the control v0 did not have.
    """
    _require_corpus()
    from f1pred.dataset import enrich, experiment, load_corpus

    wanted = _parse_seasons(seasons) if seasons else None
    with _Timer() as timer:
        # The raw enriched corpus, kept: the leakage checks read raw schema
        # columns that the assembled frame no longer carries, and rebuilding a
        # second corpus for them would let the two drift apart.
        raw = enrich(load_corpus(wanted))
        built = experiment(raw)

    click.echo(f"{len(built)} rows, {len(built.feature_names)} features, {timer.seconds:.1f}s")
    if built.absent_sources:
        click.echo(
            f"absent sources ({len(built.absent_sources)}): "
            f"{', '.join(built.absent_sources)} — every feature reading them is "
            f"null throughout, which is recorded rather than filled."
        )
    flags = built.matrix.flags
    click.echo(f"{int(flags['low_information'].sum())} row(s) flagged LOW_INFORMATION (FR-212)")
    click.echo(f"{int(flags['post_transfer'].sum())} row(s) flagged POST_TRANSFER (FR-216)")

    if check_leakage:
        _run_leakage_checks(raw)


def _run_leakage_checks(frame: pd.DataFrame) -> None:
    """FR-204 and FR-205, both layers, exit code carrying the verdict."""
    from f1pred.features import checks

    click.echo("\nleakage firewall:")
    failed = 0
    for report in checks.run_all(frame):
        status = "PASS" if report.passed else "FAIL"
        detail = (
            f"{report.features_checked} feature(s), {report.races_checked} race(s)"
            if report.passed
            else f"{len(report.violations)} violation(s)"
        )
        click.echo(f"  {status}  {report.check}: {detail}")
        for violation in report.violations:
            click.echo(f"          {violation}")
        failed += 0 if report.passed else 1
    if failed:
        raise click.ClickException(
            f"{failed} leakage check(s) failed. A feature reading post-race "
            f"information is the defect this project exists to prevent; fix the "
            f"feature rather than the check."
        )


# --------------------------------------------------------------------------
# 3. train
# --------------------------------------------------------------------------


@cli.command()
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Model stem. Default: data/processed/ranker")
@click.option("--holdout-seasons", default=3, show_default=True, help="Seasons held out chronologically.")
def train(out: Path | None, holdout_seasons: int) -> None:
    """Fit the ranker on the training seasons and save it.

    The split is applied inside the trainer, not here: a trainer handed a
    pre-cut training set cannot check that the cut respected the boundary
    (FR-403).
    """
    _require_corpus()
    from f1pred.dataset import prepare
    from f1pred.evaluate.runner import train_model
    from f1pred.evaluate.split import default_split

    stem = Path(out) if out is not None else artifacts.MODEL_STEM
    split = default_split(holdout_seasons=holdout_seasons)
    with _Timer() as timer:
        built = prepare()
        model = train_model(built, split=split)
        model_path = model.save(stem)

    click.echo(
        f"trained on {model.training_rows} rows from "
        f"{split.training_seasons[0]}-{split.training_seasons[-1]}, "
        f"{model.trees} trees kept, {timer.seconds:.1f}s"
    )
    click.echo(f"model:  {model_path}")
    click.echo(f"record: {stem.with_suffix('.json')}")
    click.echo(f"commit: {model.code_commit}")


# --------------------------------------------------------------------------
# 4. evaluate
# --------------------------------------------------------------------------


@cli.command()
@click.option("--holdout-seasons", default=3, show_default=True, help="Seasons held out chronologically.")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Results file. Default: results/model_vs_baselines_<holdout>.json")
def evaluate(holdout_seasons: int, out: Path | None) -> None:
    """Score the model against all three baselines and write the results file.

    The baselines are not optional and not selected here: they are constructed
    by the runner so no invocation can quietly drop one (FR-304, FR-305).
    """
    _require_corpus()
    from f1pred.evaluate.harness import render_verdict
    from f1pred.evaluate.runner import run
    from f1pred.evaluate.split import default_split

    split = default_split(holdout_seasons=holdout_seasons)
    with _Timer() as timer:
        record, model = run(split=split, path=out)

    # NFR-01: the measurement goes into the file, not just onto the terminal.
    _record_runtime(record, timer.seconds, out or _default_evaluation_path(record))

    click.echo(render_verdict(record))
    click.echo("")
    _echo_scores(record)
    click.echo(f"\ntrained on {model.training_rows} rows, {model.code_commit}")
    click.echo(f"evaluation wall-clock: {timer.seconds:.1f}s (NFR-01 budget: 300s)")


def _default_evaluation_path(record: dict[str, Any]) -> Path:
    holdout = record["split"]["holdout_seasons"]
    return RESULTS_DIR / f"model_vs_baselines_{holdout[0]}_{holdout[-1]}.json"


def _record_runtime(record: dict[str, Any], seconds: float, path: Path) -> None:
    """Write the measured wall-clock into the results file it describes (NFR-01)."""
    from f1pred.evaluate.harness import write_results

    record["runtime"] = {
        "wall_clock_seconds": round(seconds, 2),
        "budget_seconds": 300,
        "within_budget": seconds < 300,
        "reference_machine": "Apple Silicon, macOS 25.6, Python 3.13 (plan.md §4)",
        "measured": date.today().isoformat(),
    }
    write_results(record, Path(path))


def _echo_scores(record: dict[str, Any]) -> None:
    """FR-804 rule 2 in practice: every score at the precision its CI supports."""
    metric = record["primary_metric"]
    rows: list[list[str]] = []
    for name, result in record["predictors"].items():
        interval = result["metrics"][metric]
        rows.append(
            [
                name,
                render.format_measurement(
                    interval["point"],
                    low=interval["ci_low"],
                    high=interval["ci_high"],
                    with_interval=True,
                ),
            ]
        )
    _echo_table([f"predictor", f"{metric} (95% CI)"], rows)


# --------------------------------------------------------------------------
# 5. simulate
# --------------------------------------------------------------------------


@cli.command()
@click.option("--draws", default=1000, show_default=True, help="Monte Carlo draws per race.")
@click.option("--seed", default=SEED, show_default=True, help="FR-604: the run is deterministic given this.")
@click.option("--holdout-seasons", default=3, show_default=True)
def simulate(draws: int, seed: int, holdout_seasons: int) -> None:
    """Simulate the holdout races and measure calibration.

    Writes the reliability curves, Brier scores and interval coverage that
    decide whether this project's probabilities may carry a claim (FR-701..706).
    """
    _require_corpus()
    from f1pred.evaluate.calibration import run_calibration
    from f1pred.evaluate.split import default_split

    split = default_split(holdout_seasons=holdout_seasons)
    with _Timer() as timer:
        record = run_calibration(split=split, n_simulations=draws, seed=seed)

    calibration = record["calibration"]
    rows = [
        [
            name,
            render.format_probability(event["max_reliability_gap"]),
            render.format_probability(event["brier_score"]["model"]["point"]),
            render.format_probability(event["brier_score"]["grid_baseline"]["point"]),
            event["status"],
        ]
        for name, event in calibration["events"].items()
    ]
    _echo_table(
        ["event", "worst gap", "Brier", "Brier (grid)", "status"], rows
    )

    coverage = calibration["interval_coverage"]
    click.echo(
        f"\nintervals: stated {render.format_probability(coverage['stated'])}, "
        f"observed {render.format_probability(coverage['observed']['point'])} "
        f"({render.format_probability(coverage['observed']['ci_low'])} to "
        f"{render.format_probability(coverage['observed']['ci_high'])}) "
        f"-> {coverage['status']}"
    )
    statuses = {f"{k} probability": v["status"] for k, v in calibration["events"].items()}
    statuses["position intervals"] = coverage["status"]
    click.echo("\n" + render.render_calibration_note(statuses))
    click.echo(f"\n{draws} draws per race, seed {seed}, {timer.seconds:.0f}s")


# --------------------------------------------------------------------------
# 6. predict
# --------------------------------------------------------------------------


@cli.command()
@click.argument("grid_csv", type=click.Path(path_type=Path), required=False)
@click.option("--model", "model_path", type=click.Path(path_type=Path), default=None, help="Model stem. Default: data/processed/ranker")
@click.option("--draws", default=1000, show_default=True, help="Monte Carlo draws for the probabilities.")
@click.option("--seed", default=SEED, show_default=True)
@click.option("--schema", "show_schema", is_flag=True, help="Print the documented CSV schema and exit.")
def predict(
    grid_csv: Path | None,
    model_path: Path | None,
    draws: int,
    seed: int,
    show_schema: bool,
) -> None:
    """Rank a starting grid supplied as CSV (FR-803).

    The file describes one race that has not run. Every violation of the schema
    is reported at once. Columns describing the race being predicted — observed
    weather, or its outcome — are refused rather than ignored.
    """
    if show_schema:
        click.echo(SCHEMA_DOCUMENTATION)
        return
    if grid_csv is None:
        raise click.UsageError(
            "give a grid CSV to predict from, or --schema to see what one looks like"
        )

    _require_corpus()
    stem = Path(model_path) if model_path is not None else artifacts.MODEL_STEM
    artifacts.require("model", stem, exists=artifacts.model_is_present(stem))

    from f1pred.models.ranker import TrainedRanker
    from f1pred.predict import predict_race

    grid = read_grid_csv(grid_csv)
    model = TrainedRanker.load(stem)
    result, notes = predict_race(grid, model, draws=draws, seed=seed)

    statuses, calibration_path = _calibration_statuses()
    columns = [
        render.ProbabilityColumn("win", "win_probability", statuses["win probability"]),
        render.ProbabilityColumn("podium", "podium_probability", statuses["podium probability"]),
        render.ProbabilityColumn("points", "points_probability", statuses["points probability"]),
    ]
    interval_status = statuses["position intervals"]
    rows = result.to_dict("records")
    headings, cells = render.probability_table(
        rows,
        columns,
        label_columns=[
            ("pos", "predicted_rank"),
            ("driver", "driver_id"),
            ("grid", "grid_position"),
            ("interval", "interval"),
            ("flags", "flags"),
        ],
    )
    click.echo(
        f"{grid['season'].iloc[0]} round {grid['round'].iloc[0]} — "
        f"{grid['circuit_id'].iloc[0]}, {len(grid)} entries\n"
    )
    _echo_table(headings, cells)
    click.echo(
        f"\nintervals are {result['interval_coverage'].iloc[0]:.0%} bands "
        f"[{interval_status}]"
    )
    click.echo("\n" + render.render_calibration_note(statuses))
    if calibration_path is not None:
        click.echo(f"calibration measured in {calibration_path}")
    for note in notes:
        click.echo(note)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the CLI, turning a missing artifact into an actionable exit (FR-802).

    Click would render `MissingArtifactError` as a traceback. The message is
    the useful part — it names the command that produces what is missing — so
    it is printed alone, and the traceback is suppressed.
    """
    try:
        cli.main(args=argv, standalone_mode=False)
    except (MissingArtifactError, GridCsvError) as error:
        click.echo(str(error), err=True)
        return 2
    except click.ClickException as error:
        error.show()
        return error.exit_code
    except click.exceptions.Abort:
        click.echo("aborted", err=True)
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
