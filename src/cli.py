"""CLI interface for the F1 Race Result Predictor."""

import logging
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import PLOTS_DIR

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """🏁 F1 Race Result Predictor - Predict race outcomes using ML."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


@cli.command()
@click.option("--start-year", default=2018, help="Start year for data collection")
@click.option("--end-year", default=2024, help="End year for data collection")
def collect(start_year, end_year):
    """📊 Collect historical F1 data from FastF1."""
    from src.config import HISTORICAL_YEARS_START, HISTORICAL_YEARS_END
    import src.config as config

    config.HISTORICAL_YEARS_START = start_year
    config.HISTORICAL_YEARS_END = end_year

    console.print("\n[bold red]🏎️  F1 Data Collector[/bold red]")
    console.print(f"Collecting data from {start_year} to {end_year}...\n")

    from src.data_collector import collect_all_seasons
    df = collect_all_seasons()

    table = Table(title="Data Collection Summary")
    table.add_column("Year", style="cyan")
    table.add_column("Races", style="green")
    table.add_column("Records", style="yellow")

    for year in range(start_year, end_year + 1):
        year_data = df[df["year"] == year]
        races = year_data["round"].nunique()
        records = len(year_data)
        table.add_row(str(year), str(races), str(records))

    console.print(table)
    console.print(f"\n[green]✅ Total records: {len(df)}[/green]")


@cli.command()
def engineer():
    """🔧 Engineer features from raw data."""
    console.print("\n[bold red]🔧 Feature Engineering[/bold red]")

    from src.data_collector import load_cached_data
    from src.feature_engineer import engineer_features, prepare_training_data

    raw_df = load_cached_data()
    processed_df = engineer_features(raw_df)
    X, y, feature_list = prepare_training_data(processed_df)

    table = Table(title="Feature Engineering Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total samples", str(len(X)))
    table.add_row("Number of features", str(len(feature_list)))
    table.add_row("Target range", f"{y.min()} - {y.max()}")
    table.add_row("Target mean", f"{y.mean():.2f}")

    console.print(table)

    console.print("\n[bold]Features:[/bold]")
    for f in feature_list:
        console.print(f"  • {f}")


@cli.command()
@click.option("--save/--no-save", default=True, help="Save trained model")
def train(save):
    """🧠 Train the prediction model."""
    console.print("\n[bold red]🧠 Training F1 Race Result Predictor[/bold red]\n")

    from src.data_collector import load_cached_data
    from src.feature_engineer import engineer_features, prepare_training_data
    from src.model import RacePredictor, plot_feature_importance, plot_predictions_vs_actual, plot_position_distribution

    # Load and prepare data
    with console.status("Loading data..."):
        raw_df = load_cached_data()

    with console.status("Engineering features..."):
        processed_df = engineer_features(raw_df)
        X, y, feature_list = prepare_training_data(processed_df)

    # Train model
    with console.status("Training model..."):
        predictor = RacePredictor()
        metrics = predictor.train(
            X,
            y,
            race_keys=processed_df[["year", "round", "benchmark_group", "benchmark_group_code"]],
        )

    # Display metrics
    metrics_table = Table(title="Model Evaluation Metrics", show_header=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")

    metric_labels = {
        "mae": "Mean Absolute Error",
        "rmse": "Root Mean Squared Error",
        "expected_finish_mae": "Expected Finish MAE",
        "exact_accuracy": "Exact Position Accuracy",
        "within_1_accuracy": "Within 1 Position",
        "within_2_accuracy": "Within 2 Positions",
        "within_3_accuracy": "Within 3 Positions",
        "podium_accuracy": "Podium Prediction Accuracy",
        "top_3_accuracy": "Top 3 Accuracy",
        "top_5_accuracy": "Top 5 Accuracy",
        "spearman_mean": "Spearman Rank Correlation",
        "cv_mae_mean": "Cross-Val MAE (mean)",
        "cv_mae_std": "Cross-Val MAE (std)",
    }

    for key, label in metric_labels.items():
        if key in metrics:
            value = metrics[key]
            if "accuracy" in key or key == "spearman_mean":
                metrics_table.add_row(label, f"{value:.1%}")
            else:
                metrics_table.add_row(label, f"{value:.3f}")

    console.print(metrics_table)

    benchmark_metrics = metrics.get("benchmark_metrics", {})
    if benchmark_metrics:
        benchmark_table = Table(title="Benchmark Suite by Track Type")
        benchmark_table.add_column("Group", style="cyan")
        benchmark_table.add_column("MAE", style="green", justify="right")
        benchmark_table.add_column("Top 3", style="yellow", justify="right")
        benchmark_table.add_column("Top 5", style="yellow", justify="right")
        benchmark_table.add_column("Spearman", style="magenta", justify="right")

        for group_name, values in benchmark_metrics.items():
            benchmark_table.add_row(
                group_name,
                f"{values['mae']:.3f}",
                f"{values['top_3_accuracy']:.1%}",
                f"{values['top_5_accuracy']:.1%}",
                f"{values['spearman']:.1%}",
            )

        console.print(benchmark_table)

    # Feature importance
    importance = predictor.get_feature_importance().head(10)
    imp_table = Table(title="Top 10 Feature Importance")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Importance", style="yellow", justify="right")

    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 100)
        imp_table.add_row(row["feature"], f"{row['importance']:.4f} {bar}")

    console.print(imp_table)

    # Save and plot
    if save:
        predictor.save()
        console.print("\n[green]✅ Model saved[/green]")

        try:
            plot_feature_importance(predictor)
            eval_data = predictor.evaluation_data or {}
            if eval_data:
                plot_predictions_vs_actual(
                    predictor,
                    eval_data["X_test"],
                    eval_data["y_test"],
                    race_keys=eval_data.get("test_keys"),
                )
                plot_position_distribution(
                    predictor,
                    eval_data["X_test"],
                    eval_data["y_test"],
                    race_keys=eval_data.get("test_keys"),
                )
            console.print(f"[green]✅ Plots saved to {PLOTS_DIR}/[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not generate plots: {e}[/yellow]")


@cli.command()
@click.option("--year", type=int, help="Race year")
@click.option("--round", type=int, help="Round number")
def predict(year, round):
    """🏁 Predict race results."""
    console.print("\n[bold red]🏁 Race Prediction[/bold red]\n")

    from src.predict import predict_race, predict_next_race

    if year and round:
        with console.status(f"Predicting race results for {year} Round {round}..."):
            results = predict_race(year, round)
    else:
        with console.status("Predicting next race results..."):
            results = predict_next_race()

    if results.empty:
        console.print("[red]❌ No predictions generated. Make sure the model is trained.[/red]")
        return

    # Display results
    table = Table(title="🏁 Predicted Race Results", show_lines=True)
    table.add_column("Pos", style="bold", justify="center")
    table.add_column("Driver", style="cyan")
    table.add_column("Team", style="white")
    table.add_column("Grid", style="yellow", justify="center")
    table.add_column("Pred. Pos", style="bold green", justify="center")
    table.add_column("Range", style="magenta", justify="center")

    for idx, row in results.iterrows():
        pos = idx
        emoji = "🥇" if pos == 1 else "🥈" if pos == 2 else "🥉" if pos == 3 else ""
        grid_position = row.get("grid_position")
        grid_display = "-" if pd.isna(grid_position) else str(int(grid_position))
        table.add_row(
            f"{emoji} {pos}",
            row["driver"],
            row.get("team", ""),
            grid_display,
            str(row["predicted_position"]),
            f"{row['predicted_position_lower']:.1f}-{row['predicted_position_upper']:.1f}",
        )

    console.print(table)


@cli.command("predict-grid")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="CSV file containing a starting grid and qualifying data")
@click.option("--output", "output_path", type=click.Path(path_type=Path), help="Optional path to save predictions as CSV")
@click.option("--model-name", default="race_predictor", help="Saved model name to load")
def predict_grid(input_path: Path, output_path: Path | None, model_name: str):
    """Predict race results from a manually prepared starting-grid CSV."""
    console.print("\n[bold red]🏁 Manual Grid Prediction[/bold red]\n")

    from src.manual_predict import predict_from_grid_csv

    with console.status(f"Predicting from {input_path.name}..."):
        results = predict_from_grid_csv(input_path, model_name=model_name)

    table = Table(title=f"🏁 Predicted Results - {input_path.name}", show_lines=True)
    table.add_column("Rank", style="bold", justify="center")
    table.add_column("Driver", style="cyan")
    table.add_column("Team", style="white")
    table.add_column("Grid", style="yellow", justify="center")
    table.add_column("Pred. Pos", style="bold green", justify="center")
    table.add_column("Range", style="magenta", justify="center")

    for idx, row in results.iterrows():
        table.add_row(
            str(idx),
            row["driver"],
            row["team"],
            str(int(row["grid_position"])),
            str(row["predicted_position"]),
            f"{row['predicted_position_lower']:.1f}-{row['predicted_position_upper']:.1f}",
        )

    console.print(table)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=True)
        console.print(f"\n[green]✅ Saved predictions to {output_path}[/green]")


@cli.command("simulate-grid")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="CSV file containing a starting grid and qualifying data")
@click.option("--output", "output_path", type=click.Path(path_type=Path), help="Optional path to save simulation results as CSV")
@click.option("--model-name", default="race_predictor", help="Saved model name to load")
@click.option("--simulations", default=1000, show_default=True, help="Number of Monte Carlo race simulations")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducible simulations")
@click.option("--overtaking/--no-overtaking", default=True, show_default=True, help="Enable probabilistic adjacent overtakes")
@click.option("--strategy/--no-strategy", default=True, show_default=True, help="Enable simplified tyre/pit strategy effects")
def simulate_grid(
    input_path: Path,
    output_path: Path | None,
    model_name: str,
    simulations: int,
    seed: int,
    overtaking: bool,
    strategy: bool,
):
    """Run probabilistic Monte Carlo race simulations from a starting-grid CSV."""
    console.print("\n[bold red]🎲 Probabilistic Race Simulation[/bold red]\n")

    from src.manual_predict import simulate_from_grid_csv

    with console.status(f"Running {simulations} simulations from {input_path.name}..."):
        results = simulate_from_grid_csv(
            input_path,
            model_name=model_name,
            n_simulations=simulations,
            random_state=seed,
            enable_overtaking=overtaking,
            enable_strategy=strategy,
        )

    table = Table(title=f"🎲 Simulation Results - {input_path.name}", show_lines=True)
    table.add_column("Rank", style="bold", justify="center")
    table.add_column("Driver", style="cyan")
    table.add_column("Team", style="white")
    table.add_column("Grid", style="yellow", justify="center")
    table.add_column("Exp. Finish", style="bold green", justify="right")
    table.add_column("P5-P95", style="magenta", justify="center")
    table.add_column("Win", style="green", justify="right")
    table.add_column("Podium", style="green", justify="right")

    for _, row in results.iterrows():
        table.add_row(
            str(int(row["predicted_rank"])),
            row["driver"],
            row["team"],
            str(int(row["grid_position"])),
            f"{row['expected_finish']:.2f}",
            f"{row['finish_p05']:.1f}-{row['finish_p95']:.1f}",
            f"{row['win_probability']:.1%}",
            f"{row['podium_probability']:.1%}",
        )

    console.print(table)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        console.print(f"\n[green]✅ Saved simulation results to {output_path}[/green]")


@cli.command()
def info():
    """ℹ️  Show project information."""
    console.print(Panel.fit(
        "[bold red]🏎️  F1 Race Result Predictor[/bold red]\n\n"
        "Train a machine learning model on historical F1 data,\n"
        "qualifying pace, and strategy to predict race outcomes.\n\n"
        "[bold]Features:[/bold]\n"
        "  • Grid position, teammate delta, and gap to pole\n"
        "  • Dynamic track-aware grid weighting\n"
        "  • Driver Elo ratings plus recent grid/finish trends\n"
        "  • Team performance index and circuit-specific priors\n"
        "  • Driver and constructor circuit history\n"
        "  • Weather conditions (dry/wet/mixed)\n"
        "  • Benchmark metrics by track type\n\n"
        "[bold]Model:[/bold] XGBoost Ranker + Regressor base pace model\n"
        "[bold]Simulation:[/bold] Track-adjusted Monte Carlo race simulator\n"
        "[bold]Data Source:[/bold] FastF1 (2018-2024)\n"
        "[bold]Target:[/bold] Final race position (1-20)",
        title="About",
        border_style="red",
    ))


if __name__ == "__main__":
    cli()
