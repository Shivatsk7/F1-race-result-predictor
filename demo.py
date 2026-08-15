"""
Demo script for the F1 Race Result Predictor.

This script generates synthetic F1 data to demonstrate the full pipeline
without requiring live API access to FastF1.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import RAW_DIR, PROCESSED_DIR, PLOTS_DIR, MODELS_DIR
from src.feature_engineer import engineer_features, prepare_training_data
from src.model import RacePredictor, plot_feature_importance, plot_predictions_vs_actual, plot_position_distribution

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)


def generate_synthetic_data() -> pd.DataFrame:
    """Generate realistic synthetic F1 data for demonstration."""
    logger.info("Generating synthetic F1 data...")

    years = range(2018, 2025)
    circuits = [
        "Melbourne", "Bahrain", "Shanghai", "Baku", "Barcelona",
        "Monaco", "Montreal", "Paul Ricard", "Silverstone", "Hockenheim",
        "Hungaroring", "Spa", "Monza", "Singapore", "Sochi",
        "Suzuka", "Austin", "Mexico City", "Interlagos", "Abu Dhabi",
    ]
    drivers = [
        "HAM", "VER", "LEC", "NOR", "SAI", "PER", "RUS", "ALO",
        "PIA", "STR", "OCO", "GAS", "TSU", "ALB", "ZHO", "MAG",
        "HUL", "BOT", "RIC", "LAT",
    ]
    teams = [
        "Mercedes", "Red Bull", "Ferrari", "McLaren", "Ferrari",
        "Red Bull", "Mercedes", "Aston Martin", "McLaren", "Aston Martin",
        "Alpine", "Alpine", "AlphaTauri", "Williams", "Alfa Romeo",
        "Haas", "Haas", "Alfa Romeo", "AlphaTauri", "Williams",
    ]

    # Team strength affects base position
    team_strength = {
        "Mercedes": 2.5, "Red Bull": 2.0, "Ferrari": 3.5, "McLaren": 5.0,
        "Aston Martin": 7.0, "Alpine": 8.0, "AlphaTauri": 10.0,
        "Williams": 13.0, "Haas": 11.0, "Alfa Romeo": 9.0,
    }

    records = []
    for year in years:
        for round_num, circuit in enumerate(circuits, 1):
            for i, driver in enumerate(drivers):
                team = teams[i]
                base_pos = team_strength.get(team, 10.0)

                # Grid position: correlated with team strength + randomness
                grid_pos = max(1, min(20, int(np.random.normal(base_pos, 2.5))))

                # Qualifying gap to pole (seconds)
                quali_gap = max(0, np.random.exponential(0.3) * grid_pos * 0.1)

                # Race position: influenced by grid + randomness + DNF chance
                dnf_chance = np.random.random()
                if dnf_chance < 0.08:  # 8% DNF rate
                    race_pos = 21  # DNF marker
                    status = "DNF"
                else:
                    race_pos = max(1, min(20, int(np.random.normal(base_pos + (grid_pos - base_pos) * 0.3, 2.0))))
                    status = "Finished"

                # Weather
                weather_rand = np.random.random()
                if weather_rand < 0.7:
                    weather = {"dry": 1, "wet": 0, "mixed": 0}
                elif weather_rand < 0.9:
                    weather = {"dry": 0, "wet": 1, "mixed": 0}
                else:
                    weather = {"dry": 0, "wet": 0, "mixed": 1}

                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

                records.append({
                    "year": year,
                    "round": round_num,
                    "race_name": f"{circuit} Grand Prix",
                    "circuit": circuit,
                    "driver": driver,
                    "driver_number": i + 1,
                    "team": team,
                    "grid_position": grid_pos,
                    "qualifying_gap_to_pole": round(quali_gap, 3),
                    "race_position": race_pos if race_pos <= 20 else None,
                    "status": status,
                    "points": points_map.get(race_pos, 0),
                    "weather_dry": weather["dry"],
                    "weather_wet": weather["wet"],
                    "weather_mixed": weather["mixed"],
                })

    df = pd.DataFrame(records)
    # Remove DNF entries (no race position)
    df = df.dropna(subset=["race_position"])

    logger.info(f"Generated {len(df)} synthetic records")
    return df


def run_demo():
    """Run the full demo pipeline."""
    print("\n" + "=" * 60)
    print("  🏎️  F1 RACE RESULT PREDICTOR - DEMO")
    print("=" * 60)

    # Step 1: Generate data
    print("\n📊 Step 1: Generating synthetic F1 data...")
    df = generate_synthetic_data()
    df.to_parquet(RAW_DIR / "all_seasons.parquet", index=False)
    print(f"   → {len(df)} records across {df['year'].nunique()} seasons")

    # Step 2: Feature engineering
    print("\n🔧 Step 2: Engineering features...")
    processed_df = engineer_features(df)
    X, y, feature_list = prepare_training_data(processed_df)
    print(f"   → {len(feature_list)} features, {len(X)} samples")

    # Step 3: Train model
    print("\n🧠 Step 3: Training XGBoost model...")
    predictor = RacePredictor()
    metrics = predictor.train(X, y, race_keys=processed_df[["year", "round"]])

    # Step 4: Display results
    print("\n" + "=" * 60)
    print("  📈 MODEL EVALUATION")
    print("=" * 60)

    print(f"\n  Mean Absolute Error:      {metrics['mae']:.3f} positions")
    print(f"  Rounded Position MAE:     {metrics['mae_rounded']:.3f} positions")
    print(f"  Root Mean Squared Error:  {metrics['rmse']:.3f}")
    print(f"  R² Score:                 {metrics['r2']:.3f}")
    print(f"  Cross-Val MAE:            {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f}")
    print(f"\n  Exact Position Accuracy:  {metrics['exact_accuracy']:.1%}")
    print(f"  Within 1 Position:        {metrics['within_1_accuracy']:.1%}")
    print(f"  Within 2 Positions:       {metrics['within_2_accuracy']:.1%}")
    print(f"  Within 3 Positions:       {metrics['within_3_accuracy']:.1%}")
    print(f"  Podium Prediction:        {metrics['podium_accuracy']:.1%}")

    # Feature importance
    print("\n" + "=" * 60)
    print("  🔑 TOP 10 FEATURE IMPORTANCE")
    print("=" * 60)

    importance = predictor.get_feature_importance().head(10)
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 80)
        print(f"  {row['feature']:40s} {row['importance']:.4f} {bar}")

    # Step 5: Save model and plots
    print("\n💾 Step 5: Saving model and generating plots...")
    predictor.save()

    try:
        plot_feature_importance(predictor)
        eval_data = predictor.evaluation_data or {}
        if eval_data:
            plot_predictions_vs_actual(predictor, eval_data["X_test"], eval_data["y_test"])
            plot_position_distribution(predictor, eval_data["X_test"], eval_data["y_test"])
        print(f"   → Plots saved to {PLOTS_DIR}/")
    except Exception as e:
        print(f"   ⚠️  Could not generate plots: {e}")

    # Step 6: Demo prediction
    print("\n" + "=" * 60)
    print("  🏁 SAMPLE PREDICTION - 2024 Miami Grand Prix")
    print("=" * 60)

    # Create a sample prediction scenario
    sample_drivers = ["VER", "LEC", "NOR", "HAM", "SAI", "PER", "RUS", "ALO", "PIA", "STR"]
    sample_teams = ["Red Bull", "Ferrari", "McLaren", "Mercedes", "Ferrari",
                    "Red Bull", "Mercedes", "Aston Martin", "McLaren", "Aston Martin"]
    sample_grids = [1, 3, 2, 5, 4, 6, 7, 8, 9, 10]

    sample_data = []
    for i, (driver, team, grid) in enumerate(zip(sample_drivers, sample_teams, sample_grids)):
        sample_data.append({
            "year": 2024, "round": 21, "race_name": "Miami Grand Prix",
            "circuit": "Miami", "driver": driver, "driver_number": i + 1,
            "team": team, "grid_position": grid,
            "qualifying_gap_to_pole": grid * 0.08,
            "race_position": None, "status": None, "points": None,
            "weather_dry": 1, "weather_wet": 0, "weather_mixed": 0,
            "_is_prediction_row": True,
        })

    sample_df = pd.DataFrame(sample_data)
    historical_df = df.copy()
    historical_df["_is_prediction_row"] = False
    combined = pd.concat([historical_df, sample_df], ignore_index=True)
    combined = engineer_features(combined, keep_unlabeled=True, save_processed=False)

    pred_features = combined[combined["_is_prediction_row"]].copy()

    if not pred_features.empty and all(f in pred_features.columns for f in predictor.feature_names):
        X_sample = pred_features[predictor.feature_names]
        prediction_frame = predictor.predict_with_confidence(X_sample)

        results = pred_features[["driver", "team", "grid_position"]].copy()
        results = pd.concat([results.reset_index(drop=True), prediction_frame], axis=1)
        results = results.sort_values("predicted_position").reset_index(drop=True)

        print(f"\n  {'Pos':<5} {'Driver':<8} {'Team':<15} {'Grid':<6} {'Pred':<6} {'Range':<11}")
        print("  " + "-" * 58)
        for idx, row in results.iterrows():
            emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "  "
            interval = f"{row['predicted_position_lower']:.1f}-{row['predicted_position_upper']:.1f}"
            print(
                f"  {emoji}{idx+1:<3} {row['driver']:<8} {row['team']:<15} "
                f"{int(row['grid_position']):<6} {row['predicted_position']:<6} {interval:<11}"
            )

    print("\n" + "=" * 60)
    print("  ✅ Demo complete!")
    print("=" * 60)
    print(f"\n  Model saved to: {MODELS_DIR}/")
    print(f"  Plots saved to: {PLOTS_DIR}/")
    print(f"  Data saved to:  {RAW_DIR}/")
    print()


if __name__ == "__main__":
    run_demo()
