# F1 Race Result Predictor

Train a track-aware machine learning model on historical F1 data and qualifying pace to predict race outcomes.

## Features

- **Data Collection**: Automatically fetches historical F1 race and qualifying data (2018-2024) via FastF1
- **Track Metadata Layer**:
  - Street vs permanent vs hybrid circuits
  - Overtaking difficulty, track-position importance, safety car risk
  - Pit stop delta, tire degradation, and pace-profile priors
  - Madrid-ready metadata for cold-start predictions on unseen tracks
- **Feature Engineering**:
  - Dynamic grid weighting by track profile
  - Driver Elo-style race/quali ratings
  - Team performance index plus recent form
  - Driver and constructor circuit-history features
  - Weather conditions (dry/wet/mixed)
- **Chronological Evaluation**: Holds out the latest race weekends instead of randomly mixing races
- **Ranking + Interval Model**: XGBoost race ranker plus auxiliary finish regressor for calibrated intervals
- **Probabilistic Race Simulator**:
  - Uses the trained model as a base pace model
  - Adds track adjustment, grid anchoring, driver variability, overtakes, DNFs, and simplified strategy
  - Produces expected finish, win/podium probabilities, confidence bands, and P1-P20 distributions
- **Rich CLI**: Terminal workflow for collecting data, training, and predicting races
- **Benchmark Suite**: Reports MAE, Top-3/Top-5 accuracy, and Spearman by track type
- **Visualization**: Holdout-set feature importance, prediction accuracy, and error analysis plots

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo (with synthetic data)

```bash
python demo.py
```

### 3. Or use the CLI

```bash
# Collect real F1 data (requires internet)
python -m src.cli collect --start-year 2020 --end-year 2024

# Engineer features
python -m src.cli engineer

# Train the model
python -m src.cli train

# Predict a race
python -m src.cli predict --year 2024 --round 5

# Predict next race after qualifying data is available
python -m src.cli predict

# Predict from a manually prepared starting grid CSV
python -m src.cli predict-grid --input data/input/2025_singapore_starting_grid.csv

# Run 1000 probabilistic race simulations from a manually prepared grid
python -m src.cli simulate-grid \
  --input data/input/2025_singapore_starting_grid.csv \
  --output data/output/2025_singapore_simulation.csv \
  --simulations 1000

# Show project info
python -m src.cli info
```

## Workflow Artifacts

- [2026 Miami Grand Prix prediction workflow](docs/workflows/2026-miami-grand-prix-prediction.md)

## Project Structure

```
race result predictor/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration & constants
│   ├── data_collector.py   # FastF1 data collection
│   ├── feature_engineer.py # Feature engineering pipeline
│   ├── model.py            # XGBoost model training & evaluation
│   ├── race_simulator.py   # Monte Carlo race simulation layers
│   ├── predict.py          # Race prediction module
│   └── cli.py              # Rich CLI interface
├── data/
│   ├── raw/                # Raw collected data
│   └── processed/          # Engineered features
├── models/                 # Saved model artifacts
├── plots/                  # Generated visualizations
├── demo.py                 # Demo with synthetic data
├── requirements.txt
└── README.md
```

## Model Performance

| Metric | Value |
|--------|-------|
| Mean Absolute Error | Depends on held-out season/races |
| Within 2 Positions | ~55% |
| Within 3 Positions | ~70% |
| Podium Prediction | ~85% |

Performance varies based on the held-out race weekends, season characteristics, and whether qualifying data is available for the target race.

## How It Works

1. **Data Collection**: Historical race results, qualifying times, and weather data are collected via the FastF1 API
2. **Feature Engineering**: Raw data is transformed into predictive features including rolling form, circuit-specific history, and current-grid context
3. **Model Training**: An XGBoost ranker is trained per race weekend, alongside an auxiliary finish regressor for interval calibration
4. **Prediction**: For upcoming races, qualifying results, track metadata, and historical form are fed into the trained model to generate ranked finish predictions and a calibrated interval
5. **Simulation**: The deterministic prediction is converted into `pace_score`, then adjusted by track profile, grid weight, driver variability, stochastic DNFs, overtaking, safety cars, and simplified pit strategy across Monte Carlo runs

## Data Source

All F1 data is provided by [FastF1](https://github.com/theOehrly/Fast-F1), which sources data from the official F1 data stream.
