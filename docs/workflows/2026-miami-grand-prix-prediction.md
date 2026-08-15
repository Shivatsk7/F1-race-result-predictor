# 2026 Miami Grand Prix Prediction Workflow

Reusable workflow for generating a prediction for the 2026 Miami Grand Prix using prior seasons plus 2026 season-to-date data.

## Parameters

- `TARGET_YEAR=2026`
- `TARGET_ROUND=4`
- `HISTORICAL_START_YEAR=2018`
- `HISTORICAL_END_YEAR=2026`

## Prerequisites

- A Python environment with the project dependencies installed.
- FastF1 cache enabled locally.
- Access to the repo's cached raw data under `data/raw/` and `data/fastf1_cache/`.

## Workflow

1. Refresh the raw dataset so it includes all prior years plus 2026 season-to-date.

```bash
python -m src.cli collect --start-year 2018 --end-year 2026
```

2. Build the engineered training dataset.

```bash
python -m src.cli engineer
```

3. Train the model on the engineered data.

```bash
python -m src.cli train
```

4. Generate the Miami prediction.

```bash
python -m src.cli predict --year 2026 --round 4
```

5. If live FastF1 session access is unavailable but you do have the Miami qualifying order and times, fill in `data/input/2026_miami_grid_template.csv` and run the manual-grid path instead.

```bash
python -m src.cli predict-grid --input data/input/2026_miami_grid_template.csv
```

## Expected Output

- A ranked table of predicted finishing positions.
- Rounded predicted position plus a simple confidence interval.
- Saved model artifacts in `models/` if training is run with save enabled.
- Diagnostic plots in `plots/`.

## Reuse Notes

- To predict a different race, change `TARGET_YEAR` and `TARGET_ROUND`.
- If 2026 season data is already cached, you can skip the collect step and rerun only engineer, train, and predict.
- For the most realistic pre-race result, run the workflow after qualifying is available for the target round.
- The manual-grid command is useful when you want an offline pre-race prediction from an official starting grid or qualifying results page.
