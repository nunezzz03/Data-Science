"""
Lab 5 - Time Series Data Preparation: Differentiation
Dataset: TrafficTwoMonth.csv
Target: Total (total vehicle count)

This script compares different differentiation orders and selects the best one
based on model performance (Persistence and Linear Regression).

Pipeline Step 3: Takes aggregated data, outputs differenced data.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import sqrt

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SCIENCE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, DATA_SCIENCE_ROOT)
sys.path.insert(0, os.path.join(DATA_SCIENCE_ROOT, "utils"))

from utils.dslabs_functions import (
    HEIGHT,
    dataframe_temporal_train_test_split,
    plot_forecasting_series,
    plot_forecasting_eval,
    FORECAST_MEASURES,
)

plt.style.use(f"{DATA_SCIENCE_ROOT}/utils/dslabs.mplstyle")

# Configuration
DATASET_TAG = "traffic"
TARGET = "Total"
TRAIN_PCT = 0.90
MIN_RECORDS = 10  # Minimum records required after differencing

# Input: aggregated data from step 2
INPUT_FILE = os.path.join(
    SCRIPT_DIR, "processed_data", "aggregation", f"{DATASET_TAG}_aggregated.csv"
)

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_data", "differentiation")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images", "differentiation")

# Differentiation configurations to test
DIFF_CONFIGS = [
    {
        "name": "No_Diff",
        "order": 0,
        "seasonal": None,
        "desc": "No differencing (baseline)",
    },
    {
        "name": "First_Order",
        "order": 1,
        "seasonal": None,
        "desc": "First-order (removes trend)",
    },
    {
        "name": "Second_Order",
        "order": 2,
        "seasonal": None,
        "desc": "Second-order (removes quadratic trend)",
    },
]


def load_data():
    """Load data from previous pipeline step."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Aggregated data not found: {INPUT_FILE}\nRun 02_aggregation.py first."
        )

    print(f"Loading aggregated data from: {INPUT_FILE}")
    data = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    return data


def apply_differencing(series, order, seasonal_period=None):
    """Apply differencing to make series stationary."""
    diff_series = series.copy()

    # Apply seasonal differencing first if specified
    if seasonal_period:
        diff_series = diff_series.diff(seasonal_period)

    # Apply regular differencing
    for _ in range(order):
        diff_series = diff_series.diff()

    return diff_series.dropna()


def persistence_model(train, test):
    """Persistence model: predict next = current."""
    prd_train = train.shift(1).fillna(train.iloc[0])
    prd_test = pd.Series(np.full(len(test), train.iloc[-1]), index=test.index)
    return prd_train, prd_test


def linear_regression_model(train, test):
    """Linear Regression using time index."""
    X_train = np.arange(len(train)).reshape(-1, 1)
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, train.values)

    prd_train = pd.Series(model.predict(X_train), index=train.index)
    prd_test = pd.Series(model.predict(X_test), index=test.index)
    return prd_train, prd_test


def evaluate_model(actual, predicted):
    """Calculate evaluation metrics."""
    mse = FORECAST_MEASURES["MSE"](actual.values, predicted.values)
    mae = FORECAST_MEASURES["MAE"](actual.values, predicted.values)
    r2 = FORECAST_MEASURES["R2"](actual.values, predicted.values)
    return {"MSE": mse, "MAE": mae, "R2": r2, "RMSE": sqrt(mse)}


def main():
    print("Lab 5 - Differentiation Analysis for Traffic Time Series")
    print("=" * 60)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Load data
    data = load_data()
    series = data[TARGET] if TARGET in data.columns else data.iloc[:, 0]

    print(f"\nDataset loaded: {len(series)} records")
    print(f"Date range: {series.index.min()} to {series.index.max()}")

    # Store results
    all_results = {}
    comparison_data = []

    # Test each differentiation configuration
    for config in DIFF_CONFIGS:
        name = config["name"]
        order = config["order"]
        seasonal = config["seasonal"]

        print(f"\n{'=' * 60}")
        print(f"Differentiation: {name} ({config['desc']})")
        print("=" * 60)

        # Apply differencing
        if order == 0 and seasonal is None:
            diff_series = series.copy()
        else:
            diff_series = apply_differencing(series, order, seasonal)

        print(f"After differencing: {len(diff_series)} records")

        if len(diff_series) < MIN_RECORDS:
            print(f"  WARNING: Too few records after differencing. Skipping.")
            continue

        # Train/test split
        split_idx = int(len(diff_series) * TRAIN_PCT)
        train = diff_series.iloc[:split_idx]
        test = diff_series.iloc[split_idx:]

        print(f"Train: {len(train)}, Test: {len(test)}")

        if len(test) < 2:
            print(f"  WARNING: Test set too small. Skipping.")
            continue

        # Evaluate Persistence
        prd_train_pers, prd_test_pers = persistence_model(train, test)
        metrics_pers = evaluate_model(test, prd_test_pers)
        print(
            f"\nPersistence - RMSE: {metrics_pers['RMSE']:.4f}, R2: {metrics_pers['R2']:.4f}"
        )

        # Evaluate Linear Regression
        prd_train_lr, prd_test_lr = linear_regression_model(train, test)
        metrics_lr = evaluate_model(test, prd_test_lr)
        print(
            f"LinReg     - RMSE: {metrics_lr['RMSE']:.4f}, R2: {metrics_lr['R2']:.4f}"
        )

        # Store results
        all_results[name] = {
            "differenced_series": diff_series,
            "metrics_persistence": metrics_pers,
            "metrics_lr": metrics_lr,
        }

        comparison_data.append({"Config": name, "Model": "Persistence", **metrics_pers})
        comparison_data.append(
            {"Config": name, "Model": "LinearRegression", **metrics_lr}
        )

        # Plot
        plot_forecasting_series(
            train, test, prd_test_pers, title=f"Persistence - {name}"
        )
        plt.tight_layout()
        plt.savefig(f"{IMAGES_DIR}/{DATASET_TAG}_diff_{name}_persistence.png", dpi=150)
        plt.close()

        plot_forecasting_series(
            train, test, prd_test_lr, title=f"Linear Regression - {name}"
        )
        plt.tight_layout()
        plt.savefig(f"{IMAGES_DIR}/{DATASET_TAG}_diff_{name}_lr.png", dpi=150)
        plt.close()

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(comparison_data)
    print(results_df.to_string(index=False))

    # Save results CSV
    results_df.to_csv(
        f"{IMAGES_DIR}/{DATASET_TAG}_differentiation_results.csv", index=False
    )

    # Find best configuration (by R2)
    best_idx = results_df["R2"].idxmax()
    best_row = results_df.iloc[best_idx]
    best_config_name = best_row["Config"]

    print(f"\n{'=' * 60}")
    print(f"BEST CONFIGURATION: {best_config_name}")
    print(f"  Model: {best_row['Model']}")
    print(f"  RMSE: {best_row['RMSE']:.4f}")
    print(f"  R2: {best_row['R2']:.4f}")
    print("=" * 60)

    # Save best differenced data for next step
    best_series = all_results[best_config_name]["differenced_series"]
    output_file = f"{OUTPUT_DIR}/{DATASET_TAG}_differenced.csv"
    best_series.to_frame(name=TARGET).to_csv(output_file)
    print(f"\nBest differenced data saved to: {output_file}")

    print("Done!")


if __name__ == "__main__":
    main()
