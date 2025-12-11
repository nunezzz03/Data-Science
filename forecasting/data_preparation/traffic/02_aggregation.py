"""
Lab 5 - Time Series Data Preparation: Aggregation
Dataset: TrafficTwoMonth.csv
Target: Total (total vehicle count)

This script compares different aggregation levels and selects the best one
based on model performance (Persistence and Linear Regression).

Pipeline Step 2: Takes scaled data, outputs aggregated data.
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

# Input: scaled data from step 1 (or raw if no scaling selected)
INPUT_FILE = os.path.join(
    SCRIPT_DIR, "processed_data", "scaling", f"{DATASET_TAG}_scaled.csv"
)
# Fallback to raw data if scaled doesn't exist
RAW_FILE = f"{DATA_SCIENCE_ROOT}/data/raw/TrafficTwoMonth.csv"

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_data", "aggregation")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images", "aggregation")

# Aggregation configurations to test
# For 15-min interval data: H=hourly, 4H=4-hourly, D=daily
AGGREGATION_CONFIGS = [
    {"name": "Hourly_Mean", "gran_level": "H", "agg_func": "mean"},
    {"name": "4Hourly_Mean", "gran_level": "4H", "agg_func": "mean"},
    {"name": "Daily_Mean", "gran_level": "D", "agg_func": "mean"},
]


def load_data():
    """Load data from previous pipeline step or raw data."""
    if os.path.exists(INPUT_FILE):
        print(f"Loading scaled data from: {INPUT_FILE}")
        data = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    else:
        print(f"Scaled data not found. Loading raw data from: {RAW_FILE}")
        data = pd.read_csv(RAW_FILE)
        # Create datetime index
        start_date = pd.Timestamp("2024-10-10 00:00:00")
        data["datetime"] = pd.date_range(
            start=start_date, periods=len(data), freq="15min"
        )
        data = data.set_index("datetime")
        # Keep only target
        data = data[[TARGET]]

    return data


def aggregate_series(series, gran_level, agg_func):
    """Aggregate time series to specified granularity."""
    if agg_func == "mean":
        return series.resample(gran_level).mean().dropna()
    elif agg_func == "median":
        return series.resample(gran_level).median().dropna()
    elif agg_func == "sum":
        return series.resample(gran_level).sum().dropna()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")


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
    print("Lab 5 - Aggregation Analysis for Traffic Time Series")
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

    # Test each aggregation configuration
    for config in AGGREGATION_CONFIGS:
        name = config["name"]
        gran = config["gran_level"]
        func = config["agg_func"]

        print(f"\n{'=' * 60}")
        print(f"Aggregation: {name} (level={gran}, func={func})")
        print("=" * 60)

        # Aggregate
        agg_series = aggregate_series(series, gran, func)
        print(f"Aggregated size: {len(agg_series)} records")

        if len(agg_series) < 10:
            print(f"  WARNING: Too few records after aggregation. Skipping.")
            continue

        # Train/test split
        split_idx = int(len(agg_series) * TRAIN_PCT)
        train = agg_series.iloc[:split_idx]
        test = agg_series.iloc[split_idx:]

        print(f"Train: {len(train)}, Test: {len(test)}")

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
            "aggregated_series": agg_series,
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
        plt.savefig(f"{IMAGES_DIR}/{DATASET_TAG}_agg_{name}_persistence.png", dpi=150)
        plt.close()

        plot_forecasting_series(
            train, test, prd_test_lr, title=f"Linear Regression - {name}"
        )
        plt.tight_layout()
        plt.savefig(f"{IMAGES_DIR}/{DATASET_TAG}_agg_{name}_lr.png", dpi=150)
        plt.close()

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(comparison_data)
    print(results_df.to_string(index=False))

    # Save results CSV
    results_df.to_csv(
        f"{IMAGES_DIR}/{DATASET_TAG}_aggregation_results.csv", index=False
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

    # Save best aggregated data for next step
    best_series = all_results[best_config_name]["aggregated_series"]
    output_file = f"{OUTPUT_DIR}/{DATASET_TAG}_aggregated.csv"
    best_series.to_frame(name=TARGET).to_csv(output_file)
    print(f"\nBest aggregated data saved to: {output_file}")

    print("Done!")


if __name__ == "__main__":
    main()
