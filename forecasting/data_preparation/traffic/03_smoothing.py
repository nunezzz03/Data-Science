"""
Lab 5 - Time Series Data Preparation: Smoothing
Dataset: TrafficTwoMonth.csv
Target: Total (total vehicle count per 15-min interval)

This script compares different smoothing window sizes using rolling mean,
following Professor Claudia Antunes' DSLabs style.

For each smoothing approach, we train Persistence and Linear Regression models
and compare their performance using MSE, MAE, R², and MAPE.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import read_csv, DataFrame, Series
from sklearn.linear_model import LinearRegression
from math import sqrt

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SCIENCE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, DATA_SCIENCE_ROOT)
sys.path.insert(0, os.path.join(DATA_SCIENCE_ROOT, "utils"))

from utils.dslabs_functions import (
    plot_line_chart,
    HEIGHT,
    dataframe_temporal_train_test_split,
    plot_forecasting_series,
    plot_forecasting_eval,
    FORECAST_MEASURES,
)

plt.style.use(f"{DATA_SCIENCE_ROOT}/utils/dslabs.mplstyle")

# Configuration
DATASET_FILE = f"{DATA_SCIENCE_ROOT}/data/raw/TrafficTwoMonth.csv"
DATASET_TAG = "traffic"
TARGET = "Total"
TRAIN_PCT = 0.90

# Window sizes for smoothing (professor's style)
# For 15-min intervals: 4=1h, 24=6h, 96=1day
WIN_SIZES: list[int] = [4, 24, 96]


def load_and_prepare_data() -> DataFrame:
    """Load traffic data and create proper datetime index."""
    data = read_csv(DATASET_FILE, sep=",", decimal=".")

    # Create synthetic datetime index (15-min intervals)
    start_date = pd.Timestamp("2024-10-10 00:00:00")
    data["datetime"] = pd.date_range(start=start_date, periods=len(data), freq="15min")
    data = data.set_index("datetime")

    # Keep only numeric columns for time series analysis
    numeric_cols = ["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]
    data = data[numeric_cols]

    return data


def persistence_model(train: Series, test: Series) -> tuple[Series, Series]:
    """
    Persistence (naive) model: predict next value = current value.
    """
    # Training predictions: shifted by 1 (first value is NaN)
    prd_train = train.shift(1).fillna(train.iloc[0])

    # Test predictions: first = last train value, rest = previous test value
    prd_test = test.shift(1)
    prd_test.iloc[0] = train.iloc[-1]

    return prd_train, prd_test


def linear_regression_model(train: Series, test: Series) -> tuple[Series, Series]:
    """
    Linear Regression model using time index as feature.
    """
    # Create time-based features (integer index)
    X_train = np.arange(len(train)).reshape(-1, 1)
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    y_train = train.values

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    prd_train = Series(model.predict(X_train), index=train.index)
    prd_test = Series(model.predict(X_test), index=test.index)

    return prd_train, prd_test


def safe_mape(actual: Series, predicted: Series, epsilon: float = 1.0) -> float:
    """Calculate MAPE safely, avoiding division by zero."""
    mask = abs(actual) > epsilon
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate_model(
    train: Series, test: Series, prd_train: Series, prd_test: Series
) -> dict:
    """Calculate evaluation metrics for train and test sets."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics = {
        "RMSE_train": sqrt(mean_squared_error(train, prd_train)),
        "RMSE_test": sqrt(mean_squared_error(test, prd_test)),
        "MAE_train": mean_absolute_error(train, prd_train),
        "MAE_test": mean_absolute_error(test, prd_test),
        "R2_train": r2_score(train, prd_train),
        "R2_test": r2_score(test, prd_test),
        "MAPE_train": safe_mape(train, prd_train),
        "MAPE_test": safe_mape(test, prd_test),
    }
    return metrics


def main():
    print("Lab 5 - Smoothing Analysis for Traffic Time Series")
    print("=" * 60)

    # Load data
    data = load_and_prepare_data()
    series: Series = data[TARGET]
    print(f"\nDataset loaded: {len(series)} records")
    print(f"Target: {TARGET}")
    print(f"Date range: {series.index.min()} to {series.index.max()}")

    # Create images directory
    images_dir = os.path.join(SCRIPT_DIR, "images/smoothing")
    os.makedirs(images_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Plot 1: Smoothing comparison (professor's style)
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(
        len(WIN_SIZES), 1, figsize=(3 * HEIGHT, HEIGHT * len(WIN_SIZES))
    )
    fig.suptitle(f"{DATASET_TAG} - Smoothing Effect on {TARGET}")

    for i, win in enumerate(WIN_SIZES):
        smoothed = series.rolling(window=win).mean()
        ax = axs[i] if len(WIN_SIZES) > 1 else axs
        ax.plot(series.index, series.values, label="Original", alpha=0.5)
        ax.plot(
            smoothed.index, smoothed.values, label=f"Smoothed (w={win})", color="red"
        )
        ax.set_title(f"Window Size = {win} ({win * 15} min)")
        ax.set_xlabel("Time")
        ax.set_ylabel(TARGET)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{images_dir}/{DATASET_TAG}_smoothing_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved: {DATASET_TAG}_smoothing_comparison.png")

    # -------------------------------------------------------------------------
    # Evaluate each smoothing approach with Persistence & Linear Regression
    # -------------------------------------------------------------------------
    all_results = []

    # First: No smoothing (baseline)
    smoothing_configs = [("none", 0)] + [("rolling", w) for w in WIN_SIZES]

    for method, win in smoothing_configs:
        if method == "none":
            name = "No Smoothing"
            series_smooth = series.copy()
        else:
            name = f"Rolling (w={win})"
            series_smooth = series.rolling(window=win).mean()
            series_smooth = series_smooth.fillna(series)  # Fill NaN at start

        print(f"\n{'=' * 60}")
        print(f"Smoothing: {name}")
        print("=" * 60)

        # Create dataframe for train/test split
        df_smooth = DataFrame({TARGET: series_smooth})

        # Train/test split
        train_df, test_df = dataframe_temporal_train_test_split(df_smooth, TRAIN_PCT)
        train = train_df[TARGET]
        test = test_df[TARGET]

        print(f"Train size: {len(train)} ({TRAIN_PCT * 100:.0f}%)")
        print(f"Test size: {len(test)} ({(1 - TRAIN_PCT) * 100:.0f}%)")

        # --- Persistence Model ---
        print("\n--- Persistence Model ---")
        prd_train_pers, prd_test_pers = persistence_model(train, test)
        metrics_pers = evaluate_model(train, test, prd_train_pers, prd_test_pers)
        for metric, value in metrics_pers.items():
            print(f"  {metric}: {value:.6f}")

        # Plot persistence forecast
        plot_forecasting_series(
            train, test, prd_test_pers, title=f"Persistence - {name}"
        )
        safe_name = (
            name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        )
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_persistence_{safe_name}.png", dpi=150)
        plt.close()

        # Plot persistence evaluation
        plot_forecasting_eval(
            train, test, prd_train_pers, prd_test_pers, title=f"Persistence - {name}"
        )
        plt.tight_layout()
        plt.savefig(
            f"{images_dir}/{DATASET_TAG}_persistence_eval_{safe_name}.png", dpi=150
        )
        plt.close()

        # --- Linear Regression Model ---
        print("\n--- Linear Regression Model ---")
        prd_train_lr, prd_test_lr = linear_regression_model(train, test)
        metrics_lr = evaluate_model(train, test, prd_train_lr, prd_test_lr)
        for metric, value in metrics_lr.items():
            print(f"  {metric}: {value:.6f}")

        # Plot LR forecast
        plot_forecasting_series(
            train, test, prd_test_lr, title=f"Linear Regression - {name}"
        )
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_lr_{safe_name}.png", dpi=150)
        plt.close()

        # Plot LR evaluation
        plot_forecasting_eval(
            train, test, prd_train_lr, prd_test_lr, title=f"Linear Regression - {name}"
        )
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_lr_eval_{safe_name}.png", dpi=150)
        plt.close()

        # Store results
        all_results.append({"Smoothing": name, "Model": "Persistence", **metrics_pers})
        all_results.append(
            {"Smoothing": name, "Model": "Linear Regression", **metrics_lr}
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    results_df = DataFrame(all_results)
    print("\nAll Results:")
    print(results_df.to_string(index=False))

    # Save to CSV
    results_df.to_csv(f"{images_dir}/{DATASET_TAG}_smoothing_results.csv", index=False)

    # Best by model type
    print("\n" + "-" * 60)
    print("PERSISTENCE MODEL (sorted by RMSE_test):")
    pers_df = results_df[results_df["Model"] == "Persistence"].sort_values("RMSE_test")
    print(
        pers_df[["Smoothing", "RMSE_test", "MAE_test", "R2_test"]].to_string(
            index=False
        )
    )

    print("\n" + "-" * 60)
    print("LINEAR REGRESSION MODEL (sorted by RMSE_test):")
    lr_df = results_df[results_df["Model"] == "Linear Regression"].sort_values(
        "RMSE_test"
    )
    print(
        lr_df[["Smoothing", "RMSE_test", "MAE_test", "R2_test"]].to_string(index=False)
    )

    # Best overall
    best_pers = pers_df.iloc[0]
    best_lr = lr_df.iloc[0]

    print("\n" + "=" * 60)
    print("CONCLUSIONS:")
    print("-" * 60)
    print(
        f"Best Persistence: {best_pers['Smoothing']} (RMSE={best_pers['RMSE_test']:.4f})"
    )
    print(f"Best Linear Reg:  {best_lr['Smoothing']} (RMSE={best_lr['RMSE_test']:.4f})")
    print("=" * 60)

    # Comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(3 * HEIGHT, 2 * HEIGHT))
    fig.suptitle(f"{DATASET_TAG} - Smoothing Results Comparison")

    metrics_to_plot = ["RMSE_test", "MAE_test", "R2_test", "MAPE_test"]
    titles = ["RMSE (Test)", "MAE (Test)", "R² (Test)", "MAPE (Test)"]

    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        pivot = results_df.pivot(index="Smoothing", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=45)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{images_dir}/{DATASET_TAG}_smoothing_comparison_chart.png", dpi=150)
    plt.close()

    print(f"\nAll images saved to: {images_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
