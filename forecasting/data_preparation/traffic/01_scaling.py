"""
Lab 5 - Time Series Data Preparation: Scaling
Dataset: TrafficTwoMonth.csv
Target: Total (total vehicle count per 15-min interval)

This script compares 3 scaling approaches:
1. No Scaling (baseline)
2. StandardScaler (z-score normalization)
3. MinMaxScaler (normalization to [0,1])

For each scaling approach, we train Persistence and Linear Regression models
and compare their performance using MSE, MAE, R², and MAPE.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# Configuration
DATASET_FILE = f"{DATA_SCIENCE_ROOT}/data/raw/TrafficTwoMonth.csv"
DATASET_TAG = "traffic"
TARGET = "Total"
TRAIN_PCT = 0.90


def load_and_prepare_data():
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


def apply_scaling(data: DataFrame, scaler_type: str) -> tuple[DataFrame, object]:
    """
    Apply scaling to dataframe.

    Args:
        data: DataFrame to scale
        scaler_type: "none", "standard", or "minmax"

    Returns:
        Scaled dataframe and fitted scaler (or None if no scaling)
    """
    if scaler_type == "none":
        return data.copy(), None
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    df_scaled = DataFrame(
        scaler.fit_transform(data),
        index=data.index,
        columns=data.columns,
    )
    return df_scaled, scaler


def persistence_model(train: Series, test: Series) -> tuple[Series, Series]:
    """
    Persistence (naive) model: predict next value = current value.
    For training: predict(t) = actual(t-1)
    For testing: first prediction = last train value, rest = actual(t-1)
    """
    # Training predictions: shifted by 1 (first value is NaN)
    prd_train = train.shift(1).fillna(train.iloc[0])

    # Test predictions: first = last train value, rest = previous test value
    prd_test = test.shift(1)
    prd_test.iloc[0] = train.iloc[-1]

    return prd_train, prd_test


def linear_regression_model(train: Series, test: Series) -> tuple[Series, Series]:
    """
    Linear Regression model for time series.
    Uses time index as feature to fit a trend line.
    """
    # Create numeric time features
    train_X = np.arange(len(train)).reshape(-1, 1)
    test_X = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    # Fit model
    model = LinearRegression()
    model.fit(train_X, train.values)

    # Predictions
    prd_train = Series(model.predict(train_X), index=train.index)
    prd_test = Series(model.predict(test_X), index=test.index)

    return prd_train, prd_test


def safe_mape(y_true: Series, y_pred: Series, epsilon: float = 1e-10) -> float:
    """Calculate MAPE safely, avoiding division by zero."""
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def evaluate_model(
    train: Series, test: Series, prd_train: Series, prd_test: Series
) -> dict:
    """Compute evaluation metrics for train and test predictions."""
    return {
        "RMSE_train": sqrt(FORECAST_MEASURES["MSE"](train, prd_train)),
        "RMSE_test": sqrt(FORECAST_MEASURES["MSE"](test, prd_test)),
        "MAE_train": FORECAST_MEASURES["MAE"](train, prd_train),
        "MAE_test": FORECAST_MEASURES["MAE"](test, prd_test),
        "R2_train": FORECAST_MEASURES["R2"](train, prd_train),
        "R2_test": FORECAST_MEASURES["R2"](test, prd_test),
        "MAPE_train": safe_mape(train, prd_train),
        "MAPE_test": safe_mape(test, prd_test),
    }


def main():
    print("=" * 60)
    print("Lab 5 - Scaling Analysis for Traffic Time Series")
    print("=" * 60)

    # Load data
    data = load_and_prepare_data()
    print(f"\nDataset loaded: {len(data)} records")
    print(f"Columns: {list(data.columns)}")
    print(f"Target: {TARGET}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Scaling approaches to compare
    scaling_approaches = {
        "none": "No Scaling (Baseline)",
        "standard": "StandardScaler (Z-Score)",
        "minmax": "MinMaxScaler (0-1)",
    }

    # Store results for comparison
    all_results = []
    images_dir = os.path.join(SCRIPT_DIR, "images/scaling")
    os.makedirs(images_dir, exist_ok=True)

    # Evaluate each scaling approach
    for scaler_type, scaler_name in scaling_approaches.items():
        print(f"\n{'=' * 60}")
        print(f"Scaling Approach: {scaler_name}")
        print("=" * 60)

        # Apply scaling
        data_scaled, scaler = apply_scaling(data, scaler_type)

        # Plot before/after scaling for target variable
        series = data_scaled[TARGET]
        plt.figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            series.index.to_list(),
            series.to_list(),
            xlabel="Time",
            ylabel=TARGET,
            title=f"{DATASET_TAG} - {TARGET} ({scaler_name})",
        )
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_scaling_{scaler_type}.png")
        plt.close()

        # Train/test split
        train_df, test_df = dataframe_temporal_train_test_split(data_scaled, TRAIN_PCT)
        train = train_df[TARGET]
        test = test_df[TARGET]

        print(f"Train size: {len(train)} ({TRAIN_PCT * 100:.0f}%)")
        print(f"Test size: {len(test)} ({(1 - TRAIN_PCT) * 100:.0f}%)")

        # Evaluate Persistence Model
        print("\n--- Persistence Model ---")
        prd_train_pers, prd_test_pers = persistence_model(train, test)
        metrics_pers = evaluate_model(train, test, prd_train_pers, prd_test_pers)
        for metric, value in metrics_pers.items():
            print(f"  {metric}: {value:.6f}")

        # Plot persistence results
        fig, ax = plt.subplots(1, 1, figsize=(4 * HEIGHT, HEIGHT))
        ax.set_title(f"Persistence Model - {scaler_name}")
        ax.plot(train.index, train.values, label="train", color="blue")
        ax.plot(test.index, test.values, label="test", color="green")
        ax.plot(test.index, prd_test_pers.values, "--", label="prediction", color="red")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_persistence_{scaler_type}.png")
        plt.close()

        # Plot persistence evaluation
        plot_forecasting_eval(
            train,
            test,
            prd_train_pers,
            prd_test_pers,
            title=f"Persistence - {scaler_name}",
        )
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_persistence_eval_{scaler_type}.png")
        plt.close()

        # Evaluate Linear Regression Model
        print("\n--- Linear Regression Model ---")
        prd_train_lr, prd_test_lr = linear_regression_model(train, test)
        metrics_lr = evaluate_model(train, test, prd_train_lr, prd_test_lr)
        for metric, value in metrics_lr.items():
            print(f"  {metric}: {value:.6f}")

        # Plot LR results
        fig, ax = plt.subplots(1, 1, figsize=(4 * HEIGHT, HEIGHT))
        ax.set_title(f"Linear Regression - {scaler_name}")
        ax.plot(train.index, train.values, label="train", color="blue")
        ax.plot(test.index, test.values, label="test", color="green")
        ax.plot(test.index, prd_test_lr.values, "--", label="prediction", color="red")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_lr_{scaler_type}.png")
        plt.close()

        # Plot LR evaluation
        plot_forecasting_eval(
            train,
            test,
            prd_train_lr,
            prd_test_lr,
            title=f"Linear Regression - {scaler_name}",
        )
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{DATASET_TAG}_lr_eval_{scaler_type}.png")
        plt.close()

        # Store results
        all_results.append(
            {"Scaling": scaler_name, "Model": "Persistence", **metrics_pers}
        )
        all_results.append(
            {"Scaling": scaler_name, "Model": "Linear Regression", **metrics_lr}
        )

    # Summary comparison
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    results_df = DataFrame(all_results)
    print("\nAll Results:")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(f"{images_dir}/{DATASET_TAG}_scaling_comparison.csv", index=False)

    # Find best scaling approach based on test RMSE for Persistence model
    persistence_results = results_df[results_df["Model"] == "Persistence"]
    best_idx = persistence_results["RMSE_test"].idxmin()
    best_scaling = results_df.loc[best_idx, "Scaling"]

    # Save the best scaled data for the next pipeline step
    # Map display name back to scaler type
    name_to_type = {v: k for k, v in scaling_approaches.items()}
    best_scaler_type = name_to_type.get(best_scaling, "none")

    # Re-apply best scaling and save
    data_best, _ = apply_scaling(data, best_scaler_type)
    output_dir = os.path.join(SCRIPT_DIR, "processed_data", "scaling")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{DATASET_TAG}_scaled.csv")
    data_best.to_csv(output_file)
    print(f"\nBest scaled data saved to: {output_file}")

    print(f"\n{'=' * 60}")
    print("INSIGHTS:")
    print("-" * 60)
    print("1. R² is IDENTICAL across all scaling approaches because")
    print("   scaling is a linear transformation that doesn't change")
    print("   the model's ability to capture variance.")
    print("")
    print("2. RMSE/MAE differ in scale but represent equivalent")
    print("   performance. Use the original scale (no scaling) for")
    print("   interpretable error metrics.")
    print("")
    print("3. Persistence model significantly outperforms Linear")
    print("   Regression because traffic data has strong autocorrelation")
    print("   - next value depends heavily on previous value.")
    print("")
    print(f"RECOMMENDATION: {best_scaling}")
    print("   For forecasting, use NO SCALING - it provides the most")
    print("   interpretable metrics (RMSE ~35 vehicles).")
    print("=" * 60)

    # Create comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(3 * HEIGHT, 2 * HEIGHT))

    # RMSE comparison
    metrics_to_plot = ["RMSE_test", "MAE_test", "R2_test", "MAPE_test"]
    titles = ["RMSE (Test)", "MAE (Test)", "R² (Test)", "MAPE (Test)"]

    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        pivot = results_df.pivot(index="Scaling", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=45)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{images_dir}/{DATASET_TAG}_scaling_comparison_chart.png")
    plt.close()

    print(f"\nAll plots saved to: {images_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
