"""
Lab 5 - Time Series Data Preparation: Scaling
Dataset: economic_indicators_dataset_2010_2023.csv
Target: Inflation Rate (%) - USA only

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

plt.style.use(f"{DATA_SCIENCE_ROOT}/utils/dslabs.mplstyle")

# Configuration
DATASET_FILE = f"{DATA_SCIENCE_ROOT}/data/raw/economic_indicators_dataset_2010_2023.csv"
DATASET_TAG = "economic_usa"
TARGET = "Inflation Rate (%)"
COUNTRY = "USA"
TRAIN_PCT = 0.90


def load_and_prepare_data():
    """Load economic data for USA and create proper datetime index."""
    data = read_csv(DATASET_FILE, sep=",", decimal=".")

    # Filter for USA only
    data = data[data["Country"] == COUNTRY].copy()

    # Parse date and set as index
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    data = data.set_index("Date")

    # Keep only numeric columns for time series analysis
    numeric_cols = [
        "Inflation Rate (%)",
        "GDP Growth Rate (%)",
        "Unemployment Rate (%)",
        "Interest Rate (%)",
        "Stock Index Value",
    ]
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


def safe_mape(actual: Series, predicted: Series, epsilon: float = 0.1) -> float:
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
    print("=" * 60)
    print("Lab 5 - Scaling Analysis for Economic Time Series (USA)")
    print("=" * 60)

    # Load data
    data = load_and_prepare_data()
    print(f"\nDataset loaded: {len(data)} records (USA only)")
    print(f"Columns: {list(data.columns)}")
    print(f"Target: {TARGET}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Scaling approaches to compare
    scaling_approaches = [
        ("none", "No Scaling (Baseline)"),
        ("standard", "StandardScaler (Z-Score)"),
        ("minmax", "MinMaxScaler ([0,1])"),
    ]

    # Store results for comparison
    all_results = []
    images_dir = os.path.join(SCRIPT_DIR, "images/scaling")
    os.makedirs(images_dir, exist_ok=True)

    # Evaluate each scaling approach
    for scaler_type, name in scaling_approaches:
        print(f"\n{'=' * 60}")
        print(f"Scaling Approach: {name}")
        print("=" * 60)

        # Apply scaling
        data_scaled, scaler = apply_scaling(data, scaler_type)

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
        safe_name = (
            name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        )
        plot_forecasting_series(
            train, test, prd_test_pers, title=f"Persistence - {name}"
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

        # Evaluate Linear Regression Model
        print("\n--- Linear Regression Model ---")
        prd_train_lr, prd_test_lr = linear_regression_model(train, test)
        metrics_lr = evaluate_model(train, test, prd_train_lr, prd_test_lr)
        for metric, value in metrics_lr.items():
            print(f"  {metric}: {value:.6f}")

        # Plot LR results
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
        all_results.append({"Scaling": name, "Model": "Persistence", **metrics_pers})
        all_results.append(
            {"Scaling": name, "Model": "Linear Regression", **metrics_lr}
        )

    # Summary comparison
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    results_df = DataFrame(all_results)
    print("\nAll Results:")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(f"{images_dir}/{DATASET_TAG}_scaling_results.csv", index=False)

    # Best by model type
    print("\n" + "-" * 60)
    print("PERSISTENCE MODEL (sorted by RMSE_test):")
    pers_df = results_df[results_df["Model"] == "Persistence"].sort_values("RMSE_test")
    print(
        pers_df[["Scaling", "RMSE_test", "MAE_test", "R2_test"]].to_string(index=False)
    )

    print("\n" + "-" * 60)
    print("LINEAR REGRESSION MODEL (sorted by RMSE_test):")
    lr_df = results_df[results_df["Model"] == "Linear Regression"].sort_values(
        "RMSE_test"
    )
    print(lr_df[["Scaling", "RMSE_test", "MAE_test", "R2_test"]].to_string(index=False))

    # Best overall
    best_pers = pers_df.iloc[0]
    best_lr = lr_df.iloc[0]

    print("\n" + "=" * 60)
    print("CONCLUSIONS:")
    print("-" * 60)
    print(
        f"Best Persistence: {best_pers['Scaling']} (RMSE={best_pers['RMSE_test']:.4f})"
    )
    print(f"Best Linear Reg:  {best_lr['Scaling']} (RMSE={best_lr['RMSE_test']:.4f})")
    print("=" * 60)

    # Comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(3 * HEIGHT, 2 * HEIGHT))
    fig.suptitle(f"{DATASET_TAG} - Scaling Results Comparison")

    metrics_to_plot = ["RMSE_test", "MAE_test", "R2_test", "MAPE_test"]
    titles = ["RMSE (Test)", "MAE (Test)", "R² (Test)", "MAPE (Test)"]

    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        pivot = results_df.pivot(index="Scaling", columns="Model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=45)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{images_dir}/{DATASET_TAG}_scaling_comparison_chart.png", dpi=150)
    plt.close()

    print(f"\nAll images saved to: {images_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
