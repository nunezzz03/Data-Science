"""
Lab 6 - Time Series Forecasting: Exponential Smoothing
Datasets: TrafficTwoMonth.csv, Economic Indicators

This script implements Exponential Smoothing forecasting models:
1. Simple Exponential Smoothing (SES) - for stationary data
2. Holt's Linear Trend - for data with trend
3. Holt-Winters Seasonal - for data with trend and seasonality

For each model, we study the hyperparameters to find the best configuration.
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import read_csv, Series
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from math import sqrt

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORECASTING_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SCIENCE_ROOT = os.path.dirname(FORECASTING_ROOT)
sys.path.insert(0, DATA_SCIENCE_ROOT)
sys.path.insert(0, os.path.join(DATA_SCIENCE_ROOT, "utils"))

from utils.dslabs_functions import (
    plot_line_chart,
    HEIGHT,
    FORECAST_MEASURES,
    plot_forecasting_series,
    plot_forecasting_eval,
)

# Configuration
TRAIN_PCT = 0.90
DELTA_IMPROVE = 0.001
MEASURE = "R2"

# Dataset configurations
DATASETS = {
    "traffic": {
        "path": f"{DATA_SCIENCE_ROOT}/data/prepared/traffic_smoothed.csv",
        "target": "Total",
        "date_column": "datetime",
        "file_tag": "traffic",
        "seasonal_period": 24,
    },
    "economic": {
        "path": f"{DATA_SCIENCE_ROOT}/data/prepared/economic_usa_smoothed.csv",
        "target": "Inflation Rate (%)",
        "date_column": "Date",
        "file_tag": "economic",
        "seasonal_period": 4,
    },
}

# Output directories
IMAGES_DIR = f"{FORECASTING_ROOT}/outputs/models"
RESULTS_DIR = f"{FORECASTING_ROOT}/outputs/models"


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_traffic_data():
    """Load traffic data using the prepared datetime index."""
    config = DATASETS["traffic"]
    data = read_csv(
        config["path"], sep=",", decimal=".", parse_dates=[config["date_column"]]
    )

    data = data.set_index(config["date_column"])
    data = data.sort_index()

    return data[config["target"]]


def load_economic_data():
    """Load economic data."""
    config = DATASETS["economic"]
    data = read_csv(
        config["path"], sep=",", decimal=".", parse_dates=[config["date_column"]]
    )

    data = data.set_index(config["date_column"])
    data = data.sort_index()

    return data[config["target"]]


def series_train_test_split(series: Series, trn_pct: float = 0.90):
    """Split a time series into train and test sets."""
    trn_size = int(len(series) * trn_pct)
    train = series.iloc[:trn_size]
    test = series.iloc[trn_size:]
    return train, test


def simple_exponential_smoothing_study(
    train: Series, test: Series, file_tag: str, measure: str = "R2"
) -> tuple:
    """
    Study Simple Exponential Smoothing (SES) with different alpha values.
    SES is suitable for data without trend or seasonality.

    Args:
        train: Training series
        test: Test series
        file_tag: Dataset identifier for saving plots
        measure: Evaluation metric (R2, MSE, MAE, MAPE)

    Returns:
        best_model, best_params dict
    """
    alpha_values = [i / 10 for i in range(1, 10)]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params = {
        "name": "Simple Exponential Smoothing",
        "metric": measure,
        "params": {},
    }
    best_performance = -100000 if measure in ["R2"] else float("inf")

    yvalues = []
    for alpha in alpha_values:
        try:
            tool = SimpleExpSmoothing(train, initialization_method="estimated")
            model = tool.fit(smoothing_level=alpha, optimized=False)
            prd_tst = model.forecast(steps=len(test))

            eval_score = FORECAST_MEASURES[measure](test, prd_tst)

            # For R2, higher is better; for MSE/MAE, lower is better
            if measure in ["R2"]:
                if (
                    eval_score > best_performance
                    and abs(eval_score - best_performance) > DELTA_IMPROVE
                ):
                    best_performance = eval_score
                    best_params["params"] = {"alpha": alpha}
                    best_model = model
            else:
                if (
                    eval_score < best_performance
                    and abs(eval_score - best_performance) > DELTA_IMPROVE
                ):
                    best_performance = eval_score
                    best_params["params"] = {"alpha": alpha}
                    best_model = model

            yvalues.append(eval_score)
        except Exception as e:
            print(f"  Warning: SES with alpha={alpha} failed: {e}")
            yvalues.append(np.nan)

    # Plot the study
    plt.figure(figsize=(8, HEIGHT))
    plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing - {file_tag} ({measure})",
        xlabel="alpha (α)",
        ylabel=measure,
        percentage=flag,
    )
    plt.tight_layout()
    plt.savefig(
        f"{IMAGES_DIR}/{file_tag}_exponential_smoothing_{measure}_study.png", dpi=150
    )
    plt.close()

    if best_params["params"]:
        print(
            f"  SES best with alpha={best_params['params']['alpha']:.1f} -> {measure}={best_performance:.4f}"
        )
    else:
        print(f"  SES: No valid model found")

    return best_model, best_params


def holt_study(
    train: Series, test: Series, file_tag: str, measure: str = "R2"
) -> tuple:
    """
    Study Holt's Linear Trend method with different alpha and beta values.
    Holt's method is suitable for data with trend but no seasonality.

    Args:
        train: Training series
        test: Test series
        file_tag: Dataset identifier for saving plots
        measure: Evaluation metric

    Returns:
        best_model, best_params dict
    """
    alpha_values = [i / 10 for i in range(1, 10)]
    beta_values = [i / 10 for i in range(1, 10)]

    best_model = None
    best_params = {"name": "Holt Linear Trend", "metric": measure, "params": {}}
    best_performance = -100000 if measure in ["R2"] else float("inf")

    # Store results for heatmap
    results_matrix = np.zeros((len(alpha_values), len(beta_values)))

    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            try:
                model = ExponentialSmoothing(
                    train, trend="add", seasonal=None, initialization_method="estimated"
                ).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
                prd_tst = model.forecast(steps=len(test))

                eval_score = FORECAST_MEASURES[measure](test, prd_tst)
                results_matrix[i, j] = eval_score

                if measure in ["R2"]:
                    if (
                        eval_score > best_performance
                        and abs(eval_score - best_performance) > DELTA_IMPROVE
                    ):
                        best_performance = eval_score
                        best_params["params"] = {"alpha": alpha, "beta": beta}
                        best_model = model
                else:
                    if (
                        eval_score < best_performance
                        and abs(eval_score - best_performance) > DELTA_IMPROVE
                    ):
                        best_performance = eval_score
                        best_params["params"] = {"alpha": alpha, "beta": beta}
                        best_model = model
            except Exception as e:
                results_matrix[i, j] = np.nan

    # Plot heatmap of results
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        results_matrix, cmap="RdYlGn" if measure == "R2" else "RdYlGn_r", aspect="auto"
    )

    ax.set_xticks(range(len(beta_values)))
    ax.set_yticks(range(len(alpha_values)))
    ax.set_xticklabels([f"{b:.1f}" for b in beta_values])
    ax.set_yticklabels([f"{a:.1f}" for a in alpha_values])
    ax.set_xlabel("Beta (β) - Trend Smoothing")
    ax.set_ylabel("Alpha (α) - Level Smoothing")
    ax.set_title(f"Holt Linear Trend - {file_tag} ({measure})")

    plt.colorbar(im, ax=ax, label=measure)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/{file_tag}_holt_{measure}_study.png", dpi=150)
    plt.close()

    if best_params["params"]:
        print(
            f"  Holt best with alpha={best_params['params']['alpha']:.1f}, beta={best_params['params']['beta']:.1f} -> {measure}={best_performance:.4f}"
        )
    else:
        print(f"  Holt: No valid model found")

    return best_model, best_params


def holt_winters_study(
    train: Series,
    test: Series,
    file_tag: str,
    seasonal_period: int,
    measure: str = "R2",
) -> tuple:
    """
    Study Holt-Winters method with different parameters.
    Holt-Winters is suitable for data with trend and seasonality.

    Args:
        train: Training series
        test: Test series
        file_tag: Dataset identifier for saving plots
        seasonal_period: Number of observations per seasonal cycle
        measure: Evaluation metric

    Returns:
        best_model, best_params dict
    """
    alpha_values = [0.2, 0.4, 0.6, 0.8]  # Reduced for speed
    beta_values = [0.1, 0.3, 0.5]
    gamma_values = [0.1, 0.3, 0.5]
    seasonal_types = ["add", "mul"]

    best_model = None
    best_params = {"name": "Holt-Winters", "metric": measure, "params": {}}
    best_performance = -100000 if measure in ["R2"] else float("inf")

    results = []

    # Check if we have enough data for seasonal decomposition
    if len(train) < 2 * seasonal_period:
        print(
            f"  Warning: Not enough data for seasonal period {seasonal_period}. Skipping Holt-Winters."
        )
        return None, best_params

    for seasonal in seasonal_types:
        for alpha in alpha_values:
            for beta in beta_values:
                for gamma in gamma_values:
                    try:
                        model = ExponentialSmoothing(
                            train,
                            trend="add",
                            seasonal=seasonal,
                            seasonal_periods=seasonal_period,
                            initialization_method="estimated",
                        ).fit(
                            smoothing_level=alpha,
                            smoothing_trend=beta,
                            smoothing_seasonal=gamma,
                            optimized=False,
                        )
                        prd_tst = model.forecast(steps=len(test))

                        eval_score = FORECAST_MEASURES[measure](test, prd_tst)

                        results.append(
                            {
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "seasonal": seasonal,
                                measure: eval_score,
                            }
                        )

                        if measure in ["R2"]:
                            if (
                                eval_score > best_performance
                                and abs(eval_score - best_performance) > DELTA_IMPROVE
                            ):
                                best_performance = eval_score
                                best_params["params"] = {
                                    "alpha": alpha,
                                    "beta": beta,
                                    "gamma": gamma,
                                    "seasonal": seasonal,
                                    "period": seasonal_period,
                                }
                                best_model = model
                        else:
                            if (
                                eval_score < best_performance
                                and abs(eval_score - best_performance) > DELTA_IMPROVE
                            ):
                                best_performance = eval_score
                                best_params["params"] = {
                                    "alpha": alpha,
                                    "beta": beta,
                                    "gamma": gamma,
                                    "seasonal": seasonal,
                                    "period": seasonal_period,
                                }
                                best_model = model
                    except Exception as e:
                        pass  # Skip invalid combinations

    # Plot results - grouped by seasonal type
    if results:
        df_results = pd.DataFrame(results)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, seasonal in enumerate(seasonal_types):
            subset = df_results[df_results["seasonal"] == seasonal]
            if not subset.empty:
                # Group by alpha, show mean performance
                grouped = subset.groupby("alpha")[measure].mean()
                axes[idx].bar(range(len(grouped)), grouped.values)
                axes[idx].set_xticks(range(len(grouped)))
                axes[idx].set_xticklabels([f"α={a:.1f}" for a in grouped.index])
                axes[idx].set_ylabel(measure)
                axes[idx].set_title(f"Holt-Winters ({seasonal} seasonal)")

        plt.suptitle(f"Holt-Winters Study - {file_tag}", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{IMAGES_DIR}/{file_tag}_holtwinters_{measure}_study.png", dpi=150)
        plt.close()

    if best_params["params"]:
        print(
            f"  Holt-Winters best with alpha={best_params['params']['alpha']:.1f}, "
            f"beta={best_params['params']['beta']:.1f}, gamma={best_params['params']['gamma']:.1f}, "
            f"seasonal={best_params['params']['seasonal']} -> {measure}={best_performance:.4f}"
        )
    else:
        print(f"  Holt-Winters: No valid model found")

    return best_model, best_params


def plot_best_model_evaluation(
    train: Series, test: Series, model, params: dict, file_tag: str, model_name: str
):
    """Plot the best model's predictions and evaluation metrics."""
    if model is None:
        print(f"  No model to evaluate for {model_name}")
        return

    # Generate predictions
    prd_trn = model.fittedvalues
    prd_tst = model.forecast(steps=len(test))

    # Align indices
    prd_trn = pd.Series(prd_trn.values, index=train.index)
    prd_tst = pd.Series(prd_tst.values, index=test.index)

    # Plot evaluation metrics
    plt.figure(figsize=(10, HEIGHT))
    plot_forecasting_eval(
        train,
        test,
        prd_trn,
        prd_tst,
        title=f"{file_tag} - {model_name} alpha = {params['params'].get('alpha', 'N/A')}",
    )
    plt.tight_layout()
    plt.savefig(
        f"{IMAGES_DIR}/{file_tag}_{model_name.lower().replace(' ', '_').replace('-', '_')}_eval.png",
        dpi=150,
    )
    plt.close()

    # Plot forecast visualization
    plt.figure(figsize=(12, HEIGHT))
    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - {model_name} Forecast",
        xlabel="Time",
        ylabel="Value",
    )
    plt.tight_layout()
    plt.savefig(
        f"{IMAGES_DIR}/{file_tag}_{model_name.lower().replace(' ', '_').replace('-', '_')}_forecast.png",
        dpi=150,
    )
    plt.close()


def run_exponential_smoothing_analysis(dataset_name: str):
    """
    Run complete Exponential Smoothing analysis for a dataset.

    Args:
        dataset_name: 'traffic' or 'economic'
    """
    print(f"\n{'='*60}")
    print(f"Exponential Smoothing Analysis: {dataset_name.upper()}")
    print(f"{'='*60}")

    config = DATASETS[dataset_name]
    file_tag = config["file_tag"]
    seasonal_period = config["seasonal_period"]

    # Load data
    if dataset_name == "traffic":
        series = load_traffic_data()
    else:
        series = load_economic_data()

    print(f"\nDataset: {len(series)} observations")
    print(f"Target: {config['target']}")
    print(f"Seasonal period: {seasonal_period}")

    # Train-test split
    train, test = series_train_test_split(series, TRAIN_PCT)
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # Store results
    all_results = []

    # 1. Simple Exponential Smoothing
    print(f"\n--- Simple Exponential Smoothing ---")
    ses_model, ses_params = simple_exponential_smoothing_study(
        train, test, file_tag, MEASURE
    )
    if ses_model:
        plot_best_model_evaluation(
            train, test, ses_model, ses_params, file_tag, "Exponential Smoothing"
        )
        prd_tst = ses_model.forecast(steps=len(test))
        all_results.append(
            {
                "Model": "Simple Exponential Smoothing",
                "Parameters": str(ses_params["params"]),
                "R2": FORECAST_MEASURES["R2"](test, prd_tst),
                "MSE": FORECAST_MEASURES["MSE"](test, prd_tst),
                "MAE": FORECAST_MEASURES["MAE"](test, prd_tst),
                "RMSE": sqrt(FORECAST_MEASURES["MSE"](test, prd_tst)),
            }
        )

    # 2. Holt's Linear Trend
    print(f"\n--- Holt's Linear Trend ---")
    holt_model, holt_params = holt_study(train, test, file_tag, MEASURE)
    if holt_model:
        plot_best_model_evaluation(
            train, test, holt_model, holt_params, file_tag, "Holt"
        )
        prd_tst = holt_model.forecast(steps=len(test))
        all_results.append(
            {
                "Model": "Holt Linear Trend",
                "Parameters": str(holt_params["params"]),
                "R2": FORECAST_MEASURES["R2"](test, prd_tst),
                "MSE": FORECAST_MEASURES["MSE"](test, prd_tst),
                "MAE": FORECAST_MEASURES["MAE"](test, prd_tst),
                "RMSE": sqrt(FORECAST_MEASURES["MSE"](test, prd_tst)),
            }
        )

    # 3. Holt-Winters (if enough data)
    print(f"\n--- Holt-Winters Seasonal ---")
    hw_model, hw_params = holt_winters_study(
        train, test, file_tag, seasonal_period, MEASURE
    )
    if hw_model:
        plot_best_model_evaluation(
            train, test, hw_model, hw_params, file_tag, "Holt_Winters"
        )
        prd_tst = hw_model.forecast(steps=len(test))
        all_results.append(
            {
                "Model": "Holt-Winters",
                "Parameters": str(hw_params["params"]),
                "R2": FORECAST_MEASURES["R2"](test, prd_tst),
                "MSE": FORECAST_MEASURES["MSE"](test, prd_tst),
                "MAE": FORECAST_MEASURES["MAE"](test, prd_tst),
                "RMSE": sqrt(FORECAST_MEASURES["MSE"](test, prd_tst)),
            }
        )

    # Save results
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(
            f"{RESULTS_DIR}/{file_tag}_exponential_smoothing_results.csv", index=False
        )
        print(f"\n--- Results Summary ---")
        print(df_results.to_string(index=False))

    return all_results


def create_comparison_plot(traffic_results: list, economic_results: list):
    """Create a comparison plot of all models across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = [("Traffic", traffic_results), ("Economic", economic_results)]

    for idx, (name, results) in enumerate(datasets):
        if not results:
            continue

        models = [r["Model"] for r in results]
        r2_values = [r["R2"] for r in results]

        bars = axes[idx].bar(
            range(len(models)),
            r2_values,
            color=["#3498db", "#e74c3c", "#2ecc71"][: len(models)],
        )
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
        axes[idx].set_ylabel("R²")
        axes[idx].set_title(f"{name} Dataset")
        axes[idx].axhline(y=0, color="black", linestyle="--", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, r2_values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("Exponential Smoothing Models Comparison (R²)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/exponential_smoothing_comparison.png", dpi=150)
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("LAB 6 - EXPONENTIAL SMOOTHING FORECASTING")
    print("=" * 70)

    ensure_directories()

    # Run analysis for both datasets
    traffic_results = run_exponential_smoothing_analysis("traffic")
    economic_results = run_exponential_smoothing_analysis("economic")

    # Create comparison plot
    create_comparison_plot(traffic_results, economic_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
