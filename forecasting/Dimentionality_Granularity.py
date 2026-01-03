# -*- coding: utf-8 -*-
"""
Time Series Granularity Pipeline with Saved Figures, Multivariate Plots & 70/30 Train/Test Split
TrafficTwoMonth.csv (target = Total)
economic_indicators_dataset_2010_2023.csv (target = Inflation Rate (%))
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure, show, subplots
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pandas import DataFrame, Series

# Add utils to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT / "utils"))

from dslabs_functions import plot_line_chart, HEIGHT

# -----------------------------
# ===== Helper Functions ======
# -----------------------------


def convert_time_to_24h(time_str: str) -> int:
    """Convert '12:00:00 AM' style string to integer hour 0-23"""
    return pd.to_datetime(time_str, format="%I:%M:%S %p").hour


def encode_cyclical(values: Series, period: int) -> DataFrame:
    """Encode numeric values cyclically using sin/cos"""
    sin_col = np.sin(2 * np.pi * values / period)
    cos_col = np.cos(2 * np.pi * values / period)
    return pd.DataFrame({f"{values.name}_sin": sin_col, f"{values.name}_cos": cos_col})


def ts_aggregation_by(
    df: DataFrame | Series, group_size: int, agg_func: str = "mean"
) -> DataFrame | Series:
    """Aggregate by fixed-size window (used for traffic synthetic timeline)"""
    return df.groupby(df.index // group_size).agg(agg_func)


def ts_aggregation_by_period(
    df: DataFrame | Series, freq: str, agg_func: str = "mean"
) -> DataFrame | Series:
    """Aggregate by calendar frequency (used for economic dataset)"""
    return df.resample(freq).agg(agg_func)


def plot_line_and_save(x, y, xlabel, ylabel, title, save_path: Path):
    """Plot a univariate line chart and save figure"""
    fig = figure(figsize=(3 * HEIGHT, HEIGHT / 2))
    plot_line_chart(x, y, xlabel=xlabel, ylabel=ylabel, title=title)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    # show()


def plot_ts_multivariate_chart(df: DataFrame, title: str, save_path: Path) -> None:
    """Plot all numeric columns in df, one subplot per column, and save figure."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return  # nothing to plot

    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(
        len(numeric_cols), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(numeric_cols))
    )
    fig.suptitle(title)

    for i, col in enumerate(numeric_cols):
        plot_line_chart(
            df.index.to_list(),
            df[col].to_list(),
            ax=axs[i],
            xlabel=df.index.name,
            ylabel=col,
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    # show()


# -----------------------------
# ===== Script Folder Setup ===
# -----------------------------
SCRIPT_DIR = Path(
    __file__
).parent  # ensures CSVs are found in the same folder as script
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

# -----------------------------
# ======== TRAFFIC DATA =======
# -----------------------------
traffic_file = PROJECT_ROOT / "data" / "raw" / "TrafficTwoMonth.csv"
traffic_target = "Total"
traffic_df = pd.read_csv(traffic_file)

# Convert Time to 24h
traffic_df["Hour24"] = traffic_df["Time"].apply(convert_time_to_24h)

# Encode cyclic features
day_cyc = encode_cyclical(traffic_df["Date"], period=31)
hour_cyc = encode_cyclical(traffic_df["Hour24"], period=24)
traffic_df = pd.concat([traffic_df, day_cyc, hour_cyc], axis=1)

# Remove target NaNs
traffic_df = traffic_df.dropna(subset=[traffic_target])

# Remove categorical variables
cat_cols = traffic_df.select_dtypes(include=["object"]).columns.tolist()
traffic_df = traffic_df.drop(columns=cat_cols)

# Synthetic timeline as index
traffic_df = traffic_df.reset_index(drop=True)

# Output folder for traffic figures
traffic_out = OUTPUT_ROOT / "TrafficTwoMonth"
traffic_out.mkdir(exist_ok=True)

# Traffic granularities
traffic_grans = {"hourly": 1, "daily": 24, "weekly": 24 * 7}

for name, window in traffic_grans.items():
    agg = ts_aggregation_by(traffic_df, group_size=window, agg_func="mean")

    # Univariate plot
    fig_path_uni = traffic_out / f"traffic_{name}.png"
    plot_line_and_save(
        list(agg.index),
        agg[traffic_target].to_list(),
        xlabel="Synthetic timeline",
        ylabel=traffic_target,
        title=f"Traffic {name} aggregation ({traffic_target})",
        save_path=fig_path_uni,
    )

    # Multivariate plot
    fig_path_multi = traffic_out / f"traffic_{name}_multivariate.png"
    plot_ts_multivariate_chart(
        agg, title=f"Traffic {name} multivariate", save_path=fig_path_multi
    )

# Train/Test Split (70/30 chronological)
split_idx = int(len(traffic_df) * 0.7)
traffic_train = traffic_df.iloc[:split_idx]
traffic_test = traffic_df.iloc[split_idx:]

# -----------------------------
# ======= ECONOMIC DATA =======
# -----------------------------
econ_file = PROJECT_ROOT / "data" / "raw" / "economic_indicators_dataset_2010_2023.csv"
econ_target = "Inflation Rate (%)"
econ_df = pd.read_csv(econ_file, parse_dates=["Date"])
econ_df = econ_df.set_index("Date")

# Filter USA first
econ_df = econ_df[econ_df["Country"] == "USA"]

# Remove categorical variables
cat_cols = econ_df.select_dtypes(include=["object"]).columns.tolist()
econ_df = econ_df.drop(columns=cat_cols)

# Convert all remaining columns to numeric, coerce invalid entries to NaN
for col in econ_df.columns:
    econ_df[col] = pd.to_numeric(econ_df[col], errors="coerce")

# Drop rows with any NaN values
econ_df = econ_df.dropna()

# Output folder for economic figures
econ_out = OUTPUT_ROOT / "EconomicUSA"
econ_out.mkdir(exist_ok=True)

# Economic granularities
econ_grans = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}

for name, freq in econ_grans.items():
    agg = ts_aggregation_by_period(econ_df, freq=freq, agg_func="mean")

    # Univariate plot
    fig_path_uni = econ_out / f"economic_{name}.png"
    plot_line_and_save(
        list(agg.index),
        agg[econ_target].to_list(),
        xlabel="Date",
        ylabel=econ_target,
        title=f"Economic {name} aggregation ({econ_target})",
        save_path=fig_path_uni,
    )

    # Multivariate plot
    fig_path_multi = econ_out / f"economic_{name}_multivariate.png"
    plot_ts_multivariate_chart(
        agg, title=f"Economic {name} multivariate", save_path=fig_path_multi
    )

# Train/Test Split (70/30 chronological)
split_idx = int(len(econ_df) * 0.7)
econ_train = econ_df.iloc[:split_idx]
econ_test = econ_df.iloc[split_idx:]
