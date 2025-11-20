import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from pandas import Series, DataFrame, to_numeric, to_datetime

# Get paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # One level up from data_profiling/


def process_dataset(filename):
    print(f"--- Processing {filename} ---")
    try:
        # Read file from datasets folder (consistent path)
        filepath = os.path.join(PROJECT_ROOT, "data", "raw", filename)
        df = pd.read_csv(filepath)

        file_tag = os.path.splitext(filename)[0]
        # Dimensionality Summary Study
        dimensionality_summary(df, file_tag)

    except FileNotFoundError:
        print(f"❌ Could not find {filepath}")
    return


# FUNCTIONS FOR DIMENTIONALITY ANALYSIS


def plot_bar_chart(
    xvalues: list,
    yvalues: list,
    ax: Axes = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    percentage: bool = False,
) -> Axes:

    if ax is None:
        ax = plt.gca()

    # Labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # If percentage mode, convert x tick labels to percentages
    if percentage:
        ax.set_xticks(range(len(xvalues)))
        ax.set_xticklabels([f"{x:.2f}%" for x in xvalues])
        x_for_plot = range(len(xvalues))
        tick_labels = None  # avoid double-labeling
    else:
        x_for_plot = xvalues
        tick_labels = xvalues

    # Plot bars
    bars: BarContainer = ax.bar(
        x_for_plot,
        yvalues,
        tick_label=tick_labels,
    )

    # Add bar labels
    fmt = "%.2f" if percentage else "%.0f"
    ax.bar_label(bars, fmt=fmt, fontsize=12)
    # For readibility
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.margins(y=0.15)  # padding above bars
    plt.xticks(rotation=20)  # rotate for readability

    return ax


def records_x_variables(data, file_tag):
    plt.figure(figsize=(6, 6))
    values: dict[str, int] = {
        "nr records": data.shape[0],
        "nr variables": data.shape[1],
    }
    plot_bar_chart(
        list(values.keys()),
        list(values.values()),
        title="Nr of records vs nr variables",
    )
    img_path = os.path.join(
        PROJECT_ROOT, "images", "dimensionality", f"{file_tag}_records_variables.png"
    )
    plt.savefig(img_path, bbox_inches="tight")
    return


def missing_values_per_variable(data, file_tag):
    """
    Count missing values per column, considering both NaN and 'unknown' as missing.
    Plot a bar chart and save the figure.
    """
    mv: dict[str, int] = {}

    for var in data.columns:
        # Count NaN
        nr_na = data[var].isna().sum()
        # Count 'unknown' strings (case-insensitive)
        nr_unknown = (data[var].astype(str).str.lower() == "unknown").sum()

        # Total missing
        total_missing = nr_na + nr_unknown

        if total_missing > 0:
            mv[var] = total_missing

    # Plot
    plt.figure(figsize=(20, 8))
    plot_bar_chart(
        list(mv.keys()),
        list(mv.values()),
        title="Nr of missing values per variable",
        xlabel="Variables",
        ylabel="Nr missing values",
    )

    # Save figure
    img_path = os.path.join(
        PROJECT_ROOT, "images", "dimensionality", f"{file_tag}_mv.png"
    )
    plt.savefig(img_path, bbox_inches="tight")
    return


def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": [], "date": [], "symbolic": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                try:
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    variable_types["symbolic"].append(c)

    return variable_types


def plot_variable_type_counts(data, file_tag):
    # Get variable types
    variable_types: dict[str, list] = get_variable_types(data)

    # Count variables per type
    counts: dict[str, int] = {tp: len(cols) for tp, cols in variable_types.items()}

    # Plot
    plt.figure(figsize=(6, 8))
    plot_bar_chart(
        list(counts.keys()),
        list(counts.values()),
        title="Nr of variables per type",
        xlabel="Variable type",
        ylabel="Nr of variables",
    )

    img_path = os.path.join(
        PROJECT_ROOT, "images", "dimensionality", f"{file_tag}_variable_types.png"
    )
    plt.savefig(img_path, bbox_inches="tight")
    return


def dimensionality_summary(df, file_tag):
    """
    Generates summary bar charts for a DataFrame:
      - Number of records vs variables
      - Missing values per variable
      - Number of variables per type (numeric, binary, date, symbolic)

    Saves all figures to the `images/` folder with filenames prefixed by `file_tag`.

    Args:
        df: pandas DataFrame
        file_tag: string prefix for saved images
    """
    # Ensure images folder exists in project root
    os.makedirs(os.path.join(PROJECT_ROOT, "images", "dimensionality"), exist_ok=True)

    # 1️⃣ Records vs Variables
    records_x_variables(df, file_tag)

    # 2️⃣ Missing values per variable
    missing_values_per_variable(df, file_tag)

    # 3️⃣ Variables per type
    plot_variable_type_counts(df, file_tag)

    print("Dimentionality Summary Done - see images for results")
    return


# --- RUN LIST ---
# 1. Accidents - Keep as is (no issues found)
print("1️⃣ ACCIDENTS DATASET")
process_dataset("traffic_accidents.csv")

# 2. Flights - Remove ALL features only known after arrival (DATA LEAKAGE FIX)
print("\n2️⃣ FLIGHTS DATASET")

process_dataset("Combined_Flights_2022.csv")

print("\n✅ Dimentionality Profiling Done! Check Images.")
