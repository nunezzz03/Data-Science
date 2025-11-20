import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, lognorm
import os
import math
import warnings

# Suppress runtime warnings related to log fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # One level up from data_profiling/

# --- Configuration & Setup ---
HEIGHT = 5
NR_STDEV = 2
IQR_FACTOR = 1.5

os.makedirs(os.path.join(PROJECT_ROOT, "images", "distribution"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

DATASETS = [
    {
        "name": "Accidents",
        "file": os.path.join(PROJECT_ROOT, "data", "raw", "traffic_accidents.csv"),
        "target": "crash_type",
    },
    {
        "name": "Flights",
        "file": os.path.join(PROJECT_ROOT, "data", "raw", "Combined_Flights_2022.csv"),
        "target": "ArrDel15",
    },
]

# --- Helper Functions (Standard implementations for profiling) ---


def define_grid(n_vars):
    """Calculates optimal subplot grid size."""
    rows = int(math.ceil(math.sqrt(n_vars)))
    cols = int(math.ceil(n_vars / rows))
    return rows, cols


def get_variable_types(df, target_name):
    """Separates variables into numeric, symbolic, and binary."""
    numeric = []
    symbolic = []
    binary = []

    cols_to_profile = [col for col in df.columns if col != target_name]

    for col in cols_to_profile:
        dtype = df[col].dtype
        unique_count = df[col].nunique()

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[col]):
            if unique_count == 2:
                binary.append(col)
            elif unique_count > 50 and pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            else:
                symbolic.append(col)

    return {"numeric": numeric, "symbolic": symbolic, "binary": binary}


def plot_multibar_chart(labels, values, title, xlabel, ylabel, path):
    """Generates a multibar chart for outlier analysis."""
    fig, ax = plt.subplots(figsize=(12, HEIGHT))
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, values["stdev"], width, label="StDev Criterion")
    rects2 = ax.bar(x + width / 2, values["iqr"], width, label="IQR Criterion")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def determine_outlier_thresholds_for_var(summary5, std_based=True, threshold=NR_STDEV):
    """Calculates outlier thresholds (top/bottom) based on criteria."""
    if std_based:
        std = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr
    return top, bottom


def count_outliers(data, numeric, nrstdev=NR_STDEV, iqrfactor=IQR_FACTOR):
    """Counts outliers in numeric variables using IQR and StDev criteria."""
    outliers_iqr = []
    outliers_stdev = []

    # We describe the full DataFrame, but will access only numeric columns
    summary5 = data.describe()

    for var in numeric:
        # Need to ensure the summary has the column, especially if it was empty/all NaN
        if var not in summary5.columns:
            outliers_stdev.append(0)
            outliers_iqr.append(0)
            continue

        # Get the non-missing, numerically cast values for counting
        var_values = data[var].dropna().astype(float)

        # StDev Criteria
        top_s, bottom_s = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        count_s = (var_values > top_s).sum() + (var_values < bottom_s).sum()
        outliers_stdev.append(count_s)

        # IQR Criteria
        top_i, bottom_i = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        count_i = (var_values > top_i).sum() + (var_values < bottom_i).sum()
        outliers_iqr.append(count_i)

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}


def compute_known_distributions(x_values):
    """Fits Normal, Exponential, and LogNormal distributions to data."""
    distributions = {}
    positive_values = np.array([v for v in x_values if v > 0])

    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)

    # Exponential
    loc, scale = expon.fit(x_values)
    if scale > 0:
        distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)

    # LogNorm (Fit only on positive values)
    if len(positive_values) > 1:
        sigma_l, loc_l, scale_l = lognorm.fit(positive_values)
        try:
            log_pdf = lognorm.pdf(x_values, sigma_l, loc_l, scale_l)
            distributions["LogNor(%.1f,%.2f)" % (math.log(scale_l), sigma_l)] = log_pdf
        except Exception:
            pass

    return distributions


def histogram_with_distributions(ax, series, var):
    """Plots a histogram with fitted PDF curves."""
    values = series.dropna().astype(float).sort_values().to_list()
    if not values:
        return

    ax.hist(values, bins="auto", density=True, label="Histogram")

    distributions = compute_known_distributions(values)

    for name, pdf in distributions.items():
        x_sorted = series.dropna().astype(float).sort_values().to_list()
        ax.plot(x_sorted, pdf, label=name)

    ax.set_title(f"Best fit for {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Probability Density")
    ax.legend()


# --- Main Profiling Loop ---
for ds in DATASETS:
    file_tag = ds["name"].lower()
    target_var = ds["target"]

    try:
        data = pd.read_csv(ds["file"], na_values="")

        if ds["name"] == "Accidents":
            data = data[
                [
                    "crash_type",
                    "num_units",
                    "injuries_total",
                    "injuries_fatal",
                    "injuries_incapacitating",
                    "injuries_non_incapacitating",
                    "injuries_reported_not_evident",
                    "injuries_no_indication",
                    "crash_hour",
                    "crash_day_of_week",
                    "crash_month",
                    "traffic_control_device",
                    "weather_condition",
                    "lighting_condition",
                    "first_crash_type",
                    "trafficway_type",
                    "alignment",
                    "roadway_surface_cond",
                    "road_defect",
                    "intersection_related_i",
                ]
            ]

        print(
            f"\n========================================================\n  STARTING DISTRIBUTION PROFILING: {ds['name']} ({len(data)} records)\n========================================================"
        )

    except FileNotFoundError:
        print(
            f"\n  ❌ Skipping {ds['name']}: File {ds['file']} not found. Please ensure both files are accessible."
        )
        continue
    except Exception as e:
        print(f"\n  ❌ Skipping {ds['name']}: Error loading file: {e}")
        continue

    # 1. Identify Variable Types
    variables_types = get_variable_types(data, target_var)
    numeric = variables_types["numeric"]
    symbolic = variables_types["symbolic"] + variables_types["binary"]

    if "crash_date" in symbolic:
        symbolic.remove("crash_date")
    if "Date" in symbolic:
        symbolic.remove("Date")

    print(f"  Variables: Numeric={len(numeric)}, Symbolic/Binary={len(symbolic)}")

    # 2. Summary Table (Five-Number Summary)
    summary_path = os.path.join(
        PROJECT_ROOT, "results", f"{file_tag}_distribution_summary.csv"
    )
    data.describe(include="all").T.to_csv(summary_path)
    print(f"  ✅ Summary table saved to: {summary_path}")

    # --- NUMERIC VARIABLES ANALYSIS ---
    if numeric:

        # A. Global Boxplot
        plt.figure(figsize=(10, 5))
        data[numeric].astype(float).boxplot(rot=45)
        plt.title(f"{ds['name']} Global Boxplot (Full Data)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PROJECT_ROOT, "images", "distribution", f"{file_tag}_global_boxplot.png"
            )
        )
        plt.close()
        print(f"  ✅ Global Boxplot saved.")

        # B. Single Boxplots
        rows, cols = define_grid(len(numeric))
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        for n, var in enumerate(numeric):
            i, j = divmod(n, cols)
            axs[i, j].set_title(f"Boxplot for {var}")

            # FIX: Explicitly cast to float to prevent numpy boolean subtract TypeError
            values = data[var].dropna().astype(float).values

            if len(values) > 0:
                axs[i, j].boxplot(values)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PROJECT_ROOT,
                "images",
                "distribution",
                f"{file_tag}_single_boxplots.png",
            )
        )
        plt.close()
        print(f"  ✅ Single Boxplots saved.")

        # C. Outliers Count
        outliers = count_outliers(data.copy(), numeric)
        plot_multibar_chart(
            numeric,
            outliers,
            title=f"{ds['name']} Nr of Outliers per Variable (StDev vs IQR)",
            xlabel="Variables",
            ylabel="Nr Outliers",
            path=os.path.join(
                PROJECT_ROOT,
                "images",
                "distribution",
                f"{file_tag}_outliers_comparison.png",
            ),
        )
        print(f"  ✅ Outliers Count chart saved.")

        # D. Single Histograms
        rows, cols = define_grid(len(numeric))
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        for n, var in enumerate(numeric):
            i, j = divmod(n, cols)
            axs[i, j].set_title(f"Histogram for {var}")
            axs[i, j].set_xlabel(var)
            axs[i, j].set_ylabel("Nr Records")
            axs[i, j].hist(data[var].dropna().astype(float).values, bins="auto")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PROJECT_ROOT,
                "images",
                "distribution",
                f"{file_tag}_single_histograms.png",
            )
        )
        plt.close()
        print(f"  ✅ Single Histograms saved.")

        # E. Histograms with PDF Fits
        rows, cols = define_grid(len(numeric))
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        for n, var in enumerate(numeric):
            i, j = divmod(n, cols)
            # FIX: histogram_with_distributions internally handles casting, but call must be safe
            histogram_with_distributions(axs[i, j], data[var], var)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PROJECT_ROOT,
                "images",
                "distribution",
                f"{file_tag}_histogram_numeric_distribution.png",
            )
        )
        plt.close()
        print(f"  ✅ Histograms with PDF Fits saved.")

    else:
        print("  ⚠️ No numeric variables to profile.")

    # --- SYMBOLIC/BINARY VARIABLES ANALYSIS (Histograms) ---
    if symbolic:
        rows, cols = define_grid(len(symbolic))
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        for n, var in enumerate(symbolic):
            i, j = divmod(n, cols)
            counts = data[var].value_counts(dropna=False).sort_index()
            axs[i, j].bar(counts.index.astype(str), counts.values)
            axs[i, j].set_title(f"Histogram for {var}")
            axs[i, j].set_xlabel(var)
            axs[i, j].set_ylabel("Nr Records")
            axs[i, j].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(f"images/distribution/{file_tag}_histograms_symbolic.png")
        plt.close()
        print(f"  ✅ Symbolic/Binary Histograms saved.")
    else:
        print("  ⚠️ No symbolic/binary variables to profile.")

    # --- CLASS DISTRIBUTION ---
    if target_var in data.columns:
        plt.figure(figsize=(6, 4))
        values = data[target_var].value_counts().sort_index()
        plt.bar(values.index.astype(str), values.values)
        plt.title(f"Target distribution (target={target_var})")
        plt.ylabel("Nr Records")
        plt.xlabel("Class")
        plt.tight_layout()
        plt.savefig(f"images/distribution/{file_tag}_class_distribution.png")
        plt.close()
        print(f"  ✅ Class Distribution saved.")

    print("========================================================\n")
