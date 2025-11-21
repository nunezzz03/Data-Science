import pandas as pd
from pandas import read_csv, DataFrame
from matplotlib.pyplot import subplots, savefig, figure, close
from seaborn import heatmap
import matplotlib.pyplot as plt
import os
from itertools import combinations

# Get paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # One level up from data_profiling/

# Configuration
HEIGHT = 4

# Create output directories
os.makedirs(os.path.join(PROJECT_ROOT, "images", "correlation_matrix"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "images", "sparsity_study"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "images", "sparsity_per_class"), exist_ok=True)

# Datasets configuration
datasets = [
    {
        "file_tag": "accidents",
        "filename": os.path.join(PROJECT_ROOT, "data", "raw", "traffic_accidents.csv"),
        "target": "crash_type",
    },
    {
        "file_tag": "flights",
        "filename": os.path.join(
            PROJECT_ROOT, "data", "raw", "Combined_Flights_2022.csv"
        ),
        "target": "ArrDel15",
    },
]


def get_variable_types(df: DataFrame) -> dict:
    """Separates variables into numeric, symbolic, and binary."""
    numeric = []
    symbolic = []
    binary = []

    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[col]):
            if unique_count == 2:
                binary.append(col)
            else:
                symbolic.append(col)

    return {"numeric": numeric, "symbolic": symbolic, "binary": binary}


def plot_multi_scatters_chart(
    data: DataFrame, var1: str, var2: str, target: str = None, ax=None
):
    """Plot scatter chart, optionally colored by target class."""
    if ax is None:
        ax = plt.gca()

    if target and target in data.columns:
        # Color by class
        for cls in data[target].unique():
            mask = data[target] == cls
            ax.scatter(
                data.loc[mask, var1],
                data.loc[mask, var2],
                label=str(cls),
                alpha=0.6,
                s=10,
            )
        ax.legend(fontsize=6)
    else:
        # No class discrimination
        ax.scatter(data[var1], data[var2], alpha=0.6, s=10)

    ax.set_xlabel(var1, fontsize=8)
    ax.set_ylabel(var2, fontsize=8)
    ax.tick_params(labelsize=6)


def process_correlation_matrix(dataset):
    """Generate correlation matrix for numeric variables."""
    file_tag = dataset["file_tag"]
    filename = dataset["filename"]
    target = dataset["target"]

    print(f"\n{'='*60}")
    print(f"CORRELATION MATRIX: {file_tag.upper()}")
    print("=" * 60)

    # Load and clean data
    data: DataFrame = read_csv(filename, na_values="")
    print(f"   Loaded {len(data)} records")

    data = data.dropna()
    print(f"   After dropping NA: {len(data)} records")

    # Get numeric variables
    variables_types = get_variable_types(data)
    numeric = variables_types["numeric"]

    if len(numeric) < 2:
        print(
            f"   ⚠️ Need at least 2 numeric variables, found {len(numeric)}. Skipping."
        )
        return

    print(f"   Found {len(numeric)} numeric variables")

    # Generate correlation matrix
    print(f"   Generating correlation matrix...")
    corr_mtx: DataFrame = data[numeric].corr().abs()

    figure(figsize=(12, 10))
    heatmap(
        corr_mtx,
        xticklabels=numeric,
        yticklabels=numeric,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    output_path = os.path.join(
        PROJECT_ROOT,
        "images",
        "correlation_matrix",
        f"{file_tag}_correlation_analysis.png",
    )
    savefig(output_path, bbox_inches="tight", dpi=100)
    close()
    print(f"   ✅ Saved correlation matrix: {output_path}")


def process_sparsity_study(dataset):
    """Generate sparsity study (scatter plots without class discrimination)."""
    file_tag = dataset["file_tag"]
    filename = dataset["filename"]

    print(f"\n{'='*60}")
    print(f"SPARSITY STUDY: {file_tag.upper()}")
    print("=" * 60)

    # Load and clean data
    data: DataFrame = read_csv(filename, na_values="")
    print(f"   Loaded {len(data)} records")

    data = data.dropna()
    print(f"   After dropping NA: {len(data)} records")

    # Get only numeric columns
    numeric_data = data.select_dtypes(include=["number"])
    all_vars = numeric_data.columns.to_list()

    if len(all_vars) < 2:
        print(
            f"   ⚠️ Need at least 2 numeric variables, found {len(all_vars)}. Skipping."
        )
        return

    print(f"   Found {len(all_vars)} numeric variables")

    # Generate sparsity study
    print(f"   Generating sparsity scatter plots...")
    n = len(all_vars)
    fig, axs = subplots(
        n - 1, n - 1, figsize=((n - 1) * HEIGHT, (n - 1) * HEIGHT), squeeze=False
    )

    for i in range(len(all_vars)):
        var1 = all_vars[i]
        for j in range(i + 1, len(all_vars)):
            var2 = all_vars[j]
            plot_multi_scatters_chart(numeric_data, var1, var2, ax=axs[i, j - 1])

    output_path = os.path.join(
        PROJECT_ROOT, "images", "sparsity_study", f"{file_tag}_sparsity_study.png"
    )
    savefig(output_path, bbox_inches="tight", dpi=100)
    close()
    print(f"   ✅ Saved sparsity study: {output_path}")


def process_sparsity_per_class(dataset):
    """Generate sparsity study per class (scatter plots colored by target)."""
    file_tag = dataset["file_tag"]
    filename = dataset["filename"]
    target = dataset["target"]

    print(f"\n{'='*60}")
    print(f"SPARSITY PER CLASS: {file_tag.upper()}")
    print("=" * 60)

    # Load and clean data
    data: DataFrame = read_csv(filename, na_values="")
    print(f"   Loaded {len(data)} records")

    data = data.dropna()
    print(f"   After dropping NA: {len(data)} records")

    # Check target exists
    if target not in data.columns:
        print(f"   ⚠️ Target '{target}' not found. Skipping per-class analysis.")
        return

    # Get numeric variables (excluding target)
    numeric_data = data.select_dtypes(include=["number"])
    vars_no_target = [col for col in numeric_data.columns if col != target]

    if len(vars_no_target) < 2:
        print(
            f"   ⚠️ Need at least 2 numeric variables, found {len(vars_no_target)}. Skipping."
        )
        return

    print(f"   Found {len(vars_no_target)} numeric variables (excluding target)")

    # For large variable sets (like flights), split into chunks to avoid RAM issues
    if len(vars_no_target) > 25:
        # Split into chunks of ~20 variables each
        chunk_size = 20
        num_chunks = (len(vars_no_target) + chunk_size - 1) // chunk_size
        print(
            f"   ⚠️ Large variable set detected. Splitting into {num_chunks} chunks of ~{chunk_size} variables"
        )

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(vars_no_target))
            chunk_vars = vars_no_target[start_idx:end_idx]

            if len(chunk_vars) < 2:
                continue

            print(
                f"   Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_vars)} variables)..."
            )

            n = len(chunk_vars)
            fig, axs = subplots(
                n - 1,
                n - 1,
                figsize=((n - 1) * HEIGHT, (n - 1) * HEIGHT),
                squeeze=False,
            )

            for i in range(len(chunk_vars)):
                var1 = chunk_vars[i]
                for j in range(i + 1, len(chunk_vars)):
                    var2 = chunk_vars[j]
                    plot_multi_scatters_chart(
                        data, var1, var2, target, ax=axs[i, j - 1]
                    )

            output_path = os.path.join(
                PROJECT_ROOT,
                "images",
                "sparsity_per_class",
                f"{file_tag}_sparsity_per_class_study_chunk{chunk_idx + 1}.png",
            )
            savefig(output_path, bbox_inches="tight", dpi=100)
            close()
            print(f"   ✅ Saved chunk {chunk_idx + 1}: {output_path}")
    else:
        # Small enough to do in one plot
        print(f"   Generating single sparsity per class plot...")
        n = len(vars_no_target)
        fig, axs = subplots(
            n - 1, n - 1, figsize=((n - 1) * HEIGHT, (n - 1) * HEIGHT), squeeze=False
        )

        for i in range(len(vars_no_target)):
            var1 = vars_no_target[i]
            for j in range(i + 1, len(vars_no_target)):
                var2 = vars_no_target[j]
                plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])

        output_path = os.path.join(
            PROJECT_ROOT,
            "images",
            "sparsity_per_class",
            f"{file_tag}_sparsity_per_class_study.png",
        )
        savefig(output_path, bbox_inches="tight", dpi=100)
        close()
        print(f"   ✅ Saved per-class study: {output_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPARSITY & CORRELATION PROFILING")
    print("=" * 60)

    for dataset in datasets:
        try:
            # 1. Correlation Matrix
            process_correlation_matrix(dataset)

            # 2. Sparsity Study (no class discrimination)
            process_sparsity_study(dataset)

            # 3. Sparsity Per Class (with class discrimination)
            process_sparsity_per_class(dataset)

        except FileNotFoundError:
            print(f"\n   ❌ Error: File not found for {dataset['file_tag']}")
            print(f"      Path: {dataset['filename']}")
        except Exception as e:
            print(f"\n   ❌ Error processing {dataset['file_tag']}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ SPARSITY & CORRELATION PROFILING COMPLETE!")
    print("=" * 60 + "\n")
