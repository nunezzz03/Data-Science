import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, figure, close
from seaborn import heatmap
from dslabs_functions import HEIGHT, plot_multi_scatters_chart, get_variable_types
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent crashes

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Configuration for both datasets
datasets = [
    {"file_tag": "accidents", "filename": os.path.join(WORKSPACE_ROOT, "data/raw/traffic_accidents_.csv"), "target": "crash_type"},
    {"file_tag": "flights", "filename": os.path.join(WORKSPACE_ROOT, "data/raw/Combined_Flights_2022.csv"), "target": "ArrDel15"},
]

OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "lab2/images/sparsity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR_CLASS = os.path.join(WORKSPACE_ROOT, "lab2/images/sparsity_per_class")
os.makedirs(OUTPUT_DIR_CLASS, exist_ok=True)

OUTPUT_DIR_CORRELATION = os.path.join(WORKSPACE_ROOT, "lab2/images/correlation")
os.makedirs(OUTPUT_DIR_CORRELATION, exist_ok=True)

print("\nSTARTING SPARSITY PROFILING")
print("=" * 50)

for dataset in datasets:
    file_tag = dataset["file_tag"]
    filename = dataset["filename"]
    target = dataset["target"]
    
    print(f"\nProcessing: {file_tag.upper()}")
    
    # Load and clean data
    data: DataFrame = read_csv(filename, na_values="")
    print(f"   Loaded {len(data)} records")
    
    data = data.dropna()
    print(f"   After dropping NA: {len(data)} records")
    
    # Sample if dataset is large (for performance)
    if len(data) > 5000:
        data = data.sample(n=5000, random_state=42)
        print(f"   Sampled to {len(data)} records for performance")
    
    # Get only numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    all_vars: list = numeric_data.columns.to_list()
    
    if len(all_vars) == 0:
        print(f"   No numeric variables found, skipping {file_tag}")
        continue
    
    print(f"   Found {len(all_vars)} numeric variables")
    
    # === PART 0: CORRELATION MATRIX ===
    print(f"   Generating correlation matrix...")
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]
    
    if len(numeric) > 1:
        corr_mtx: DataFrame = data[numeric].corr().abs()
        
        figure()
        heatmap(
            abs(corr_mtx),
            xticklabels=numeric,
            yticklabels=numeric,
            annot=False,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        output_path = f"{OUTPUT_DIR_CORRELATION}/{file_tag}_correlation_analysis.png"
        savefig(output_path)
        close()
        print(f"   ✓ Saved correlation matrix: {output_path}")
    
    # === PART 1: SPARSITY STUDY (without class discrimination) ===
    print(f"   Generating sparsity study...")
    if len(all_vars) > 1:
        n: int = len(all_vars)
        fig: Figure
        axs: ndarray
        fig, axs = subplots(n - 1, n - 1, figsize=((n - 1) * HEIGHT, (n - 1) * HEIGHT), squeeze=False)
        
        for i in range(len(all_vars)):
            var1: str = all_vars[i]
            for j in range(i + 1, len(all_vars)):
                var2: str = all_vars[j]
                plot_multi_scatters_chart(numeric_data, var1, var2, ax=axs[i, j - 1])
        
        output_path = f"{OUTPUT_DIR}/{file_tag}_sparsity_study.png"
        savefig(output_path)
        close()
        print(f"   ✓ Saved sparsity study: {output_path}")
    
    # === PART 2: SPARSITY PER CLASS (with class discrimination) ===
    if target not in data.columns:
        print(f"   Warning: Target '{target}' not found, skipping per-class analysis")
        continue
    
    print(f"   Generating sparsity per class study...")
    # Exclude target from variables for per-class analysis
    vars_no_target: list = [col for col in all_vars if col != target]
    
    if len(vars_no_target) > 1:
        n: int = len(vars_no_target)
        fig, axs = subplots(n - 1, n - 1, figsize=((n - 1) * HEIGHT, (n - 1) * HEIGHT), squeeze=False)
        
        for i in range(len(vars_no_target)):
            var1: str = vars_no_target[i]
            for j in range(i + 1, len(vars_no_target)):
                var2: str = vars_no_target[j]
                plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
        
        output_path = f"{OUTPUT_DIR_CLASS}/{file_tag}_sparsity_per_class_study.png"
        savefig(output_path)
        close()
        print(f"   ✓ Saved per-class study: {output_path}")
    
    print(f"   ✓ Completed {file_tag.upper()}")

print("\n" + "=" * 50)
print("SPARSITY PROFILING COMPLETE!")
print(f"Correlation matrices saved to: {OUTPUT_DIR_CORRELATION}")
print(f"Basic sparsity plots saved to: {OUTPUT_DIR}")
print(f"Per-class plots saved to: {OUTPUT_DIR_CLASS}")