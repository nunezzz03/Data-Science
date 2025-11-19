import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart

# Configuration for both datasets
datasets = [
    {"file_tag": "accidents", "filename": "data/raw/traffic_accidents.csv", "target": "crash_type"},
    {"file_tag": "flights", "filename": "data/raw/Combined_Flights_2022.csv", "target": "ArrDel15"},
]

# Create output directory
OUTPUT_DIR = "lab2/images/sparsity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR_CLASS = "lab2/images/sparsity_per_class"
os.makedirs(OUTPUT_DIR_CLASS, exist_ok=True)

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
    
    # === PART 1: SPARSITY STUDY (without class discrimination) ===
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
        print(f"   Saved sparsity study: {output_path}")
        show()
    
    # === PART 2: SPARSITY PER CLASS (with class discrimination) ===
    if target not in data.columns:
        print(f"   Warning: Target '{target}' not found, skipping per-class analysis")
        continue
    
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
        print(f"   Saved per-class study: {output_path}")
        show()

print("\n" + "=" * 50)
print("SPARSITY PROFILING COMPLETE!")
print(f"Basic plots saved to: {OUTPUT_DIR}")
print(f"Per-class plots saved to: {OUTPUT_DIR_CLASS}")