"""
STEP 3: Outlier Treatment
Tests Standard Deviation vs IQR methods
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils.dslabs_functions import determine_outlier_thresholds_for_var, get_variable_types
import lab3_config as config
import utils as flight_utils


def run_outliers():
    print("\n" + "=" * 60)
    print("STEP 3: OUTLIER TREATMENT")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_IMPUTED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_IMPUTED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Get numeric variables
    numeric_vars = get_variable_types(X_train)["numeric"]
    print(f"   Numeric variables: {len(numeric_vars)}")
    
    if len(numeric_vars) == 0:
        print("   SKIPPED: No numeric variables for outlier removal")
        flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_OUTLIERS)
        return
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: STANDARD DEVIATION (2 std) =====
    print("\n   Approach A: Std-based (2 std)")
    X_train_a = X_train.copy()
    y_train_a = y_train.copy()
    
    summary5 = X_train_a[numeric_vars].describe()
    outlier_indices = []
    
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=True, threshold=2)
        outliers = X_train_a[(X_train_a[var] > top) | (X_train_a[var] < bottom)]
        outlier_indices.extend(outliers.index.tolist())
    
    outlier_indices = list(set(outlier_indices))
    X_train_a = X_train_a.drop(outlier_indices, axis=0)
    y_train_a = y_train_a.drop(outlier_indices, axis=0)
    
    print(f"      Removed {len(outlier_indices)} outlier records")
    results['Std-based'] = flights_utils.evaluate_models(X_train_a, y_train_a, X_test, y_test, "Std-based")
    datasets['Std-based'] = (X_train_a, y_train_a, X_test, y_test)
    
    # ===== APPROACH B: IQR (1.5 IQR) =====
    print("\n   Approach B: IQR-based (1.5 IQR)")
    X_train_b = X_train.copy()
    y_train_b = y_train.copy()
    
    summary5 = X_train_b[numeric_vars].describe()
    outlier_indices = []
    
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=1.5)
        outliers = X_train_b[(X_train_b[var] > top) | (X_train_b[var] < bottom)]
        outlier_indices.extend(outliers.index.tolist())
    
    outlier_indices = list(set(outlier_indices))
    X_train_b = X_train_b.drop(outlier_indices, axis=0)
    y_train_b = y_train_b.drop(outlier_indices, axis=0)
    
    print(f"      Removed {len(outlier_indices)} outlier records")
    results['IQR-based'] = flights_utils.evaluate_models(X_train_b, y_train_b, X_test, y_test, "IQR-based")
    datasets['IQR-based'] = (X_train_b, y_train_b, X_test, y_test)
    
    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Save best dataset
    X_train, y_train, X_test, y_test = datasets[best_approach]
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_OUTLIERS,
                 metadata={'approach': best_approach, 'f1_score': best_f1})
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "03_outliers_comparison.png")
    flights_utils.plot_comparison(results, "Step 3: Outlier Treatment", chart_path)
    
    print(f"\n   Step 3 Complete!")


if __name__ == "__main__":
    run_outliers()
