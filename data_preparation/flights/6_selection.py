"""
STEP 6: Feature Selection
Tests Low Variance with different thresholds
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils.dslabs_functions import select_low_variance_variables
import lab3_config as config
import utils as flight_utils


def run_selection():
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE SELECTION")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_BALANCED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_BALANCED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: LOW VARIANCE (0.1) =====
    print("\n   Approach A: Low Variance (threshold=0.1)")
    X_train_a = X_train.copy()
    X_test_a = X_test.copy()
    
    train_temp = pd.concat([X_train_a, y_train], axis=1)
    vars_to_drop = select_low_variance_variables(train_temp, max_threshold=0.1, target=config.TARGET)
    
    X_train_a = X_train_a.drop(columns=[c for c in vars_to_drop if c in X_train_a.columns])
    X_test_a = X_test_a.drop(columns=[c for c in vars_to_drop if c in X_test_a.columns])
    
    print(f"      Dropped {len(vars_to_drop)} variables")
    results['Variance(0.1)'] = flights_utils.evaluate_models(X_train_a, y_train, X_test_a, y_test, "Variance(0.1)")
    datasets['Variance(0.1)'] = (X_train_a, y_train, X_test_a, y_test)
    
    # ===== APPROACH B: LOW VARIANCE (0.01) =====
    print("\n   Approach B: Low Variance (threshold=0.01)")
    X_train_b = X_train.copy()
    X_test_b = X_test.copy()
    
    train_temp = pd.concat([X_train_b, y_train], axis=1)
    vars_to_drop = select_low_variance_variables(train_temp, max_threshold=0.01, target=config.TARGET)
    
    X_train_b = X_train_b.drop(columns=[c for c in vars_to_drop if c in X_train_b.columns])
    X_test_b = X_test_b.drop(columns=[c for c in vars_to_drop if c in X_test_b.columns])
    
    print(f"      Dropped {len(vars_to_drop)} variables")
    results['Variance(0.01)'] = flights_utils.evaluate_models(X_train_b, y_train, X_test_b, y_test, "Variance(0.01)")
    datasets['Variance(0.01)'] = (X_train_b, y_train, X_test_b, y_test)
    
    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Save best dataset (FINAL)
    X_train, y_train, X_test, y_test = datasets[best_approach]
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_SELECTED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "06_selection_comparison.png")
    flights_utils.plot_comparison(results, "Step 6: Feature Selection", chart_path)
    
    print(f"\n   Step 6 Complete!")
    print(f"\n   FINAL DATASET: {config.FILE_SELECTED}")
    print(f"   Final shape - Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    run_selection()
