"""
STEP 2: Missing Value Imputation
Tests Most Frequent vs Constant strategies
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils.dslabs_functions import mvi_by_filling
import lab3_config as config
import utils as flight_utils


def run_imputation():
    print("\n" + "=" * 60)
    print("STEP 2: MISSING VALUE IMPUTATION")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_ENCODED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_ENCODED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: MOST FREQUENT =====
    print("\n   Approach A: Most Frequent")
    train_idx, test_idx = X_train.index, X_test.index
    
    X_train_a = mvi_by_filling(X_train.copy(), strategy="frequent")
    X_test_a = mvi_by_filling(X_test.copy(), strategy="frequent")
    
    X_train_a.index = train_idx
    X_test_a.index = test_idx
    
    results['Frequent'] = flights_utils.evaluate_models(X_train_a, y_train, X_test_a, y_test, "Frequent")
    datasets['Frequent'] = (X_train_a, y_train, X_test_a, y_test)
    
    # ===== APPROACH B: CONSTANT =====
    print("\n   Approach B: Constant")
    X_train_b = mvi_by_filling(X_train.copy(), strategy="constant")
    X_test_b = mvi_by_filling(X_test.copy(), strategy="constant")
    
    X_train_b.index = train_idx
    X_test_b.index = test_idx
    
    results['Constant'] = flights_utils.evaluate_models(X_train_b, y_train, X_test_b, y_test, "Constant")
    datasets['Constant'] = (X_train_b, y_train, X_test_b, y_test)
    
    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Save best dataset
    X_train, y_train, X_test, y_test = datasets[best_approach]
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_IMPUTED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "02_imputation_comparison.png")
    flights_utils.plot_comparison(results, "Step 2: Missing Value Imputation", chart_path)
    
    print(f"\n   Step 2 Complete!")


if __name__ == "__main__":
    run_imputation()
