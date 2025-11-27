"""
STEP 4: Feature Scaling
Tests StandardScaler vs MinMaxScaler
"""
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils.dslabs_functions import get_variable_types
import lab3_config as config
import flights_utils


def run_scaling():
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE SCALING")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_OUTLIERS}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_OUTLIERS)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Get numeric variables
    numeric_vars = get_variable_types(X_train)["numeric"]
    print(f"   Numeric variables: {len(numeric_vars)}")
    
    if len(numeric_vars) == 0:
        print("   SKIPPED: No numeric variables for scaling")
        flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_SCALED)
        return
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: STANDARD SCALER =====
    print("\n   Approach A: StandardScaler")
    X_train_a = X_train.copy()
    X_test_a = X_test.copy()
    
    scaler = StandardScaler()
    X_train_a[numeric_vars] = scaler.fit_transform(X_train_a[numeric_vars])
    X_test_a[numeric_vars] = scaler.transform(X_test_a[numeric_vars])
    
    results['StandardScaler'] = flights_utils.evaluate_models(X_train_a, y_train, X_test_a, y_test, "StandardScaler")
    datasets['StandardScaler'] = (X_train_a, y_train, X_test_a, y_test)
    
    # ===== APPROACH B: MINMAX SCALER =====
    print("\n   Approach B: MinMaxScaler")
    X_train_b = X_train.copy()
    X_test_b = X_test.copy()
    
    scaler = MinMaxScaler()
    X_train_b[numeric_vars] = scaler.fit_transform(X_train_b[numeric_vars])
    X_test_b[numeric_vars] = scaler.transform(X_test_b[numeric_vars])
    
    results['MinMaxScaler'] = flights_utils.evaluate_models(X_train_b, y_train, X_test_b, y_test, "MinMaxScaler")
    datasets['MinMaxScaler'] = (X_train_b, y_train, X_test_b, y_test)
    
    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Save best dataset
    X_train, y_train, X_test, y_test = datasets[best_approach]
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_SCALED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "04_scaling_comparison.png")
    flights_utils.plot_comparison(results, "Step 4: Feature Scaling", chart_path)
    
    print(f"\n   Step 4 Complete!")


if __name__ == "__main__":
    run_scaling()
