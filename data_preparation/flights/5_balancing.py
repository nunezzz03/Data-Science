"""
STEP 5: Class Balancing
Tests Random Oversampling vs Random Undersampling
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

import lab3_config as config
import flights_utils


def run_balancing():
    print("\n" + "=" * 60)
    print("STEP 5: CLASS BALANCING")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_SCALED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_SCALED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Check class distribution
    class_counts = y_train.value_counts()
    print(f"   Original class distribution: {class_counts.to_dict()}")
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: RANDOM OVERSAMPLING =====
    print("\n   Approach A: Random Oversampling")
    X_train_a, y_train_a = flights_utils.random_oversampling(X_train.copy(), y_train.copy())
    print(f"      New class distribution: {y_train_a.value_counts().to_dict()}")
    
    results['Oversampling'] = flights_utils.evaluate_models(X_train_a, y_train_a, X_test, y_test, "Oversampling")
    datasets['Oversampling'] = (X_train_a, y_train_a, X_test, y_test)
    
    # ===== APPROACH B: RANDOM UNDERSAMPLING =====
    print("\n   Approach B: Random Undersampling")
    X_train_b, y_train_b = flights_utils.random_undersampling(X_train.copy(), y_train.copy())
    print(f"      New class distribution: {y_train_b.value_counts().to_dict()}")
    
    results['Undersampling'] = flights_utils.evaluate_models(X_train_b, y_train_b, X_test, y_test, "Undersampling")
    datasets['Undersampling'] = (X_train_b, y_train_b, X_test, y_test)
    
    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Save best dataset
    X_train, y_train, X_test, y_test = datasets[best_approach]
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_BALANCED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "05_balancing_comparison.png")
    flights_utils.plot_comparison(results, "Step 5: Class Balancing", chart_path)
    
    print(f"\n   Step 5 Complete!")


if __name__ == "__main__":
    run_balancing()
