"""
STEP 1: Variable Encoding
Tests One-Hot vs Ordinal Encoding
"""
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Add project root and utils to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
utils_path = os.path.join(project_root, 'utils')
sys.path.insert(0, project_root)
sys.path.insert(0, utils_path)
sys.path.insert(0, os.path.dirname(__file__))

from utils.dslabs_functions import dummify, get_variable_types
import lab3_config as config
import flights_utils


def run_encoding():
    print("\n" + "=" * 60)
    print("STEP 1: VARIABLE ENCODING")
    print("=" * 60)
    
    # Load raw data
    print(f"\n   Loading data from {config.RAW_DATA}")
    df = pd.read_csv(config.RAW_DATA, na_values="", parse_dates=True)
    
    # Sample for performance
    df = df.sample(frac=config.SAMPLE_FRAC, random_state=config.RANDOM_STATE)
    print(f"   Sampled {config.SAMPLE_FRAC*100}%: {df.shape}")
    
    # Remove leakage columns
    df = df.drop(columns=[c for c in config.LEAKAGE_COLS if c in df.columns])
    print(f"   Removed leakage columns: {df.shape}")
    
    # Get variable types
    var_types = get_variable_types(df)
    symbolic_vars = var_types["symbolic"]
    
    print(f"   Symbolic variables to encode: {len(symbolic_vars)}")
    
    # Store results for comparison
    results = {}
    datasets = {}
    
    # ===== APPROACH A: ONE-HOT ENCODING =====
    print("\n   Approach A: One-Hot Encoding")
    df_onehot = df.copy()
    df_onehot = dummify(df_onehot, symbolic_vars)
    
    y_onehot = df_onehot[config.TARGET]
    X_onehot = df_onehot.drop(columns=[config.TARGET])
    
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_onehot, y_onehot, train_size=config.TRAIN_SIZE, 
        stratify=y_onehot, random_state=config.RANDOM_STATE
    )
    
    results['One-Hot'] = flights_utils.evaluate_models(X_train_a, y_train_a, X_test_a, y_test_a, "One-Hot")
    datasets['One-Hot'] = (X_train_a, y_train_a, X_test_a, y_test_a)
    
    # ===== APPROACH B: ORDINAL ENCODING =====
    print("\n   Approach B: Ordinal Encoding")
    df_ordinal = df.copy()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_ordinal[symbolic_vars] = df_ordinal[symbolic_vars].astype(str)
    df_ordinal[symbolic_vars] = enc.fit_transform(df_ordinal[symbolic_vars])
    
    y_ordinal = df_ordinal[config.TARGET]
    X_ordinal = df_ordinal.drop(columns=[config.TARGET])
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_ordinal, y_ordinal, train_size=config.TRAIN_SIZE,
        stratify=y_ordinal, random_state=config.RANDOM_STATE
    )
    
    results['Ordinal'] = flights_utils.evaluate_models(X_train_b, y_train_b, X_test_b, y_test_b, "Ordinal")
    datasets['Ordinal'] = (X_train_b, y_train_b, X_test_b, y_test_b)
    
    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Save best dataset
    X_train, y_train, X_test, y_test = datasets[best_approach]
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_ENCODED, 
                 metadata={'approach': best_approach, 'f1_score': best_f1})
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "01_encoding_comparison.png")
    flights_utils.plot_comparison(results, "Step 1: Variable Encoding", chart_path)
    
    print(f"\n   Step 1 Complete!")


if __name__ == "__main__":
    run_encoding()
