import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import os

# Fix paths to ensure 'utils' can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from utils.dslabs_functions import (
    mvi_by_filling, 
    dummify, 
    select_low_variance_variables, 
    determine_outlier_thresholds_for_var,
    get_variable_types
)

def evaluate_models(X_train, y_train, X_test, y_test, label=""):
    """
    Train and evaluate KNN and Naive Bayes models.
    Returns accuracy scores for both models.
    """
    results = {}
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results['NB'] = accuracy_score(y_test, y_pred_nb)
    
    # KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results['KNN'] = accuracy_score(y_test, y_pred_knn)
    
    avg_score = (results['NB'] + results['KNN']) / 2
    results['AVG'] = avg_score
    
    print(f"         {label}: NB={results['NB']:.4f}, KNN={results['KNN']:.4f}, AVG={avg_score:.4f}")
    
    return results

def random_oversampling(X, y):
    """
    Performs class balancing by randomly duplicating minority class examples.
    """
    df = pd.concat([X, y], axis=1)
    target_col = y.name
    
    class_counts = df[target_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    df_minority_over = df_minority.sample(len(df_majority), replace=True, random_state=42)
    
    df_balanced = pd.concat([df_majority, df_minority_over], axis=0)
    y_balanced = df_balanced[target_col]
    X_balanced = df_balanced.drop(columns=[target_col])
    
    return X_balanced, y_balanced

def random_undersampling(X, y):
    """
    Performs class balancing by randomly removing majority class examples.
    """
    df = pd.concat([X, y], axis=1)
    target_col = y.name
    
    class_counts = df[target_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    df_majority_under = df_majority.sample(len(df_minority), replace=False, random_state=42)
    
    df_balanced = pd.concat([df_majority_under, df_minority], axis=0)
    y_balanced = df_balanced[target_col]
    X_balanced = df_balanced.drop(columns=[target_col])
    
    return X_balanced, y_balanced

def prepare_flights_dataset():
    print("\nSTARTING FLIGHTS DATASET PREPARATION (ITERATIVE OPTIMIZATION)...")
    
    # 0. Loading Data and Removing Leakage
    filename = "data/raw/Combined_Flights_2022.csv"
    filepath = os.path.join(project_root, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    df = pd.read_csv(filepath, na_values="", parse_dates=True)
    
    # Sampling (10% for performance)
    df = df.sample(frac=0.1, random_state=42)
    print(f"   0. Data loaded and sampled: {df.shape}")

    # Remove Leakage columns and unique IDs
    leakage_cols = [
        "ArrTime", "ArrDelayMinutes", "ArrDelay", "ActualElapsedTime",
        "WheelsOn", "TaxiIn", "ArrivalDelayGroups", "ArrTimeBlk",
        "FlightDate", "Tail_Number"
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    print(f"      Removed leakage columns. Shape: {df.shape}")
    
    target = "Cancelled"
    var_types = get_variable_types(df)
    symbolic_vars = var_types["symbolic"]
    numeric_vars_original = var_types["numeric"]
    
    # ========== STEP 1: VARIABLE ENCODING ==========
    print("\n   STEP 1: Variable Encoding (test 2 approaches)")
    
    # Approach A: One-Hot Encoding
    print("      Approach A: One-Hot Encoding")
    df_onehot = df.copy()
    df_onehot = dummify(df_onehot, symbolic_vars)
    
    y_onehot = df_onehot[target]
    X_onehot = df_onehot.drop(columns=[target])
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_onehot, y_onehot, train_size=0.7, stratify=y_onehot, random_state=42
    )
    results_a = evaluate_models(X_train_a, y_train_a, X_test_a, y_test_a, "One-Hot")
    
    # Approach B: Ordinal Encoding
    print("      Approach B: Ordinal Encoding")
    df_ordinal = df.copy()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_ordinal[symbolic_vars] = df_ordinal[symbolic_vars].astype(str)
    df_ordinal[symbolic_vars] = enc.fit_transform(df_ordinal[symbolic_vars])
    
    y_ordinal = df_ordinal[target]
    X_ordinal = df_ordinal.drop(columns=[target])
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_ordinal, y_ordinal, train_size=0.7, stratify=y_ordinal, random_state=42
    )
    results_b = evaluate_models(X_train_b, y_train_b, X_test_b, y_test_b, "Ordinal")
    
    # Select best encoding
    if results_a['AVG'] >= results_b['AVG']:
        print(f"      SELECTED: One-Hot Encoding (AVG={results_a['AVG']:.4f})")
        X_train, X_test, y_train, y_test = X_train_a, X_test_a, y_train_a, y_test_a
        encoding_type = "onehot"
    else:
        print(f"      SELECTED: Ordinal Encoding (AVG={results_b['AVG']:.4f})")
        X_train, X_test, y_train, y_test = X_train_b, X_test_b, y_train_b, y_test_b
        encoding_type = "ordinal"
    
    # ========== STEP 2: MVI (Missing Value Imputation) ==========
    print("\n   STEP 2: MVI (test 2 strategies)")
    
    # Approach A: Most Frequent
    print("      Approach A: Most Frequent")
    X_train_a = mvi_by_filling(X_train.copy(), strategy="frequent")
    X_test_a = mvi_by_filling(X_test.copy(), strategy="frequent")
    X_train_a.index = X_train.index
    X_test_a.index = X_test.index
    results_a = evaluate_models(X_train_a, y_train, X_test_a, y_test, "Frequent")
    
    # Approach B: Mean/Constant
    print("      Approach B: Constant")
    X_train_b = mvi_by_filling(X_train.copy(), strategy="constant")
    X_test_b = mvi_by_filling(X_test.copy(), strategy="constant")
    X_train_b.index = X_train.index
    X_test_b.index = X_test.index
    results_b = evaluate_models(X_train_b, y_train, X_test_b, y_test, "Constant")
    
    # Select best MVI
    if results_a['AVG'] >= results_b['AVG']:
        print(f"      SELECTED: Most Frequent (AVG={results_a['AVG']:.4f})")
        X_train, X_test = X_train_a, X_test_a
    else:
        print(f"      SELECTED: Constant (AVG={results_b['AVG']:.4f})")
        X_train, X_test = X_train_b, X_test_b
    
    # ========== STEP 3: OUTLIERS TREATMENT ==========
    print("\n   STEP 3: Outliers Treatment (test 2 approaches)")
    
    numeric_vars_to_check = [col for col in X_train.columns if col in numeric_vars_original]
    
    if len(numeric_vars_to_check) > 0:
        # Approach A: Standard Deviation based
        print("      Approach A: Std-based (2 std)")
        X_train_a = X_train.copy()
        y_train_a = y_train.copy()
        summary5 = X_train_a[numeric_vars_to_check].describe()
        outlier_indices = []
        for var in numeric_vars_to_check:
            top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=True, threshold=2)
            outliers = X_train_a[(X_train_a[var] > top) | (X_train_a[var] < bottom)]
            outlier_indices.extend(outliers.index.tolist())
        outlier_indices = list(set(outlier_indices))
        X_train_a = X_train_a.drop(outlier_indices, axis=0)
        y_train_a = y_train_a.drop(outlier_indices, axis=0)
        results_a = evaluate_models(X_train_a, y_train_a, X_test, y_test, "Std-based")
        
        # Approach B: IQR based
        print("      Approach B: IQR-based (1.5 IQR)")
        X_train_b = X_train.copy()
        y_train_b = y_train.copy()
        summary5 = X_train_b[numeric_vars_to_check].describe()
        outlier_indices = []
        for var in numeric_vars_to_check:
            top, bottom = determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=1.5)
            outliers = X_train_b[(X_train_b[var] > top) | (X_train_b[var] < bottom)]
            outlier_indices.extend(outliers.index.tolist())
        outlier_indices = list(set(outlier_indices))
        X_train_b = X_train_b.drop(outlier_indices, axis=0)
        y_train_b = y_train_b.drop(outlier_indices, axis=0)
        results_b = evaluate_models(X_train_b, y_train_b, X_test, y_test, "IQR-based")
        
        # Select best outlier treatment
        if results_a['AVG'] >= results_b['AVG']:
            print(f"      SELECTED: Std-based (AVG={results_a['AVG']:.4f})")
            X_train, y_train = X_train_a, y_train_a
        else:
            print(f"      SELECTED: IQR-based (AVG={results_b['AVG']:.4f})")
            X_train, y_train = X_train_b, y_train_b
    else:
        print("      SKIPPED: No numeric variables for outlier removal")
    
    # ========== STEP 4: SCALING ==========
    print("\n   STEP 4: Scaling (test 2 approaches)")
    
    numeric_vars_to_scale = [col for col in X_train.columns if col in numeric_vars_original]
    
    if len(numeric_vars_to_scale) > 0:
        # Approach A: StandardScaler
        print("      Approach A: StandardScaler")
        X_train_a = X_train.copy()
        X_test_a = X_test.copy()
        scaler = StandardScaler()
        X_train_a[numeric_vars_to_scale] = scaler.fit_transform(X_train_a[numeric_vars_to_scale])
        X_test_a[numeric_vars_to_scale] = scaler.transform(X_test_a[numeric_vars_to_scale])
        results_a = evaluate_models(X_train_a, y_train, X_test_a, y_test, "StandardScaler")
        
        # Approach B: MinMaxScaler
        print("      Approach B: MinMaxScaler")
        X_train_b = X_train.copy()
        X_test_b = X_test.copy()
        scaler = MinMaxScaler()
        X_train_b[numeric_vars_to_scale] = scaler.fit_transform(X_train_b[numeric_vars_to_scale])
        X_test_b[numeric_vars_to_scale] = scaler.transform(X_test_b[numeric_vars_to_scale])
        results_b = evaluate_models(X_train_b, y_train, X_test_b, y_test, "MinMaxScaler")
        
        # Select best scaling
        if results_a['AVG'] >= results_b['AVG']:
            print(f"      SELECTED: StandardScaler (AVG={results_a['AVG']:.4f})")
            X_train, X_test = X_train_a, X_test_a
        else:
            print(f"      SELECTED: MinMaxScaler (AVG={results_b['AVG']:.4f})")
            X_train, X_test = X_train_b, X_test_b
    else:
        print("      SKIPPED: No numeric variables for scaling")
    
    # ========== STEP 5: BALANCING ==========
    print("\n   STEP 5: Balancing (test 2 approaches)")
    
    # Approach A: Random Oversampling
    print("      Approach A: Random Oversampling")
    X_train_a, y_train_a = random_oversampling(X_train.copy(), y_train.copy())
    results_a = evaluate_models(X_train_a, y_train_a, X_test, y_test, "Oversampling")
    
    # Approach B: Random Undersampling
    print("      Approach B: Random Undersampling")
    X_train_b, y_train_b = random_undersampling(X_train.copy(), y_train.copy())
    results_b = evaluate_models(X_train_b, y_train_b, X_test, y_test, "Undersampling")
    
    # Select best balancing
    if results_a['AVG'] >= results_b['AVG']:
        print(f"      SELECTED: Oversampling (AVG={results_a['AVG']:.4f})")
        X_train, y_train = X_train_a, y_train_a
    else:
        print(f"      SELECTED: Undersampling (AVG={results_b['AVG']:.4f})")
        X_train, y_train = X_train_b, y_train_b
    
    # ========== STEP 6: FEATURE SELECTION ==========
    print("\n   STEP 6: Feature Selection (test 2 approaches)")
    
    # Approach A: Low Variance (threshold=0.1)
    print("      Approach A: Low Variance (0.1)")
    X_train_a = X_train.copy()
    X_test_a = X_test.copy()
    train_temp = pd.concat([X_train_a, y_train], axis=1)
    vars_to_drop = select_low_variance_variables(train_temp, max_threshold=0.1, target=target)
    X_train_a = X_train_a.drop(columns=[c for c in vars_to_drop if c in X_train_a.columns])
    X_test_a = X_test_a.drop(columns=[c for c in vars_to_drop if c in X_test_a.columns])
    results_a = evaluate_models(X_train_a, y_train, X_test_a, y_test, "Variance(0.1)")
    
    # Approach B: Low Variance (threshold=0.01)
    print("      Approach B: Low Variance (0.01)")
    X_train_b = X_train.copy()
    X_test_b = X_test.copy()
    train_temp = pd.concat([X_train_b, y_train], axis=1)
    vars_to_drop = select_low_variance_variables(train_temp, max_threshold=0.01, target=target)
    X_train_b = X_train_b.drop(columns=[c for c in vars_to_drop if c in X_train_b.columns])
    X_test_b = X_test_b.drop(columns=[c for c in vars_to_drop if c in X_test_b.columns])
    results_b = evaluate_models(X_train_b, y_train, X_test_b, y_test, "Variance(0.01)")
    
    # Select best feature selection
    if results_a['AVG'] >= results_b['AVG']:
        print(f"      SELECTED: Variance(0.1) (AVG={results_a['AVG']:.4f})")
        X_train, X_test = X_train_a, X_test_a
    else:
        print(f"      SELECTED: Variance(0.01) (AVG={results_b['AVG']:.4f})")
        X_train, X_test = X_train_b, X_test_b
    
    # ========== STEP 7: FEATURE GENERATION ==========
    print("\n   STEP 7: Feature Generation (skipped)")
    
    # Save the final optimized dataset
    save_split(f"flights_{encoding_type}_optimized", X_train, y_train, X_test, y_test)
    
    print("\n========================================")
    print("DONE! Optimized dataset saved in data/processed/")
    print(f"Final dataset shape: Train={X_train.shape}, Test={X_test.shape}")
    print("========================================")

def save_split(prefix, X_train, y_train, X_test, y_test):
    output_dir = os.path.join(project_root, "data/processed")
    os.makedirs(output_dir, exist_ok=True)
    
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    train.to_csv(f"{output_dir}/{prefix}_train.csv", index=False)
    test.to_csv(f"{output_dir}/{prefix}_test.csv", index=False)
    print(f"      Saved: {prefix} (Train: {train.shape}, Test: {test.shape})")

if __name__ == "__main__":
    prepare_flights_dataset()
