import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
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

def random_oversampling(X, y):
    """
    Performs class balancing by randomly duplicating minority class examples (Pandas only).
    """
    print(f"      ⚖️ Balancing classes (Random Oversampling)...")
    
    # Merge X and y
    df = pd.concat([X, y], axis=1)
    target_col = y.name
    
    # Count classes
    class_counts = df[target_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    # Separate data
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] == minority_class]
    
    # Duplicate minority class
    df_minority_over = df_minority.sample(len(df_majority), replace=True, random_state=42)
    
    # Recombine
    df_balanced = pd.concat([df_majority, df_minority_over], axis=0)
    y_balanced = df_balanced[target_col]
    X_balanced = df_balanced.drop(columns=[target_col])
    
    print(f"         -> Balanced classes: {y_balanced.value_counts().to_dict()}")
    return X_balanced, y_balanced

def prepare_flights_dataset():
    print("\n✈️ STARTING FLIGHTS DATASET PREPARATION (Lab Style)...")
    
    # 1. Loading Data and Removing Leakage
    filename = "data/raw/Combined_Flights_2022.csv"
    filepath = os.path.join(project_root, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        return

    df = pd.read_csv(filepath, na_values="", parse_dates=True)
    
    # Sampling (10% for performance)
    df = df.sample(frac=0.1, random_state=42)
    print(f"   1. Data loaded and sampled: {df.shape}")

    # Remove Leakage columns and unique IDs
    leakage_cols = [
        "ArrTime", "ArrDelayMinutes", "ArrDelay", "ActualElapsedTime",
        "WheelsOn", "TaxiIn", "ArrivalDelayGroups", "ArrTimeBlk",
        "FlightDate", "Tail_Number"
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    target = "ArrDel15"
    
    # 2. Missing Values (Imputation)
    df = mvi_by_filling(df, strategy="frequent")
    print(f"   2. Missing Values handled. Shape: {df.shape}")

    # 3. Outliers (Standard Deviation Method)
    numeric_vars = get_variable_types(df)["numeric"]
    if target in numeric_vars: numeric_vars.remove(target)
    
    summary5 = df[numeric_vars].describe()
    initial_rows = df.shape[0]
    
    print("   3. Removing outliers...")
    for var in numeric_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(summary5[var])
        outliers = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        df.drop(outliers.index, axis=0, inplace=True)
        
    print(f"      -> Removed {initial_rows - df.shape[0]} records. Current shape: {df.shape}")

    # 4. Feature Selection (Low Variance)
    vars_to_drop = select_low_variance_variables(df, max_threshold=0.1, target=target)
    df = df.drop(columns=vars_to_drop)
    print(f"   4. Feature Selection complete (Low Variance). Variables: {df.shape[1]}")

    # 5. Scaling (StandardScaler)
    numeric_vars = get_variable_types(df)["numeric"]
    if target in numeric_vars: numeric_vars.remove(target)
    
    scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df[numeric_vars])
    df_scaled = df.copy()
    df_scaled[numeric_vars] = scaler.transform(df[numeric_vars])
    print("   5. Scaling applied (StandardScaler).")

    # Create Final Datasets (Ordinal vs One-Hot)
    y = df_scaled[target]
    X = df_scaled.drop(columns=[target])
    symbolic_vars = get_variable_types(X)["symbolic"]

    # A: One-Hot Encoding
    print("\n   🅰️  Creating One-Hot Encoding version...")
    X_onehot = dummify(X, symbolic_vars)
    
    # Split and Balance
    X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, train_size=0.7, stratify=y, random_state=42)
    X_train_bal, y_train_bal = random_oversampling(X_train, y_train)
    
    save_split("flights_onehot", X_train_bal, y_train_bal, X_test, y_test)

    # B: Ordinal Encoding
    print("\n   🅱️  Creating Ordinal Encoding version...")
    X_ordinal = X.copy()
    enc = OrdinalEncoder()
    
    # Ensure variables are strings to prevent errors
    X_ordinal[symbolic_vars] = X_ordinal[symbolic_vars].astype(str)
    X_ordinal[symbolic_vars] = enc.fit_transform(X_ordinal[symbolic_vars])
    
    X_train_ord, X_test_ord, y_train_ord, y_test_ord = train_test_split(X_ordinal, y, train_size=0.7, stratify=y, random_state=42)
    X_train_ord_bal, y_train_ord_bal = random_oversampling(X_train_ord, y_train_ord)
    
    save_split("flights_ordinal", X_train_ord_bal, y_train_ord_bal, X_test_ord, y_test_ord)

    print("\n✅ DONE! Files saved in data/processed/")

def save_split(prefix, X_train, y_train, X_test, y_test):
    output_dir = os.path.join(project_root, "data/processed")
    os.makedirs(output_dir, exist_ok=True)
    
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    train.to_csv(f"{output_dir}/{prefix}_train.csv", index=False)
    test.to_csv(f"{output_dir}/{prefix}_test.csv", index=False)
    print(f"      💾 Saved: {prefix} (Train: {train.shape}, Test: {test.shape})")

if __name__ == "__main__":
    prepare_flights_dataset()