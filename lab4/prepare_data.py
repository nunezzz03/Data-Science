"""
Data Preparation for Lab 4 - With Data Leakage Fix
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import sys

# Add parent to path for config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab4_config as config


def prepare_dataset(dataset_config):
    """
    Prepare a single dataset for modelling.
    - Removes data leakage columns
    - Handles missing values
    - Encodes categorical variables
    - Scales numeric features
    - Splits into train/test
    """
    name = dataset_config["name"]
    file_tag = dataset_config["file_tag"]
    raw_file = dataset_config["raw_file"]
    target = dataset_config["target"]
    sample_frac = dataset_config["sample_frac"]
    leakage_cols = dataset_config["leakage_cols"]

    print(f"\n{'='*60}")
    print(f"📊 Preparing: {name}")
    print(f"{'='*60}")

    # 1. Load raw data
    filepath = os.path.join(config.RAW_DATA_DIR, raw_file)
    if not os.path.exists(filepath):
        print(f"   ❌ File not found: {filepath}")
        return None

    print(f"   Loading {raw_file}...")
    df = pd.read_csv(filepath)
    print(f"   Original shape: {df.shape}")

    # 2. Sample if needed (for large datasets)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=config.RANDOM_STATE)
        print(f"   Sampled to {len(df)} rows ({sample_frac*100:.1f}%)")

    # 3. Remove data leakage columns
    cols_to_drop = [col for col in leakage_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Removed {len(cols_to_drop)} leakage columns")

    # 4. Handle missing values
    # First, drop rows where target is missing
    if target in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=[target])
        if len(df) < initial_len:
            print(f"   Dropped {initial_len - len(df)} rows with missing target")

    # For other columns, fill or drop based on missing percentage
    missing_pct = df.isnull().sum() / len(df) * 100
    cols_high_missing = missing_pct[missing_pct > 50].index.tolist()
    if cols_high_missing:
        df = df.drop(columns=cols_high_missing)
        print(f"   Dropped {len(cols_high_missing)} columns with >50% missing")

    # Fill remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
            )

    # 5. Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col != target:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Encode target if categorical
    if target in categorical_cols or df[target].dtype == "object":
        le_target = LabelEncoder()
        df[target] = le_target.fit_transform(df[target].astype(str))
        label_encoders[target] = le_target
        print(f"   Target classes: {list(le_target.classes_)}")

    print(f"   Final shape: {df.shape}")

    # 6. Split into train/test
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=df[target],
        )
    except ValueError:
        # Fallback without stratification
        train_df, test_df = train_test_split(
            df,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )

    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # 7. Scale numeric features (excluding target)
    scaler = StandardScaler()
    feature_cols = [col for col in train_df.columns if col != target]

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    # 8. Save processed data
    train_path = os.path.join(config.PROCESSED_DATA_DIR, f"{file_tag}_train.csv")
    test_path = os.path.join(config.PROCESSED_DATA_DIR, f"{file_tag}_test.csv")

    train_scaled.to_csv(train_path, index=False)
    test_scaled.to_csv(test_path, index=False)

    print(f"   Saved: {file_tag}_train.csv, {file_tag}_test.csv")

    # Also save unscaled version for tree-based models
    train_unscaled_path = os.path.join(
        config.PROCESSED_DATA_DIR, f"{file_tag}_train_unscaled.csv"
    )
    test_unscaled_path = os.path.join(
        config.PROCESSED_DATA_DIR, f"{file_tag}_test_unscaled.csv"
    )

    train_df.to_csv(train_unscaled_path, index=False)
    test_df.to_csv(test_unscaled_path, index=False)

    return {
        "train": train_scaled,
        "test": test_scaled,
        "train_unscaled": train_df,
        "test_unscaled": test_df,
        "feature_cols": feature_cols,
        "target": target,
    }


def prepare_all():
    """Prepare all datasets."""
    print("\n" + "=" * 60)
    print("🚀 LAB 4 - DATA PREPARATION (With Leakage Fix)")
    print("=" * 60)

    for dataset in config.DATASETS:
        prepare_dataset(dataset)

    print("\nAll datasets prepared!")
    print(f"   Data saved to: {config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    prepare_all()
