import pandas as pd
from sklearn.model_selection import train_test_split
import os


def process_dataset(
    filename, target_col, output_tag, sample_fraction=1.0, exclude_cols=None
):
    print(f"--- Processing {filename} ---")
    try:
        # Read file from datasets folder
        filepath = os.path.join("data/raw", filename)
        df = pd.read_csv(filepath)

        # If it's the massive flights file, sample it down to avoid crashes
        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)
            print(f"   ⚠️ Downsampled to {len(df)} rows for speed.")

        # 1. Drop Missing
        df = df.dropna()

        # 2. Discard Non-Numeric (Keep Target)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cols_to_keep = numeric_cols.copy()
        if target_col not in cols_to_keep:
            cols_to_keep.append(target_col)

        df_clean = df[cols_to_keep]

        # 3. Remove excluded columns (e.g., data leakage features)
        if exclude_cols:
            df_clean = df_clean.drop(
                columns=[col for col in exclude_cols if col in df_clean.columns]
            )
            print(
                f"   🚫 Excluded leakage features: {[col for col in exclude_cols if col in df.columns]}"
            )

        # 4. Split
        # Check if target allows stratification
        try:
            train_df, test_df = train_test_split(
                df_clean, test_size=0.3, random_state=42, stratify=df_clean[target_col]
            )
        except ValueError:
            # Fallback if stratification fails (e.g. for regression or rare classes)
            train_df, test_df = train_test_split(
                df_clean, test_size=0.3, random_state=42
            )

        # 5. Save to data/processed folder
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, f"{output_tag}_train.csv")
        test_path = os.path.join(output_dir, f"{output_tag}_test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(
            f"✅ Created {output_tag} files: Train={len(train_df)}, Test={len(test_df)}"
        )

    except FileNotFoundError:
        print(f"❌ Could not find {filepath}")


# --- RUN LIST ---
print("\n🔧 FIXED DATA PREPARATION - Addressing Data Leakage Issues\n")

# 1. Traffic - Remove 'Total' column (derived feature)
print("1️⃣ TRAFFIC DATASET")
process_dataset(
    "TrafficTwoMonth.csv", "Traffic Situation", "traffic", exclude_cols=["Total"]
)

# 2. Accidents - Keep as is (no issues found)
print("\n2️⃣ ACCIDENTS DATASET")
process_dataset("traffic_accidents.csv", "crash_type", "accidents")

# 3. Economic - Create proper classification task: predict GDP Growth category
print("\n3️⃣ ECONOMIC DATASET - FIXED TASK")
try:
    df_econ = pd.read_csv("data/raw/economic_indicators_dataset_2010_2023.csv")
    df_econ = df_econ.dropna()

    # Create meaningful target: GDP Growth Rate categories
    df_econ["GDP_Category"] = pd.cut(
        df_econ["GDP Growth Rate (%)"],
        bins=[-float("inf"), 0, 2, 4, float("inf")],
        labels=["Negative", "Low", "Medium", "High"],
    )

    # Keep only numeric features + new target
    numeric_cols = df_econ.select_dtypes(include=["number"]).columns.tolist()
    # Remove original GDP Growth Rate (would be leakage to predict GDP category)
    numeric_cols = [col for col in numeric_cols if col != "GDP Growth Rate (%)"]
    cols_to_keep = numeric_cols + ["GDP_Category"]

    df_econ_clean = df_econ[cols_to_keep]

    # Split with stratification
    train_df, test_df = train_test_split(
        df_econ_clean,
        test_size=0.3,
        random_state=42,
        stratify=df_econ_clean["GDP_Category"],
    )

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "economic_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "economic_test.csv"), index=False)
    print(
        f"✅ Created economic files with GDP_Category target: Train={len(train_df)}, Test={len(test_df)}"
    )
    print(
        f"   Target distribution: {train_df['GDP_Category'].value_counts().to_dict()}"
    )
except Exception as e:
    print(f"❌ Economic processing failed: {e}")

# 4. Flights - Remove ALL features only known after arrival (DATA LEAKAGE FIX)
print("\n4️⃣ FLIGHTS DATASET - FIXED DATA LEAKAGE")
leakage_features = [
    "ArrTime",  # Only known after landing
    "ArrDelayMinutes",  # Contains the answer!
    "ArrDelay",  # Contains the answer!
    "ActualElapsedTime",  # Only known after landing
    "WheelsOn",  # Only known after landing
    "TaxiIn",  # Only known after landing
    "ArrivalDelayGroups",  # Derived from target
    "ArrTimeBlk",  # Only known after landing
]
process_dataset(
    "Combined_Flights_2022.csv",
    "ArrDel15",
    "flights",
    sample_fraction=0.1,
    exclude_cols=leakage_features,
)

print("\n✅ ALL DATASETS PROCESSED WITH FIXES APPLIED!")
