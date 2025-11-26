import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import lab3_config as config

import dslabs_functions as ds


def run_encoding():
    print("\n[Step 1] Encoding...")

    # Load Data
    df = pd.read_csv(config.RAW_DATA_PATH)

    # Basic cleanup (drop duplicates)
    df = df.drop_duplicates()

    # Drop date column if present (components already exist)
    if "crash_date" in df.columns:
        df = df.drop(columns=["crash_date"])

    # Separate Target
    target = config.TARGET

    # Data Leakage ---
    # Drop injury-related columns as they are part of the outcome (crash_type)
    leakage_cols = [
        "injuries_total",
        "injuries_fatal",
        "injuries_incapacitating",
        "injuries_non_incapacitating",
        "injuries_reported_not_evident",
        "injuries_no_indication",
        "damage",  # Often correlated with severity/type directly
        "prim_contributory_cause",
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    if cols_to_drop:
        print(f"   Removing leakage variables: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Encoding (Manual Mappings + Cyclical) ---
    print("   Applying Smart Encoding (Manual Mappings + Cyclical)...")

    # 1. Manual Risk-Based Mappings
    weather_condition_values = {
        "CLEAR": 1,
        "CLOUDY/OVERCAST": 1,
        "OTHER": 1,
        "FOG/SMOKE/HAZE": 2,
        "RAIN": 2,
        "BLOWING SAND, SOIL, DIRT": 2,
        "SNOW": 3,
        "SLEET/HAIL": 3,
        "FREEZING RAIN/DRIZZLE": 4,
        "BLOWING SNOW": 4,
        "SEVERE CROSS WIND GATE": 4,
    }
    lighting_condition_values = {
        "DAYLIGHT": 1,
        "DAWN": 2,
        "DUSK": 2,
        "DARKNESS, LIGHTED ROAD": 3,
        "DARKNESS": 4,
    }
    trafficway_type_values = {
        "NOT REPORTED": 1,
        "ALLEY": 1,
        "DRIVEWAY": 1,
        "PARKING LOT": 1,
        "ONE-WAY": 1,
        "NOT DIVIDED": 1,
        "DIVIDED - W/MEDIAN (NOT RAISED)": 2,
        "DIVIDED - W/MEDIAN BARRIER": 2,
        "CENTER TURN LANE": 2,
        "OTHER": 2,
        "UNKNOWN INTERSECTION TYPE": 2,
        "T-INTERSECTION": 3,
        "L-INTERSECTION": 3,
        "FOUR WAY": 3,
        "Y-INTERSECTION": 3,
        "RAMP": 3,
        "TRAFFIC ROUTE": 3,
        "ROUNDABOUT": 4,
        "FIVE POINT, OR MORE": 4,
    }
    roadway_surface_cond_values = {
        "DRY": 1,
        "OTHER": 2,
        "WET": 2,
        "SAND, MUD, DIRT": 3,
        "SNOW OR SLUSH": 3,
        "ICE": 4,
    }
    road_defect_values = {
        "NO DEFECTS": 1,
        "OTHER": 2,
        "SHOULDER DEFECT": 2,
        "WORN SURFACE": 3,
        "DEBRIS ON ROADWAY": 3,
        "RUT, HOLES": 4,
    }
    intersection_related_i_values = {"N": 0, "Y": 1}

    encoding_map = {
        "weather_condition": weather_condition_values,
        "lighting_condition": lighting_condition_values,
        "trafficway_type": trafficway_type_values,
        "roadway_surface_cond": roadway_surface_cond_values,
        "road_defect": road_defect_values,
        "intersection_related_i": intersection_related_i_values,
    }

    # Apply mappings (only for columns that exist)
    for col, mapping in encoding_map.items():
        if col in df.columns:
            # Convert column to object/string first to ensure matching
            df[col] = df[col].astype(str).replace(mapping)
            # Force to numeric, coercing errors (unmapped values) to NaN (or keep as is? better to keep as is if mixed)
            # But we want them to be numeric. Let's try pd.to_numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Cyclical Encoding for Time
    if "crash_hour" in df.columns:
        df["crash_hour_sin"] = np.sin(2 * np.pi * df["crash_hour"] / 24) + 1
        df["crash_hour_cos"] = np.cos(2 * np.pi * df["crash_hour"] / 24) + 1

    if "crash_day_of_week" in df.columns:
        df["crash_day_sin"] = np.sin(2 * np.pi * df["crash_day_of_week"] / 7) + 1
        df["crash_day_cos"] = np.cos(2 * np.pi * df["crash_day_of_week"] / 7) + 1
        # Drop original? Maybe keep for now, or drop to avoid redundancy.
        # Colleague dropped it. Let's drop it to be clean.
        df = df.drop(columns=["crash_day_of_week"])

    if "crash_month" in df.columns:
        df["crash_month_sin"] = np.sin(2 * np.pi * df["crash_month"] / 12) + 1
        df["crash_month_cos"] = np.cos(2 * np.pi * df["crash_month"] / 12) + 1
        # Drop original?
        df = df.drop(columns=["crash_month"])

    # --- FIX: Feature Generation (Interactions & Risk Score) ---
    print("   Generating Interaction Features & Risk Score...")

    # Interaction: weather * road surface
    if "weather_condition" in df.columns and "roadway_surface_cond" in df.columns:
        df["weather_surface_interaction"] = (
            df["weather_condition"] * df["roadway_surface_cond"]
        )

    # Interaction: lighting * hour
    if "lighting_condition" in df.columns and "crash_hour" in df.columns:
        df["lighting_hour_interaction"] = df["lighting_condition"] * df["crash_hour"]

    # Interaction: trafficway * intersection
    if "trafficway_type" in df.columns and "intersection_related_i" in df.columns:
        df["trafficway_intersection_interaction"] = (
            df["trafficway_type"] * df["intersection_related_i"]
        )

    # Risk Score
    risk_features = ["weather_condition", "roadway_surface_cond", "lighting_condition"]
    existing_risk_features = [f for f in risk_features if f in df.columns]
    if existing_risk_features:
        df["risk_score"] = df[existing_risk_features].sum(axis=1)
        # Normalize risk score (optional, but good for consistency)
        if df["risk_score"].max() != 0:
            df["risk_score"] = df["risk_score"] / df["risk_score"].max()

    # Variables
    vars_types = ds.get_variable_types(df)
    symbolic_vars = vars_types["symbolic"]
    binary_vars = vars_types["binary"]

    # Combine all categorical variables to encode
    vars_to_encode = symbolic_vars + binary_vars

    # Remove target from vars_to_encode if present
    if target in vars_to_encode:
        vars_to_encode.remove(target)

    # --- FIX: High Cardinality for One-Hot ---
    # Group rare categories into 'Other' to reduce dimensionality
    print("   Grouping rare categories (threshold < 1%)...")
    for col in vars_to_encode:
        counts = df[col].value_counts(normalize=True)
        rare_cats = counts[counts < 0.01].index
        if len(rare_cats) > 0:
            df[col] = df[col].replace(rare_cats, "Other")

    # --- Approach 1: Ordinal Encoding ---
    print("   Running Approach 1: Ordinal Encoding...")
    df_ordinal = df.copy()

    # Handle NaNs for Ordinal Encoder (fill with 'Unknown' temporarily)
    for col in vars_to_encode:
        df_ordinal[col] = df_ordinal[col].fillna("Unknown")

    enc = OrdinalEncoder()
    df_ordinal[vars_to_encode] = enc.fit_transform(df_ordinal[vars_to_encode])

    # Encode target
    df_ordinal[target] = df_ordinal[target].astype("category").cat.codes

    # Fill NaNs for evaluation (SimpleImputer-like behavior for the sake of comparison)
    df_ordinal = df_ordinal.fillna(-1)

    # Split
    train_ord, test_ord = train_test_split(
        df_ordinal, test_size=0.3, random_state=42, stratify=df_ordinal[target]
    )

    # Evaluate
    eval_ord = ds.evaluate_approach(
        train_ord.copy(), test_ord.copy(), target=target, metric="f1"
    )
    print(f"      Ordinal F1 (NB, KNN): {eval_ord['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_ord, title="Ordinal Encoding Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_ordinal_eval.png"))
    plt.close()

    # --- Approach 2: One-Hot Encoding ---
    print("   Running Approach 2: One-Hot Encoding...")
    # dslabs dummify
    df_onehot = ds.dummify(df, vars_to_encode)

    # Encode target
    df_onehot[target] = df_onehot[target].astype("category").cat.codes

    # Fill NaNs for evaluation
    df_onehot = df_onehot.fillna(-1)

    # Split
    train_oh, test_oh = train_test_split(
        df_onehot, test_size=0.3, random_state=42, stratify=df_onehot[target]
    )

    # Evaluate
    eval_oh = ds.evaluate_approach(
        train_oh.copy(), test_oh.copy(), target=target, metric="f1"
    )
    print(f"      OneHot F1 (NB, KNN): {eval_oh['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_oh, title="One-Hot Encoding Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_onehot_eval.png"))
    plt.close()

    # --- Comparison & Selection ---
    # Compare average F1 of both models
    avg_ord = sum(eval_ord["f1"]) / 2
    avg_oh = sum(eval_oh["f1"]) / 2

    print(f"   Comparison: Ordinal Avg F1={avg_ord:.4f}, OneHot Avg F1={avg_oh:.4f}")

    # Plot Comparison Chart
    ds.plot_multibar_chart(
        ["NB", "KNN"],
        {"Ordinal": eval_ord["f1"], "OneHot": eval_oh["f1"]},
        title="Encoding Comparison (F1 Score)",
        ylabel="F1 Score",
        percentage=True,
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_comparison.png"))
    plt.close()

    # Plot Side-by-Side Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_ord, title="Ordinal Encoding", ax=axs[0], percentage=True
    )
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_oh, title="One-Hot Encoding", ax=axs[1], percentage=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_side_by_side.png"))
    plt.close()

    best_train = None
    best_test = None
    best_name = ""

    if avg_ord > avg_oh:
        print("   >>> Selected: Ordinal Encoding")
        df_ordinal.to_csv(config.FILE_ENCODED, index=False)
        best_train = train_ord
        best_test = test_ord
        best_name = "Ordinal"
    else:
        print("   >>> Selected: One-Hot Encoding")
        df_onehot.to_csv(config.FILE_ENCODED, index=False)
        best_train = train_oh
        best_test = test_oh
        best_name = "OneHot"

    # Plot Confusion Matrices for Best Approach
    print(f"   Generating Confusion Matrices for {best_name}...")
    trnY = best_train.pop(target).values
    trnX = best_train.values
    tstY = best_test.pop(target).values
    tstX = best_test.values
    labels = pd.unique(tstY)
    labels.sort()

    # NB
    best_nb = ds.run_NB_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_nb:
        prd_nb = best_nb.predict(tstX)
        cm = confusion_matrix(tstY, prd_nb, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.grid(False)
        plt.title(f"Confusion Matrix: {best_name} - Naive Bayes")
        plt.tight_layout()
        plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_best_nb_cm.png"))
        plt.close()

    # KNN
    best_knn = ds.run_KNN_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_knn:
        prd_knn = best_knn.predict(tstX)
        cm = confusion_matrix(tstY, prd_knn, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.grid(False)
        plt.title(f"Confusion Matrix: {best_name} - KNN")
        plt.tight_layout()
        plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_best_knn_cm.png"))
        plt.close()


if __name__ == "__main__":
    run_encoding()
