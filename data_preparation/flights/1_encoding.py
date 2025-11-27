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


def apply_cyclical_encoding(df):
    """Aplica a codificação cíclica (seno/cosseno) às variáveis de tempo."""
    if "CRSDepTime" in df.columns:
        df["CRSDepTime_sin"] = np.sin(2 * np.pi * df["CRSDepTime"] / 2400) + 1
        df["CRSDepTime_cos"] = np.cos(2 * np.pi * df["CRSDepTime"] / 2400) + 1
        df = df.drop(columns=["CRSDepTime"])

    if "CRSArrTime" in df.columns:
        df["CRSArrTime_sin"] = np.sin(2 * np.pi * df["CRSArrTime"] / 2400) + 1
        df["CRSArrTime_cos"] = np.cos(2 * np.pi * df["CRSArrTime"] / 2400) + 1
        df = df.drop(columns=["CRSArrTime"])

    if "DayOfWeek" in df.columns:
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7) + 1
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7) + 1
        df = df.drop(columns=["DayOfWeek"])

    if "Month" in df.columns:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12) + 1
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12) + 1
        df = df.drop(columns=["Month"])
    return df


def run_encoding():
    print("\n" + "=" * 60)
    print("STEP 1: VARIABLE ENCODING")
    print("=" * 60)

    # Load raw data
    print(f"\n   Loading data from {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH, na_values="", parse_dates=True)

    # Sample for performance
    df = df.sample(frac=config.SAMPLE_FRAC, random_state=config.RANDOM_STATE)
    print(f"   Sampled {config.SAMPLE_FRAC*100}%: {df.shape}")

    # Remove leakage columns
    df = df.drop(columns=[c for c in config.LEAKAGE_COLS if c in df.columns])
    print(f"   Removed leakage columns: {df.shape}")

    # Get variable types
    var_types = ds.get_variable_types(df)
    symbolic_vars = var_types["symbolic"]
    print(f"   Symbolic variables to encode: {len(symbolic_vars)}")

    # --- NEW: Create a sample with a target of 30% positive cases ---
    print("\n   Creating a sample with ~30% positive cases...")
    df_true = df[df[config.TARGET] == True]
    df_false = df[df[config.TARGET] == False]

    # Calculate sample sizes
    target_ratio = 0.30
    # Keep all positive cases if they are fewer than the target, otherwise sample them
    n_true = len(df_true)
    # Calculate how many negative cases we need for a 20% ratio
    n_false = int(n_true * (1 - target_ratio) / target_ratio)

    # Sample the negative cases
    df_false_sample = df_false.sample(n=n_false, random_state=config.RANDOM_STATE)

    # Combine to create the final sample dataframe
    df_sample = pd.concat([df_true, df_false_sample])
    print(
        f"   New sample size: {len(df_sample)} records with {len(df_true)} positive cases."
    )

    df_sample = apply_cyclical_encoding(df_sample)

    # Now, split this artificially balanced sample into train and test sets
    y_sample = df_sample[config.TARGET]
    X_sample = df_sample.drop(columns=[config.TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        train_size=config.TRAIN_SIZE,
        stratify=y_sample,
        random_state=config.RANDOM_STATE,
    )

    target = config.TARGET

    df = df_sample.copy()

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
