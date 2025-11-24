import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
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

    # Variables
    vars_types = ds.get_variable_types(df)
    symbolic_vars = vars_types["symbolic"]
    binary_vars = vars_types["binary"]

    # Combine all categorical variables to encode
    vars_to_encode = symbolic_vars + binary_vars

    # Remove target from vars_to_encode if present
    if target in vars_to_encode:
        vars_to_encode.remove(target)

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
        ["NB", "KNN"], {"Ordinal": eval_ord["f1"]}, title="Ordinal Encoding F1"
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_ordinal_f1.png"))
    plt.close()

    # --- Approach 2: One-Hot Encoding ---
    print("   Running Approach 2: One-Hot Encoding...")
    # dslabs dummify
    df_onehot = ds.dummify(df, vars_to_encode)

    # Encode target
    df_onehot[target] = df_onehot[target].astype("category").cat.codes

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
        ["NB", "KNN"], {"OneHot": eval_oh["f1"]}, title="OneHot Encoding F1"
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_onehot_f1.png"))
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
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_comparison.png"))
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
        cnf_mtx_nb = ds.confusion_matrix(tstY, prd_nb, labels=labels)
        ds.plot_confusion_matrix(cnf_mtx_nb, labels)
        plt.title(f"Confusion Matrix: {best_name} - Naive Bayes")
        plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_best_nb_cm.png"))
        plt.close()

    # KNN
    best_knn = ds.run_KNN_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_knn:
        prd_knn = best_knn.predict(tstX)
        cnf_mtx_knn = ds.confusion_matrix(tstY, prd_knn, labels=labels)
        ds.plot_confusion_matrix(cnf_mtx_knn, labels)
        plt.title(f"Confusion Matrix: {best_name} - KNN")
        plt.savefig(os.path.join(config.IMAGES_DIR, "1_encoding_best_knn_cm.png"))
        plt.close()


if __name__ == "__main__":
    run_encoding()
