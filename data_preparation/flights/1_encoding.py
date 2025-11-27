"""
STEP 1: Variable Encoding
Tests One-Hot vs Ordinal Encoding with detailed plotting.
"""
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Add project root and utils to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
utils_path = os.path.join(project_root, 'utils')
sys.path.insert(0, project_root)
sys.path.insert(0, utils_path)
sys.path.insert(0, os.path.dirname(__file__))

from utils import dslabs_functions as ds
import lab3_config as config
import flights_utils


def apply_cyclical_encoding(df):
    """Aplica a codificação cíclica (seno/cosseno) às variáveis de tempo."""
    if "CRSDepTime" in df.columns:
        df["CRSDepTime_sin"] = np.sin(2 * np.pi * df["CRSDepTime"] / 2400)
        df["CRSDepTime_cos"] = np.cos(2 * np.pi * df["CRSDepTime"] / 2400)
        df = df.drop(columns=["CRSDepTime"])
        
    if "CRSArrTime" in df.columns:
        df["CRSArrTime_sin"] = np.sin(2 * np.pi * df["CRSArrTime"] / 2400)
        df["CRSArrTime_cos"] = np.cos(2 * np.pi * df["CRSArrTime"] / 2400)
        df = df.drop(columns=["CRSArrTime"])
        
    if "DayOfWeek" in df.columns:
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
        df = df.drop(columns=["DayOfWeek"])

    if "Month" in df.columns:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df = df.drop(columns=["Month"])
    return df

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
    print(f"   New sample size: {len(df_sample)} records with {len(df_true)} positive cases.")

    # Now, split this artificially balanced sample into train and test sets
    y_sample = df_sample[config.TARGET]
    X_sample = df_sample.drop(columns=[config.TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, train_size=config.TRAIN_SIZE, stratify=y_sample, random_state=config.RANDOM_STATE
    )

    # Store results for comparison
    results = {}
    datasets = {}

    # ===== APPROACH A: ONE-HOT ENCODING =====
    print("\n   Approach A: One-Hot Encoding")
    X_train_a = ds.dummify(X_train.copy(), symbolic_vars)
    X_test_a = ds.dummify(X_test.copy(), symbolic_vars)
    # Align columns after dummify
    train_cols = X_train_a.columns
    test_cols = X_test_a.columns
    X_test_a = X_test_a.reindex(columns=train_cols, fill_value=False)

    eval_oh = flights_utils.evaluate_models(X_train_a, y_train, X_test_a, y_test, "One-Hot")
    results['One-Hot'] = eval_oh
    datasets['One-Hot'] = (X_train_a, y_train, X_test_a, y_test)
    
    # ===== APPROACH B: ORDINAL ENCODING =====
    print("\n   Approach B: Ordinal Encoding")
    X_train_b = X_train.copy()
    X_test_b = X_test.copy()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_b[symbolic_vars] = X_train_b[symbolic_vars].astype(str)
    X_test_b[symbolic_vars] = X_test_b[symbolic_vars].astype(str)

    X_train_b[symbolic_vars] = enc.fit_transform(X_train_b[symbolic_vars])
    X_test_b[symbolic_vars] = enc.transform(X_test_b[symbolic_vars])

    eval_ord = flights_utils.evaluate_models(X_train_b, y_train, X_test_b, y_test, "Ordinal")
    results['Ordinal'] = eval_ord
    datasets['Ordinal'] = (X_train_b, y_train, X_test_b, y_test)
    
    # --- NEW: Detailed Plotting ---
    print("\n   Generating detailed evaluation charts...")

    # Plot for One-Hot
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"One-Hot": [eval_oh['NB'], eval_oh['KNN']]}, 
        title="One-Hot Encoding Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "01_encoding_onehot_eval.png"))
    plt.close()

    # Plot for Ordinal
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Ordinal": [eval_ord['NB'], eval_ord['KNN']]}, 
        title="Ordinal Encoding Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "01_encoding_ordinal_eval.png"))
    plt.close()

    # Plot Side-by-Side Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"One-Hot": [eval_oh['NB'], eval_oh['KNN']]}, 
        ax=axs[0], title="One-Hot Encoding", percentage=True
    )
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Ordinal": [eval_ord['NB'], eval_ord['KNN']]}, 
        ax=axs[1], title="Ordinal Encoding", percentage=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "01_encoding_side_by_side.png"))
    plt.close()

    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "01_encoding_comparison.png")
    flights_utils.plot_comparison(results, "Step 1: Variable Encoding", chart_path)
    
    # --- Plot Confusion Matrices for Best Approach ---
    print(f"   Generating Confusion Matrices for {best_approach}...")
    X_train, y_train, X_test, y_test = datasets[best_approach]
    trnY = y_train.values
    trnX = X_train.values
    tstY = y_test.values
    tstX = X_test.values
    labels = y_train.unique()
    labels.sort()

    # NB
    nb_model = flights_utils.GaussianNB()
    nb_model.fit(trnX, trnY)
    prd_nb = nb_model.predict(tstX)
    cm = confusion_matrix(tstY, prd_nb, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.grid(False)
    plt.title(f"CM: {best_approach} - Naive Bayes")
    plt.savefig(os.path.join(config.IMAGES_DIR, "01_encoding_best_nb_cm.png"))
    plt.close()

    # KNN
    knn_model = flights_utils.KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBORS)
    knn_model.fit(trnX, trnY)
    prd_knn = knn_model.predict(tstX)
    cm = confusion_matrix(tstY, prd_knn, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.grid(False)
    plt.title(f"CM: {best_approach} - KNN")
    plt.savefig(os.path.join(config.IMAGES_DIR, "01_encoding_best_knn_cm.png"))
    plt.close()
    
    # Save best dataset at the end
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_ENCODED, 
                 metadata={'approach': best_approach, 'f1_score': best_f1})

    print(f"\n   Step 1 Complete!")


if __name__ == "__main__":
    run_encoding()
