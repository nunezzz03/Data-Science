import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import lab3_config as config
import dslabs_functions as ds


def run_imputation():
    print("\n[Step 2] Missing Value Imputation...")

    if not os.path.exists(config.FILE_ENCODED):
        print("Error: Previous step file not found. Run 1_encoding.py first.")
        return

    df = pd.read_csv(config.FILE_ENCODED)
    target = config.TARGET

    # Check for missing values
    mv = df.isnull().sum().sum()
    print(f"   Total missing values: {mv}")
    if mv == 0:
        print("   No missing values found. Skipping imputation.")
        df.to_csv(config.FILE_IMPUTED, index=False)
        return

    # --- Approach 1: Frequent/Mean Imputation ---
    print("   Running Approach 1: Frequent/Mean Imputation...")
    df_frequent = ds.mvi_by_filling(df, strategy="frequent")

    # Split
    train_freq, test_freq = train_test_split(
        df_frequent, test_size=0.3, random_state=42, stratify=df_frequent[target]
    )

    # Evaluate
    eval_freq = ds.evaluate_approach(
        train_freq.copy(), test_freq.copy(), target=target, metric="f1"
    )
    print(f"      Frequent F1 (NB, KNN): {eval_freq['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Frequent": eval_freq["f1"]}, title="Frequent Imputation F1"
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "2_imputation_frequent_f1.png"))
    plt.close()

    # --- Approach 2: KNN Imputation ---
    print("   Running Approach 2: KNN Imputation...")
    df_knn = ds.mvi_by_filling(df, strategy="knn")

    # Split
    train_knn, test_knn = train_test_split(
        df_knn, test_size=0.3, random_state=42, stratify=df_knn[target]
    )

    # Evaluate
    eval_knn = ds.evaluate_approach(
        train_knn.copy(), test_knn.copy(), target=target, metric="f1"
    )
    print(f"      KNN F1 (NB, KNN): {eval_knn['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"KNN": eval_knn["f1"]}, title="KNN Imputation F1"
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "2_imputation_knn_f1.png"))
    plt.close()

    # --- Comparison ---
    avg_freq = sum(eval_freq["f1"]) / 2
    avg_knn = sum(eval_knn["f1"]) / 2

    print(f"   Comparison: Frequent Avg F1={avg_freq:.4f}, KNN Avg F1={avg_knn:.4f}")

    # Plot Comparison Chart
    ds.plot_multibar_chart(
        ["NB", "KNN"],
        {"Frequent": eval_freq["f1"], "KNN": eval_knn["f1"]},
        title="Imputation Comparison (F1 Score)",
        ylabel="F1 Score",
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "2_imputation_comparison.png"))
    plt.close()

    best_train = None
    best_test = None
    best_name = ""

    if avg_freq > avg_knn:
        print("   >>> Selected: Frequent Imputation")
        df_frequent.to_csv(config.FILE_IMPUTED, index=False)
        best_train = train_freq
        best_test = test_freq
        best_name = "Frequent"
    else:
        print("   >>> Selected: KNN Imputation")
        df_knn.to_csv(config.FILE_IMPUTED, index=False)
        best_train = train_knn
        best_test = test_knn
        best_name = "KNN"

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
        plt.savefig(os.path.join(config.IMAGES_DIR, "2_imputation_best_nb_cm.png"))
        plt.close()

    # KNN
    best_knn = ds.run_KNN_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_knn:
        prd_knn = best_knn.predict(tstX)
        cnf_mtx_knn = ds.confusion_matrix(tstY, prd_knn, labels=labels)
        ds.plot_confusion_matrix(cnf_mtx_knn, labels)
        plt.title(f"Confusion Matrix: {best_name} - KNN")
        plt.savefig(os.path.join(config.IMAGES_DIR, "2_imputation_best_knn_cm.png"))
        plt.close()


if __name__ == "__main__":
    run_imputation()
