import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import lab3_config as config
import dslabs_functions as ds


def run_balancing():
    print("\n[Step 5] Balancing...")

    if not os.path.exists(config.FILE_SCALED):
        print("Error: Previous step file not found. Run 4_scaling.py first.")
        return

    df = pd.read_csv(config.FILE_SCALED)
    target = config.TARGET

    # Check current balance
    print("   Original Class Distribution:")
    print(df[target].value_counts(normalize=True))

    # Separate X and y
    X = df.drop(columns=[target])
    y = df[target]

    # --- Approach 1: SMOTE (Oversampling) ---
    print("   Running Approach 1: SMOTE...")
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    df_smote = pd.concat([X_smote, y_smote], axis=1)

    train_smote, test_smote = train_test_split(
        df_smote, test_size=0.3, random_state=42, stratify=df_smote[target]
    )

    eval_smote = ds.evaluate_approach(
        train_smote.copy(), test_smote.copy(), target=target, metric="f1"
    )
    print(f"      SMOTE F1 (NB, KNN): {eval_smote['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_smote, title="SMOTE Balancing Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "5_balancing_smote_eval.png"))
    plt.close()

    # --- Approach 2: Random Undersampling ---
    print("   Running Approach 2: Random Undersampling...")
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    df_rus = pd.concat([X_rus, y_rus], axis=1)

    train_rus, test_rus = train_test_split(
        df_rus, test_size=0.3, random_state=42, stratify=df_rus[target]
    )

    # Evaluate
    eval_rus = ds.evaluate_approach(
        train_rus.copy(), test_rus.copy(), target=target, metric="f1"
    )
    print(f"      Undersampling F1 (NB, KNN): {eval_rus['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_rus, title="Undersampling Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "5_balancing_under_eval.png"))
    plt.close()

    # --- Comparison ---
    avg_smote = sum(eval_smote["f1"]) / 2
    avg_rus = sum(eval_rus["f1"]) / 2

    print(f"   Comparison: SMOTE Avg F1={avg_smote:.4f}, Under Avg F1={avg_rus:.4f}")

    # Plot Comparison Chart
    ds.plot_multibar_chart(
        ["NB", "KNN"],
        {"SMOTE": eval_smote["f1"], "Under": eval_rus["f1"]},
        title="Balancing Comparison (F1 Score)",
        ylabel="F1 Score",
        percentage=True,
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "5_balancing_comparison.png"))
    plt.close()

    best_train = None
    best_test = None
    best_name = ""

    if avg_smote > avg_rus:
        print("   >>> Selected: SMOTE")
        df_smote.to_csv(config.FILE_BALANCED, index=False)
        best_train = train_smote
        best_test = test_smote
        best_name = "SMOTE"
    else:
        print("   >>> Selected: Undersampling")
        df_rus.to_csv(config.FILE_BALANCED, index=False)
        best_train = train_rus
        best_test = test_rus
        best_name = "Undersampling"

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
        plt.savefig(os.path.join(config.IMAGES_DIR, "5_balancing_best_nb_cm.png"))
        plt.close()

    # KNN
    best_knn = ds.run_KNN_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_knn:
        prd_knn = best_knn.predict(tstX)
        cnf_mtx_knn = ds.confusion_matrix(tstY, prd_knn, labels=labels)
        ds.plot_confusion_matrix(cnf_mtx_knn, labels)
        plt.title(f"Confusion Matrix: {best_name} - KNN")
        plt.savefig(os.path.join(config.IMAGES_DIR, "5_balancing_best_knn_cm.png"))
        plt.close()


if __name__ == "__main__":
    run_balancing()
