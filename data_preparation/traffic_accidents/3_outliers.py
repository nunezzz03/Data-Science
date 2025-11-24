import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import lab3_config as config
import dslabs_functions as ds


def run_outliers():
    print("\n[Step 3] Outliers Treatment...")

    if not os.path.exists(config.FILE_IMPUTED):
        print("Error: Previous step file not found. Run 2_imputation.py first.")
        return

    df = pd.read_csv(config.FILE_IMPUTED)
    target = config.TARGET

    # Identify numeric variables
    numeric_vars = df.select_dtypes(include=["number"]).columns.tolist()
    if target in numeric_vars:
        numeric_vars.remove(target)

    # --- Approach 1: Drop Outliers via IQR (1.5) ---
    print("   Running Approach 1: Drop Outliers (IQR 1.5)...")
    df_iqr = df.copy()

    # Calculate bounds
    summary5 = df_iqr[numeric_vars].describe()
    Q1 = df_iqr[numeric_vars].quantile(0.25)
    Q3 = df_iqr[numeric_vars].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Mask for outliers
    outliers_mask = (
        (df_iqr[numeric_vars] < lower_bound) | (df_iqr[numeric_vars] > upper_bound)
    ).any(axis=1)
    df_iqr = df_iqr[~outliers_mask]

    print(
        f"      Dropped {sum(outliers_mask)} records ({(sum(outliers_mask)/len(df))*100:.2f}%)"
    )

    if len(df_iqr) < 100:
        print(
            "      Warning: Too few records left. Using original df for this approach."
        )
        df_iqr = df.copy()

    # Split
    train_iqr, test_iqr = train_test_split(
        df_iqr, test_size=0.3, random_state=42, stratify=df_iqr[target]
    )

    # Evaluate
    eval_iqr = ds.evaluate_approach(
        train_iqr.copy(), test_iqr.copy(), target=target, metric="f1"
    )
    print(f"      IQR Drop F1 (NB, KNN): {eval_iqr['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"IQR Drop": eval_iqr["f1"]}, title="IQR Drop Outliers F1"
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "3_outliers_iqr_f1.png"))
    plt.close()

    # --- Approach 2: Keep Outliers (Baseline) or Drop via StDev (3) ---
    # Let's do Drop via StDev (3) which is more conservative.
    print("   Running Approach 2: Drop Outliers (StDev 3)...")
    df_std = df.copy()

    mean = df_std[numeric_vars].mean()
    std = df_std[numeric_vars].std()

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    outliers_mask_std = (
        (df_std[numeric_vars] < lower_bound) | (df_std[numeric_vars] > upper_bound)
    ).any(axis=1)
    df_std = df_std[~outliers_mask_std]

    print(
        f"      Dropped {sum(outliers_mask_std)} records ({(sum(outliers_mask_std)/len(df))*100:.2f}%)"
    )

    # Split
    train_std, test_std = train_test_split(
        df_std, test_size=0.3, random_state=42, stratify=df_std[target]
    )

    # Evaluate
    eval_std = ds.evaluate_approach(
        train_std.copy(), test_std.copy(), target=target, metric="f1"
    )
    print(f"      StDev Drop F1 (NB, KNN): {eval_std['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"StDev Drop": eval_std["f1"]}, title="StDev Drop Outliers F1"
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "3_outliers_stdev_f1.png"))
    plt.close()

    # --- Comparison ---
    avg_iqr = sum(eval_iqr["f1"]) / 2
    avg_std = sum(eval_std["f1"]) / 2

    print(f"   Comparison: IQR Avg F1={avg_iqr:.4f}, StDev Avg F1={avg_std:.4f}")

    # Plot Comparison Chart
    ds.plot_multibar_chart(
        ["NB", "KNN"],
        {"IQR Drop": eval_iqr["f1"], "StDev Drop": eval_std["f1"]},
        title="Outliers Comparison (F1 Score)",
        ylabel="F1 Score",
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "3_outliers_comparison.png"))
    plt.close()

    best_train = None
    best_test = None
    best_name = ""

    if avg_iqr > avg_std:
        print("   >>> Selected: IQR Drop")
        df_iqr.to_csv(config.FILE_OUTLIERS, index=False)
        best_train = train_iqr
        best_test = test_iqr
        best_name = "IQR Drop"
    else:
        print("   >>> Selected: StDev Drop")
        df_std.to_csv(config.FILE_OUTLIERS, index=False)
        best_train = train_std
        best_test = test_std
        best_name = "StDev Drop"

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
        plt.savefig(os.path.join(config.IMAGES_DIR, "3_outliers_best_nb_cm.png"))
        plt.close()

    # KNN
    best_knn = ds.run_KNN_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_knn:
        prd_knn = best_knn.predict(tstX)
        cnf_mtx_knn = ds.confusion_matrix(tstY, prd_knn, labels=labels)
        ds.plot_confusion_matrix(cnf_mtx_knn, labels)
        plt.title(f"Confusion Matrix: {best_name} - KNN")
        plt.savefig(os.path.join(config.IMAGES_DIR, "3_outliers_best_knn_cm.png"))
        plt.close()


if __name__ == "__main__":
    run_outliers()
