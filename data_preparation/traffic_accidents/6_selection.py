import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import lab3_config as config
import dslabs_functions as ds


def run_selection():
    print("\n[Step 6] Feature Selection...")

    if not os.path.exists(config.FILE_BALANCED):
        print("Error: Previous step file not found. Run 5_balancing.py first.")
        return

    df = pd.read_csv(config.FILE_BALANCED)
    target = config.TARGET

    train, test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[target]
    )

    # --- Approach 1: Low Variance Filter ---
    print("   Running Approach 1: Low Variance Filter...")

    # Generate variance study plot and get results
    results_var = ds.study_variance_for_feature_selection(
        train,
        test,
        target=target,
        max_threshold=0.5,
        lag=0.05,
        metric="f1",
        file_tag="traffic_accidents",
    )

    # Move the generated plot to images dir
    import shutil

    try:
        shutil.move(
            "traffic_accidents_fs_low_var_f1_study.png",
            os.path.join(config.IMAGES_DIR, "6_selection_variance_f1.png"),
        )
    except Exception as e:
        print(f"Warning: Could not move variance study plot: {e}")

    # Find best threshold from results
    max_threshold = 0.5
    lag = 0.05
    import math

    options_var = [
        round(i * lag, 3) for i in range(1, math.ceil(max_threshold / lag + lag))
    ]

    best_thresh_var = 0
    best_score_var = 0

    # Calculate average score for each threshold
    for i in range(len(options_var)):
        if i < len(results_var["NB"]) and i < len(results_var["KNN"]):
            score = (results_var["NB"][i] + results_var["KNN"][i]) / 2
            if score > best_score_var:
                best_score_var = score
                best_thresh_var = options_var[i]

    print(
        f"      Best Variance Threshold: {best_thresh_var} (Avg F1: {best_score_var:.4f})"
    )

    # Apply best variance threshold
    vars2drop_var = ds.select_low_variance_variables(
        df, max_threshold=best_thresh_var, target=target
    )
    print(f"      Dropping {len(vars2drop_var)} variables: {vars2drop_var}")

    df_var = df.drop(columns=vars2drop_var)
    train_var, test_var = train_test_split(
        df_var, test_size=0.3, random_state=42, stratify=df_var[target]
    )
    eval_var = ds.evaluate_approach(
        train_var.copy(), test_var.copy(), target=target, metric="f1"
    )

    # --- Approach 2: Redundancy (Correlation) Filter ---
    print("   Running Approach 2: Redundancy Filter...")

    results_red = ds.study_redundancy_for_feature_selection(
        train,
        test,
        target=target,
        min_threshold=0.85,
        lag=0.02,
        metric="f1",
        file_tag="traffic_accidents",
    )

    # Move the generated plot to images dir
    try:
        shutil.move(
            "traffic_accidents_fs_redundancy_f1_study.png",
            os.path.join(config.IMAGES_DIR, "6_selection_redundancy_f1.png"),
        )
    except Exception as e:
        print(f"Warning: Could not move redundancy study plot: {e}")

    # Reconstruct options
    min_threshold = 0.85
    lag = 0.02
    options_red = [
        round(min_threshold + i * lag, 3)
        for i in range(math.ceil((1 - min_threshold) / lag) + 1)
    ]

    best_thresh_red = 0
    best_score_red = 0

    for i in range(len(options_red)):
        if i < len(results_red["NB"]) and i < len(results_red["KNN"]):
            score = (results_red["NB"][i] + results_red["KNN"][i]) / 2
            if score > best_score_red:
                best_score_red = score
                best_thresh_red = options_red[i]

    print(
        f"      Best Redundancy Threshold: {best_thresh_red} (Avg F1: {best_score_red:.4f})"
    )

    vars2drop_red = ds.select_redundant_variables(
        df, min_threshold=best_thresh_red, target=target
    )
    print(f"      Dropping {len(vars2drop_red)} variables: {vars2drop_red}")

    df_red = df.drop(columns=vars2drop_red)
    train_red, test_red = train_test_split(
        df_red, test_size=0.3, random_state=42, stratify=df_red[target]
    )
    eval_red = ds.evaluate_approach(
        train_red.copy(), test_red.copy(), target=target, metric="f1"
    )

    # --- Comparison ---
    avg_var = sum(eval_var["f1"]) / 2
    avg_red = sum(eval_red["f1"]) / 2

    print(
        f"   Comparison: Variance Avg F1={avg_var:.4f}, Redundancy Avg F1={avg_red:.4f}"
    )

    # Plot Comparison Chart
    ds.plot_multibar_chart(
        ["NB", "KNN"],
        {"Variance": eval_var["f1"], "Redundancy": eval_red["f1"]},
        title="Feature Selection Comparison (F1 Score)",
        ylabel="F1 Score",
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "6_selection_comparison.png"))
    plt.close()

    best_train = None
    best_test = None
    best_name = ""

    if avg_var > avg_red:
        print("   >>> Selected: Variance Filter")
        df_var.to_csv(config.FILE_SELECTED, index=False)
        best_train = train_var
        best_test = test_var
        best_name = "Variance Filter"
    else:
        print("   >>> Selected: Redundancy Filter")
        df_red.to_csv(config.FILE_SELECTED, index=False)
        best_train = train_red
        best_test = test_red
        best_name = "Redundancy Filter"

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
        plt.savefig(os.path.join(config.IMAGES_DIR, "6_selection_best_nb_cm.png"))
        plt.close()

    # KNN
    best_knn = ds.run_KNN_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_knn:
        prd_knn = best_knn.predict(tstX)
        cnf_mtx_knn = ds.confusion_matrix(tstY, prd_knn, labels=labels)
        ds.plot_confusion_matrix(cnf_mtx_knn, labels)
        plt.title(f"Confusion Matrix: {best_name} - KNN")
        plt.savefig(os.path.join(config.IMAGES_DIR, "6_selection_best_knn_cm.png"))
        plt.close()


if __name__ == "__main__":
    run_selection()
