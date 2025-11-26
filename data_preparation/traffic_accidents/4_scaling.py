import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lab3_config as config
import dslabs_functions as ds


def run_scaling():
    print("\n[Step 4] Scaling...")

    if not os.path.exists(config.FILE_OUTLIERS):
        print("Error: Previous step file not found. Run 3_outliers.py first.")
        return

    df = pd.read_csv(config.FILE_OUTLIERS)
    target = config.TARGET

    # Separate X and y
    X = df.drop(columns=[target])
    y = df[target]

    # --- Approach 1: StandardScaler ---
    print("   Running Approach 1: StandardScaler (Z-Score)...")
    scaler_std = StandardScaler()
    X_std = pd.DataFrame(scaler_std.fit_transform(X), columns=X.columns)
    # Reattach target for splitting/saving
    df_std = pd.concat([X_std, y], axis=1)

    train_std, test_std = train_test_split(
        df_std, test_size=0.3, random_state=42, stratify=df_std[target]
    )

    eval_std = ds.evaluate_approach(
        train_std.copy(), test_std.copy(), target=target, metric="f1"
    )
    print(f"      StandardScaler F1 (NB, KNN): {eval_std['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_std, title="StandardScaler Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "4_scaling_standard_eval.png"))
    plt.close()

    # --- Approach 2: MinMaxScaler ---
    print("   Running Approach 2: MinMaxScaler (0-1)...")
    scaler_mm = MinMaxScaler()
    X_mm = pd.DataFrame(scaler_mm.fit_transform(X), columns=X.columns)
    df_mm = pd.concat([X_mm, y], axis=1)

    train_mm, test_mm = train_test_split(
        df_mm, test_size=0.3, random_state=42, stratify=df_mm[target]
    )

    # Evaluate
    eval_mm = ds.evaluate_approach(
        train_mm.copy(), test_mm.copy(), target=target, metric="f1"
    )
    print(f"      MinMaxScaler F1 (NB, KNN): {eval_mm['f1']}")
    ds.plot_multibar_chart(
        ["NB", "KNN"], eval_mm, title="MinMaxScaler Evaluation", percentage=True
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "4_scaling_minmax_eval.png"))
    plt.close()

    # --- Visualize Scaling Effect (Boxplots) ---
    # Select variables to plot (exclude binary/one-hot for clarity)
    vars_to_plot = [c for c in X.columns if X[c].nunique() > 2]

    if vars_to_plot:
        print(f"   Generating Boxplots for {len(vars_to_plot)} continuous variables...")

        fig, axs = plt.subplots(1, 3, figsize=(20, 10), squeeze=False)

        # Original
        axs[0, 0].set_title("Original Data")
        X[vars_to_plot].boxplot(ax=axs[0, 0], rot=90)
        axs[0, 0].grid(False)

        # Standard
        axs[0, 1].set_title("Z-score normalization")
        X_std[vars_to_plot].boxplot(ax=axs[0, 1], rot=90)
        axs[0, 1].grid(False)

        # MinMax
        axs[0, 2].set_title("MinMax normalization")
        X_mm[vars_to_plot].boxplot(ax=axs[0, 2], rot=90)
        axs[0, 2].grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(config.IMAGES_DIR, "4_scaling_boxplots.png"))
        plt.close()

    # --- Comparison ---
    avg_std = sum(eval_std["f1"]) / 2
    avg_mm = sum(eval_mm["f1"]) / 2

    print(f"   Comparison: Standard Avg F1={avg_std:.4f}, MinMax Avg F1={avg_mm:.4f}")

    # Plot Comparison Chart
    ds.plot_multibar_chart(
        ["NB", "KNN"],
        {"Standard": eval_std["f1"], "MinMax": eval_mm["f1"]},
        title="Scaling Comparison (F1 Score)",
        ylabel="F1 Score",
        percentage=True,
    )
    plt.savefig(os.path.join(config.IMAGES_DIR, "4_scaling_comparison.png"))
    plt.close()

    best_train = None
    best_test = None
    best_name = ""

    if avg_std > avg_mm:
        print("   >>> Selected: StandardScaler")
        df_std.to_csv(config.FILE_SCALED, index=False)
        best_train = train_std
        best_test = test_std
        best_name = "StandardScaler"
    else:
        print("   >>> Selected: MinMaxScaler")
        df_mm.to_csv(config.FILE_SCALED, index=False)
        best_train = train_mm
        best_test = test_mm
        best_name = "MinMaxScaler"

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
        plt.savefig(os.path.join(config.IMAGES_DIR, "4_scaling_best_nb_cm.png"))
        plt.close()

    # KNN
    best_knn = ds.run_KNN_model(trnX, trnY, tstX, tstY, metric="f1")
    if best_knn:
        prd_knn = best_knn.predict(tstX)
        cnf_mtx_knn = ds.confusion_matrix(tstY, prd_knn, labels=labels)
        ds.plot_confusion_matrix(cnf_mtx_knn, labels)
        plt.title(f"Confusion Matrix: {best_name} - KNN")
        plt.savefig(os.path.join(config.IMAGES_DIR, "4_scaling_best_knn_cm.png"))
        plt.close()


if __name__ == "__main__":
    run_scaling()
