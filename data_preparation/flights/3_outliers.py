"""
STEP 3: Outlier Treatment
Tests Standard Deviation vs IQR methods
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import dslabs_functions as ds
import lab3_config as config
import flights_utils


def run_outliers():
    print("\n" + "=" * 60)
    print("STEP 3: OUTLIER TREATMENT")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_IMPUTED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_IMPUTED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Get numeric variables
    numeric_vars = ds.get_variable_types(X_train)["numeric"]
    print(f"   Numeric variables: {len(numeric_vars)}")
    
    if len(numeric_vars) == 0:
        print("   SKIPPED: No numeric variables for outlier removal")
        flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_OUTLIERS)
        return
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: STANDARD DEVIATION (2 std) =====
    print("\n   Approach A: Std-based (2 std)")
    X_train_a = X_train.copy()
    y_train_a = y_train.copy()
    
    summary5 = X_train_a[numeric_vars].describe()
    outlier_indices = []
    
    for var in numeric_vars:
        top, bottom = ds.determine_outlier_thresholds_for_var(summary5[var], std_based=True, threshold=2)
        outliers = X_train_a[(X_train_a[var] > top) | (X_train_a[var] < bottom)]
        outlier_indices.extend(outliers.index.tolist())
    
    outlier_indices = list(set(outlier_indices))
    X_train_a = X_train_a.drop(outlier_indices, axis=0)
    y_train_a = y_train_a.drop(outlier_indices, axis=0)
    
    print(f"      Removed {len(outlier_indices)} outlier records")
    eval_std = flights_utils.evaluate_models(X_train_a, y_train_a, X_test, y_test, "Std-based")
    results['Std-based'] = eval_std
    datasets['Std-based'] = (X_train_a, y_train_a, X_test, y_test)
    
    # ===== APPROACH B: IQR (1.5 IQR) =====
    print("\n   Approach B: IQR-based (1.5 IQR)")
    X_train_b = X_train.copy()
    y_train_b = y_train.copy()
    
    summary5 = X_train_b[numeric_vars].describe()
    outlier_indices = []
    
    for var in numeric_vars:
        top, bottom = ds.determine_outlier_thresholds_for_var(summary5[var], std_based=False, threshold=1.5)
        outliers = X_train_b[(X_train_b[var] > top) | (X_train_b[var] < bottom)]
        outlier_indices.extend(outliers.index.tolist())
    
    outlier_indices = list(set(outlier_indices))
    X_train_b = X_train_b.drop(outlier_indices, axis=0)
    y_train_b = y_train_b.drop(outlier_indices, axis=0)
    
    print(f"      Removed {len(outlier_indices)} outlier records")
    eval_iqr = flights_utils.evaluate_models(X_train_b, y_train_b, X_test, y_test, "IQR-based")
    results['IQR-based'] = eval_iqr
    datasets['IQR-based'] = (X_train_b, y_train_b, X_test, y_test)

    # --- NEW: Detailed Plotting ---
    print("\n   Generating detailed evaluation charts...")

    # Plot for Std-based
    ds.plot_multibar_chart(["NB", "KNN"], {"Std-based": [eval_std['NB'], eval_std['KNN']]}, title="Std-based Outlier Removal", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "03_outliers_std_eval.png"))
    plt.close()

    # Plot for IQR-based
    ds.plot_multibar_chart(["NB", "KNN"], {"IQR-based": [eval_iqr['NB'], eval_iqr['KNN']]}, title="IQR-based Outlier Removal", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "03_outliers_iqr_eval.png"))
    plt.close()

    # Plot Side-by-Side Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Std-based": [eval_std['NB'], eval_std['KNN']]}, 
        ax=axs[0], title="Std-based Outlier Removal", percentage=True
    )
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"IQR-based": [eval_iqr['NB'], eval_iqr['KNN']]}, 
        ax=axs[1], title="IQR-based Outlier Removal", percentage=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "03_outliers_side_by_side.png"))
    plt.close()
    
    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")

    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "03_outliers_comparison.png")
    flights_utils.plot_comparison(results, "Step 3: Outlier Treatment", chart_path)
    
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "03_outliers_best_nb_cm.png"))
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "03_outliers_best_knn_cm.png"))
    plt.close()

    # Save best dataset at the end
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_OUTLIERS,
                 metadata={'approach': best_approach, 'f1_score': best_f1})

    print(f"\n   Step 3 Complete!")


if __name__ == "__main__":
    run_outliers()
