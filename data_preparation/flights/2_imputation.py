"""
STEP 2: Missing Value Imputation
Tests Most Frequent vs KNN strategies
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


def run_imputation():
    print("\n" + "=" * 60)
    print("STEP 2: MISSING VALUE IMPUTATION")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_ENCODED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_ENCODED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # --- FIX: Convert boolean columns to integer to avoid SimpleImputer error ---
    bool_cols_train = X_train.select_dtypes(include='bool').columns
    X_train[bool_cols_train] = X_train[bool_cols_train].astype(int)
    
    bool_cols_test = X_test.select_dtypes(include='bool').columns
    X_test[bool_cols_test] = X_test[bool_cols_test].astype(int)

    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: MOST FREQUENT =====
    print("\n   Approach A: Most Frequent")
    train_idx, test_idx = X_train.index, X_test.index
    
    X_train_a = ds.mvi_by_filling(X_train.copy(), strategy="frequent")
    X_test_a = ds.mvi_by_filling(X_test.copy(), strategy="frequent")
    
    # --- FIX: Ensure column order is the same after imputation ---
    X_train_a = X_train_a[X_train.columns]
    X_test_a = X_test_a[X_train.columns]

    X_train_a.index = train_idx
    X_test_a.index = test_idx
    
    eval_freq = flights_utils.evaluate_models(X_train_a, y_train, X_test_a, y_test, "Frequent")
    results['Frequent'] = eval_freq
    datasets['Frequent'] = (X_train_a, y_train, X_test_a, y_test)
    
    # ===== APPROACH B: KNN IMPUTATION =====
    print("\n   Approach B: KNN Imputation (pode ser lento)")
    X_train_b = ds.mvi_by_filling(X_train.copy(), strategy="knn")
    X_test_b = ds.mvi_by_filling(X_test.copy(), strategy="knn")
    
    # --- FIX: Ensure column order is the same after imputation ---
    X_train_b = X_train_b[X_train.columns]
    X_test_b = X_test_b[X_train.columns]

    X_train_b.index = train_idx
    X_test_b.index = test_idx
    
    eval_knn = flights_utils.evaluate_models(X_train_b, y_train, X_test_b, y_test, "KNN")
    results['KNN'] = eval_knn
    datasets['KNN'] = (X_train_b, y_train, X_test_b, y_test)
    
    # --- NEW: Detailed Plotting ---
    print("\n   Generating detailed evaluation charts...")

    # Plot for Frequent
    ds.plot_multibar_chart(["NB", "KNN"], {"Frequent": [eval_freq['NB'], eval_freq['KNN']]}, title="Frequent Imputation Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "02_imputation_frequent_eval.png"))
    plt.close()

    # Plot for KNN
    ds.plot_multibar_chart(["NB", "KNN"], {"KNN": [eval_knn['NB'], eval_knn['KNN']]}, title="KNN Imputation Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "02_imputation_knn_eval.png"))
    plt.close()

    # Plot Side-by-Side Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Frequent": [eval_freq['NB'], eval_freq['KNN']]}, 
        ax=axs[0], title="Frequent Imputation", percentage=True
    )
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"KNN": [eval_knn['NB'], eval_knn['KNN']]}, 
        ax=axs[1], title="KNN Imputation", percentage=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "02_imputation_side_by_side.png"))
    plt.close()

    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Save best dataset
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "02_imputation_comparison.png")
    flights_utils.plot_comparison(results, "Step 2: Missing Value Imputation", chart_path)
    
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "02_imputation_best_nb_cm.png"))
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "02_imputation_best_knn_cm.png"))
    plt.close()
    
    # Save best dataset at the end
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_IMPUTED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})

    print(f"\n   Step 2 Complete!")


if __name__ == "__main__":
    run_imputation()
