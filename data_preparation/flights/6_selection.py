"""
STEP 6: Feature Selection
Tests Low Variance with different thresholds
"""
import sys
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils import dslabs_functions as ds
import lab3_config as config
import flights_utils


def run_selection():
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE SELECTION")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_BALANCED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_BALANCED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: LOW VARIANCE (0.1) =====
    print("\n   Approach A: Low Variance (threshold=0.1)")
    X_train_a = X_train.copy()
    X_test_a = X_test.copy()
    
    train_temp = pd.concat([X_train_a, y_train], axis=1)
    vars_to_drop_a = ds.select_low_variance_variables(train_temp, max_threshold=0.1, target=config.TARGET)
    
    X_train_a = X_train_a.drop(columns=[c for c in vars_to_drop_a if c in X_train_a.columns])
    X_test_a = X_test_a.drop(columns=[c for c in vars_to_drop_a if c in X_test_a.columns])
    
    print(f"      Dropped {len(vars_to_drop_a)} variables")
    eval_var1 = flights_utils.evaluate_models(X_train_a, y_train, X_test_a, y_test, "Variance(0.1)")
    results['Variance(0.1)'] = eval_var1
    datasets['Variance(0.1)'] = (X_train_a, y_train, X_test_a, y_test)
    
    # ===== APPROACH B: LOW VARIANCE (0.01) =====
    print("\n   Approach B: Low Variance (threshold=0.01)")
    X_train_b = X_train.copy()
    X_test_b = X_test.copy()
    
    train_temp = pd.concat([X_train_b, y_train], axis=1)
    vars_to_drop_b = ds.select_low_variance_variables(train_temp, max_threshold=0.01, target=config.TARGET)
    
    X_train_b = X_train_b.drop(columns=[c for c in vars_to_drop_b if c in X_train_b.columns])
    X_test_b = X_test_b.drop(columns=[c for c in vars_to_drop_b if c in X_test_b.columns])
    
    print(f"      Dropped {len(vars_to_drop_b)} variables")
    eval_var01 = flights_utils.evaluate_models(X_train_b, y_train, X_test_b, y_test, "Variance(0.01)")
    results['Variance(0.01)'] = eval_var01
    datasets['Variance(0.01)'] = (X_train_b, y_train, X_test_b, y_test)
    
    # --- NEW: Detailed Plotting ---
    print("\n   Generating detailed evaluation charts...")

    # Plot for Variance(0.1)
    ds.plot_multibar_chart(["NB", "KNN"], {"Variance(0.1)": [eval_var1['NB'], eval_var1['KNN']]}, title="Low Variance (0.1) Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "06_selection_variance_0.1_eval.png"))
    plt.close()

    # Plot for Variance(0.01)
    ds.plot_multibar_chart(["NB", "KNN"], {"Variance(0.01)": [eval_var01['NB'], eval_var01['KNN']]}, title="Low Variance (0.01) Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "06_selection_variance_0.01_eval.png"))
    plt.close()

    # Plot Side-by-Side Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Variance(0.1)": [eval_var1['NB'], eval_var1['KNN']]}, 
        ax=axs[0], title="Low Variance (0.1)", percentage=True
    )
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Variance(0.01)": [eval_var01['NB'], eval_var01['KNN']]}, 
        ax=axs[1], title="Low Variance (0.01)", percentage=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "06_selection_side_by_side.png"))
    plt.close()

    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "06_selection_comparison.png")
    flights_utils.plot_comparison(results, "Step 6: Feature Selection", chart_path)
    
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "06_selection_best_nb_cm.png"))
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "06_selection_best_knn_cm.png"))
    plt.close()

    # Save best dataset at the end
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_SELECTED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})

    print(f"\n   Step 6 Complete!")
    print(f"\n   FINAL DATASET: {config.FILE_SELECTED}")
    print(f"   Final shape - Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    run_selection()
