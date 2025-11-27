"""
STEP 5: Class Balancing
Tests Random Oversampling vs Random Undersampling
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


def run_balancing():
    print("\n" + "=" * 60)
    print("STEP 5: CLASS BALANCING")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_SCALED}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_SCALED)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Check class distribution
    class_counts = y_train.value_counts()
    print(f"   Original class distribution: {class_counts.to_dict()}")
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: RANDOM OVERSAMPLING =====
    print("\n   Approach A: Random Oversampling")
    X_train_a, y_train_a = flights_utils.random_oversampling(X_train.copy(), y_train.copy())
    print(f"      New class distribution: {y_train_a.value_counts().to_dict()}")
    
    eval_over = flights_utils.evaluate_models(X_train_a, y_train_a, X_test, y_test, "Oversampling")
    results['Oversampling'] = eval_over
    datasets['Oversampling'] = (X_train_a, y_train_a, X_test, y_test)
    
    # ===== APPROACH B: RANDOM UNDERSAMPLING =====
    print("\n   Approach B: Random Undersampling")
    X_train_b, y_train_b = flights_utils.random_undersampling(X_train.copy(), y_train.copy())
    print(f"      New class distribution: {y_train_b.value_counts().to_dict()}")
    
    eval_under = flights_utils.evaluate_models(X_train_b, y_train_b, X_test, y_test, "Undersampling")
    results['Undersampling'] = eval_under
    datasets['Undersampling'] = (X_train_b, y_train_b, X_test, y_test)
    
    # --- NEW: Detailed Plotting ---
    print("\n   Generating detailed evaluation charts...")

    # Plot for Oversampling
    ds.plot_multibar_chart(["NB", "KNN"], {"Oversampling": [eval_over['NB'], eval_over['KNN']]}, title="Oversampling Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "05_balancing_oversampling_eval.png"))
    plt.close()

    # Plot for Undersampling
    ds.plot_multibar_chart(["NB", "KNN"], {"Undersampling": [eval_under['NB'], eval_under['KNN']]}, title="Undersampling Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "05_balancing_undersampling_eval.png"))
    plt.close()

    # Plot Side-by-Side Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Oversampling": [eval_over['NB'], eval_over['KNN']]}, 
        ax=axs[0], title="Oversampling", percentage=True
    )
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"Undersampling": [eval_under['NB'], eval_under['KNN']]}, 
        ax=axs[1], title="Undersampling", percentage=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "05_balancing_side_by_side.png"))
    plt.close()

    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "05_balancing_comparison.png")
    flights_utils.plot_comparison(results, "Step 5: Class Balancing", chart_path)
    
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "05_balancing_best_nb_cm.png"))
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "05_balancing_best_knn_cm.png"))
    plt.close()

    # Save best dataset at the end
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_BALANCED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})

    print(f"\n   Step 5 Complete!")


if __name__ == "__main__":
    run_balancing()
