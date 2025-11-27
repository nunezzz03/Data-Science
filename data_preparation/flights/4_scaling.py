"""
STEP 4: Feature Scaling
Tests StandardScaler vs MinMaxScaler
"""
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from utils import dslabs_functions as ds
import lab3_config as config
import flights_utils


def run_scaling():
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE SCALING")
    print("=" * 60)
    
    # Load data from previous step
    print(f"\n   Loading data from {config.FILE_OUTLIERS}")
    X_train, y_train, X_test, y_test = flights_utils.load_dataset(config.FILE_OUTLIERS)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Get numeric variables
    numeric_vars = ds.get_variable_types(X_train)["numeric"]
    print(f"   Numeric variables: {len(numeric_vars)}")
    
    if len(numeric_vars) == 0:
        print("   SKIPPED: No numeric variables for scaling")
        flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_SCALED)
        return
    
    # Store results
    results = {}
    datasets = {}
    
    # ===== APPROACH A: STANDARD SCALER =====
    print("\n   Approach A: StandardScaler")
    X_train_a = X_train.copy()
    X_test_a = X_test.copy()
    
    scaler = StandardScaler()
    X_train_a[numeric_vars] = scaler.fit_transform(X_train_a[numeric_vars])
    X_test_a[numeric_vars] = scaler.transform(X_test_a[numeric_vars])
    
    eval_std = flights_utils.evaluate_models(X_train_a, y_train, X_test_a, y_test, "StandardScaler")
    results['StandardScaler'] = eval_std
    datasets['StandardScaler'] = (X_train_a, y_train, X_test_a, y_test)
    
    # ===== APPROACH B: MINMAX SCALER =====
    print("\n   Approach B: MinMaxScaler")
    X_train_b = X_train.copy()
    X_test_b = X_test.copy()
    
    scaler = MinMaxScaler()
    X_train_b[numeric_vars] = scaler.fit_transform(X_train_b[numeric_vars])
    X_test_b[numeric_vars] = scaler.transform(X_test_b[numeric_vars])
    
    eval_mm = flights_utils.evaluate_models(X_train_b, y_train, X_test_b, y_test, "MinMaxScaler")
    results['MinMaxScaler'] = eval_mm
    datasets['MinMaxScaler'] = (X_train_b, y_train, X_test_b, y_test)
    
    # --- NEW: Detailed Plotting ---
    print("\n   Generating detailed evaluation charts...")

    # Plot for StandardScaler
    ds.plot_multibar_chart(["NB", "KNN"], {"StandardScaler": [eval_std['NB'], eval_std['KNN']]}, title="StandardScaler Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "04_scaling_standard_eval.png"))
    plt.close()

    # Plot for MinMaxScaler
    ds.plot_multibar_chart(["NB", "KNN"], {"MinMaxScaler": [eval_mm['NB'], eval_mm['KNN']]}, title="MinMaxScaler Evaluation", percentage=True)
    plt.savefig(os.path.join(config.IMAGES_DIR, "04_scaling_minmax_eval.png"))
    plt.close()

    # Plot Side-by-Side Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"StandardScaler": [eval_std['NB'], eval_std['KNN']]}, 
        ax=axs[0], title="StandardScaler", percentage=True
    )
    ds.plot_multibar_chart(
        ["NB", "KNN"], {"MinMaxScaler": [eval_mm['NB'], eval_mm['KNN']]}, 
        ax=axs[1], title="MinMaxScaler", percentage=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "04_scaling_side_by_side.png"))
    plt.close()

    # ===== SELECT BEST APPROACH =====
    best_approach = max(results, key=lambda k: results[k]['AVG'])
    best_f1 = results[best_approach]['AVG']
    
    print(f"\n   SELECTED: {best_approach} (AVG F1={best_f1:.4f})")
    
    # Plot comparison
    chart_path = os.path.join(config.IMAGES_DIR, "04_scaling_comparison.png")
    flights_utils.plot_comparison(results, "Step 4: Feature Scaling", chart_path)
    
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "04_scaling_best_nb_cm.png"))
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
    plt.savefig(os.path.join(config.IMAGES_DIR, "04_scaling_best_knn_cm.png"))
    plt.close()

    # Save best dataset at the end
    flights_utils.save_dataset(X_train, y_train, X_test, y_test, config.FILE_SCALED,
                 metadata={'approach': best_approach, 'f1_score': best_f1})

    print(f"\n   Step 4 Complete!")


if __name__ == "__main__":
    run_scaling()
