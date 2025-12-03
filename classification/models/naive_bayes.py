import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)


# Configuration for 2 datasets
datasets = [
    {
        "name": "Accidents",
        "path": "data/prepared/traffic_accidents/6_selected.csv",
        "target": "crash_type",
    },
    {
        "name": "Flights",
        "path": "data/prepared/flights/6_selected.csv", # Exemplo de caminho corrigido
        "target": "Cancelled",
    },
]

# --- PATH SETUP ---
# Get the project root directory by going up two levels from the current script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGES_DIR = os.path.join(project_root, "lab04", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def naive_bayes_study(name, path, target_col):
    print(f"\n🔵 STARTING NAIVE BAYES: {name.upper()}")

    # 1. Load Data
    filepath = os.path.join(project_root, path)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {name} (File not found at {filepath})")
        return

    # 2. Prepare X (Features) and Y (Target)
    y = df[target_col]
    # Drop target and any helper columns like '_split'
    X = df.drop(columns=[target_col], errors='ignore')
    if '_split' in X.columns:
        X = X.drop(columns=['_split'])

    # 3. Train/Test Split
    trnX, tstX, trnY, tstY = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"   -> Data split: {len(trnX)} train rows, {len(tstX)} test rows")

    # 3. Define the Models to Test
    estimators = {
        "GaussianNB": GaussianNB(var_smoothing=1e-9),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues = []
    yvalues_acc = []
    yvalues_prec = []
    yvalues_rec = []
    best_model = None
    best_acc = 0
    best_name = ""

    # 4. Train and Evaluate Loop
    for est_name, clf in estimators.items():
        try:
            clf.fit(trnX, trnY)
            pred = clf.predict(tstX)
            acc = accuracy_score(tstY, pred)
            train_pred = clf.predict(trnX)
            train_acc = accuracy_score(trnY, train_pred)
            print(f"   -> {est_name}: Train Accuracy: {train_acc:.4f}, Test Accuracy: {acc:.4f}")
            prec = precision_score(tstY, pred, average='weighted', zero_division=0)
            rec = recall_score(tstY, pred, average='weighted', zero_division=0)

            xvalues.append(est_name)
            yvalues_acc.append(acc)
            yvalues_prec.append(prec)
            yvalues_rec.append(rec)

            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_name = est_name

        except ValueError as e:
            # FIXED: checks lower() to handle "Negative" vs "negative"
            if "negative" in str(e).lower():
                print(f"   ⚠️ Skipped {est_name} (Data contains negative values)")
                xvalues.append(est_name)
                yvalues_acc.append(0)
                yvalues_prec.append(0)
                yvalues_rec.append(0)
            else:
                raise e

    print(f"   🏆 Best Model: {best_name} (Accuracy: {best_acc:.4f})")

    # 5. Plot Bar Charts for all three metrics
    metrics = [("accuracy", yvalues_acc), ("precision", yvalues_prec), ("recall", yvalues_rec)]
    
    file_tag = name.lower().replace(" ", "_") # Define file_tag from the dataset name
    for metric_name, metric_values in metrics:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(xvalues, metric_values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title(f"Naive Bayes Models ({metric_name.capitalize()}): {name}")
        plt.xlabel("Model Type")
        plt.ylabel(metric_name.capitalize())
        plt.ylim(0, 1.0)

        # Add text labels on bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
        chart_path = os.path.join(IMAGES_DIR, f"{file_tag}_nb_{metric_name}.png")
        plt.savefig(chart_path)
        plt.close()
    
    print(f"   📈 Charts for '{file_tag.capitalize()}' saved to the 'lab04/images' directory.")

    # Hyperparameter study for GaussianNB
    hyperparams = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    hyper_acc = []

    for param in hyperparams:
        clf = GaussianNB(var_smoothing=param)
        clf.fit(trnX, trnY)
        pred = clf.predict(tstX)
        acc = accuracy_score(tstY, pred)
        hyper_acc.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(hyperparams, hyper_acc, marker='o')
    plt.title("Hyperparameter Study: GaussianNB (var_smoothing)")
    plt.xlabel("Var Smoothing")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    chart_path = os.path.join(IMAGES_DIR, f"{file_tag}_hyperparams_study.png")
    plt.savefig(chart_path)
    plt.close()

    # Overfitting Study: Train vs Test Metrics
    plt.figure(figsize=(8, 5))
    metrics = ["Accuracy", "Precision", "Recall"]
    train_values = [train_acc, precision_score(trnY, train_pred, average='weighted'), recall_score(trnY, train_pred, average='weighted')]
    test_values = [acc, prec, rec]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, train_values, width, label='Train', color='#1f77b4')
    plt.bar(x + width/2, test_values, width, label='Test', color='#ff7f0e')
    plt.xticks(x, metrics)
    plt.title("Overfitting Study: Train vs Test Metrics")
    plt.ylabel("Score")
    plt.legend()
    chart_path = os.path.join(IMAGES_DIR, f"{file_tag}_overfitting_study.png")
    plt.savefig(chart_path)
    plt.close()

    # 6. Detailed Report
    print("   --- Classification Report ---")
    if best_model:
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))
        ConfusionMatrixDisplay.from_estimator(best_model, tstX, tstY)
        plt.title(f"Confusion Matrix: {best_name}")
        chart_path = os.path.join(IMAGES_DIR, f"{file_tag}_confusion_matrix.png")
        plt.savefig(chart_path)
        plt.close()


# === MAIN EXECUTION ===
for ds in datasets:
    naive_bayes_study(ds["name"], ds["path"], ds["target"])