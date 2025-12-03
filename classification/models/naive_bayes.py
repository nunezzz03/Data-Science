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
        "path": "data/processed/flights_pipeline/06_selected.csv",
        "target": "Cancelled",
    },
]

# --- PATH SETUP ---
# Get the project root directory by going up two levels from the current script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGES_DIR = os.path.join(project_root, "images")
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
        "GaussianNB": GaussianNB(),
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
    for name, clf in estimators.items():
        try:
            clf.fit(trnX, trnY)
            pred = clf.predict(tstX)
            acc = accuracy_score(tstY, pred)
            prec = precision_score(tstY, pred, average='weighted', zero_division=0)
            rec = recall_score(tstY, pred, average='weighted', zero_division=0)

            xvalues.append(name)
            yvalues_acc.append(acc)
            yvalues_prec.append(prec)
            yvalues_rec.append(rec)

            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_name = name

        except ValueError as e:
            # FIXED: checks lower() to handle "Negative" vs "negative"
            if "negative" in str(e).lower():
                print(f"   ⚠️ Skipped {name} (Data contains negative values)")
                xvalues.append(name)
                yvalues_acc.append(0)
                yvalues_prec.append(0)
                yvalues_rec.append(0)
            else:
                raise e

    print(f"   🏆 Best Model: {best_name} (Accuracy: {best_acc:.4f})")

    # 5. Plot Bar Charts for all three metrics
    metrics = [("accuracy", yvalues_acc), ("precision", yvalues_prec), ("recall", yvalues_rec)]
    
    file_tag = name.lower() # Define file_tag from the dataset name
    for metric_name, metric_values in metrics:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(xvalues, metric_values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title(f"Naive Bayes Models ({metric_name.capitalize()}): {file_tag.capitalize()}")
        plt.xlabel("Model Type")
        plt.ylabel(metric_name.capitalize())
        plt.ylim(0, 1.0)

        # Add text labels on bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
        chart_path = os.path.join(IMAGES_DIR, f"{file_tag}_nb_{metric_name}_singlefile.png")
        plt.savefig(chart_path)
        plt.close()
    
    print(f"   📈 Charts saved to: images/{file_tag}_nb_*_singlefile.png")

    # 6. Detailed Report
    print("   --- Classification Report ---")
    if best_model:
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))


# === MAIN EXECUTION ===
for ds in datasets:
    naive_bayes_study(ds["name"], ds["path"], ds["target"])