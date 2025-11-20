import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

# Configuration for 2 datasets
datasets = [
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Flights", "file_tag": "flights", "target": "ArrDel15"},
]


def naive_Bayes_study(file_tag, target_col):
    print(f"\n🔵 STARTING NAIVE BAYES: {file_tag.upper()}")

    # 1. Load Data
    try:
        train_df = pd.read_csv(f"data/processed/{file_tag}_train.csv")
        test_df = pd.read_csv(f"data/processed/{file_tag}_test.csv")
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {file_tag} (Files not found)")
        return

    # 2. Prepare X (Features) and Y (Target)
    trnX = train_df.drop(columns=[target_col])
    trnY = train_df[target_col]
    tstX = test_df.drop(columns=[target_col])
    tstY = test_df[target_col]

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
    
    for metric_name, metric_values in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(xvalues, metric_values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title(f"Naive Bayes Models ({metric_name.capitalize()}): {file_tag.capitalize()}")
        plt.xlabel("Model Type")
        plt.ylabel(metric_name.capitalize())
        plt.ylim(0, 1.0)

        # Add text labels on bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

        chart_path = f"images/{file_tag}_nb_{metric_name}.png"
        plt.savefig(chart_path)
        plt.close()
    
    print(f"   📈 Charts saved to: images/{file_tag}_nb_*.png")

    # 6. Detailed Report
    print("   --- Classification Report ---")
    if best_model:
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))


# === MAIN EXECUTION ===
for ds in datasets:
    naive_Bayes_study(ds["file_tag"], ds["target"])
