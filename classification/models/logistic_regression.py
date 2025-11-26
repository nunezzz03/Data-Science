import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import warnings

# Ignore "ConvergenceWarning" because we EXPECT raw data to fail converging
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

# Configuration for 2 datasets
datasets = [
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Flights", "file_tag": "flights", "target": "ArrDel15"},
]


def logistic_regression_study(file_tag, target_col):
    print(f"\n🟠 STARTING LOGISTIC REGRESSION: {file_tag.upper()}")

    # 1. Load Data
    try:
        train_df = pd.read_csv(f"../data/processed/{file_tag}_train.csv")
        test_df = pd.read_csv(f"../data/processed/{file_tag}_test.csv")
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {file_tag} (Files not found)")
        return

    # Sample if dataset is too large (slow convergence on unscaled data)
    if len(train_df) > 100000:
        sample_size = 1000
        train_df = train_df.sample(n=sample_size, random_state=42)
        test_df = test_df.sample(n=min(len(test_df), int(sample_size * 0.3)), random_state=42)
        print(f"   ⚠️ Sampled to {len(train_df)} train, {len(test_df)} test rows for performance")

    # 2. Prepare X and Y
    trnX = train_df.drop(columns=[target_col])
    trnY = train_df[target_col]
    tstX = test_df.drop(columns=[target_col])
    tstY = test_df[target_col]

    # 3. Settings
    # We test L1 (Lasso) and L2 (Ridge) penalties
    # We test different iteration counts to see if training longer helps
    penalties = ["l1", "l2"]
    iterations = [10, 50, 100, 500, 1000, 2500, 5000]

    best_acc = 0
    best_model = None
    best_params = {}

    # Plot Data: {'l1': [acc_10, acc_50...], 'l2': ...}
    plot_data = {
        "accuracy": {p: [] for p in penalties},
        "precision": {p: [] for p in penalties},
        "recall": {p: [] for p in penalties}
    }

    # 4. Training Loop
    for p in penalties:
        for itr in iterations:
            # solver='liblinear' is good for small datasets and supports both L1 and L2
            clf = LogisticRegression(
                penalty=p, solver="liblinear", max_iter=itr, random_state=42
            )

            clf.fit(trnX, trnY)
            pred = clf.predict(tstX)
            acc = accuracy_score(tstY, pred)
            prec = precision_score(tstY, pred, average='weighted', zero_division=0)
            rec = recall_score(tstY, pred, average='weighted', zero_division=0)

            plot_data["accuracy"][p].append(acc)
            plot_data["precision"][p].append(prec)
            plot_data["recall"][p].append(rec)

            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = {"penalty": p, "iter": itr}

    print(
        f"   🏆 Best: Penalty={best_params['penalty']}, Max Iter={best_params['iter']} (Acc: {best_acc:.4f})"
    )

    # 5. Plotting (Iterations vs metric) for all three metrics
    for metric in ["accuracy", "precision", "recall"]:
        plt.figure(figsize=(10, 6))

        plt.plot(iterations, plot_data[metric]["l1"], marker="o", label="L1 (Lasso)")
        plt.plot(iterations, plot_data[metric]["l2"], marker="s", label="L2 (Ridge)")

        plt.title(f"Logistic Regression {metric.capitalize()}: {file_tag.capitalize()}")
        plt.xlabel("Number of Iterations")
        plt.ylabel(metric.capitalize())
        plt.xscale("log")  # Log scale helps seeing the difference between 10 and 5000
        plt.legend()
        plt.grid(True)

        chart_path = f"images/{file_tag}_lr_{metric}.png"
        plt.savefig(chart_path)
        plt.close()
    
    print(f"   📈 Charts saved to: images/{file_tag}_lr_*.png")

    # 6. Detailed Report
    print("   --- Classification Report ---")
    if best_model:
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))


# === MAIN EXECUTION ===
for ds in datasets:
    logistic_regression_study(ds["file_tag"], ds["target"])
