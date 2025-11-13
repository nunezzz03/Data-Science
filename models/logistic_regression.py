import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Ignore "ConvergenceWarning" because we EXPECT raw data to fail converging
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

# Configuration for all 4 datasets
datasets = [
    {"name": "Traffic", "file_tag": "traffic", "target": "Traffic Situation"},
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Economic", "file_tag": "economic", "target": "GDP_Category"},
    {"name": "Flights", "file_tag": "flights", "target": "ArrDel15"},
]


def logistic_regression_study(file_tag, target_col):
    print(f"\n🟠 STARTING LOGISTIC REGRESSION: {file_tag.upper()}")

    # 1. Load Data
    try:
        train_df = pd.read_csv(f"data/processed/{file_tag}_train.csv")
        test_df = pd.read_csv(f"data/processed/{file_tag}_test.csv")
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {file_tag} (Files not found)")
        return

    # Skip if dataset is too large (slow convergence on unscaled data)
    if len(train_df) > 50000:
        print(f"   ⚠️ Skipping {file_tag} (Dataset too large: {len(train_df)} rows)")
        print(f"   💡 LogReg requires iterative optimization - very slow on large unscaled datasets")
        return

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
    plot_data = {p: [] for p in penalties}

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

            plot_data[p].append(acc)

            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = {"penalty": p, "iter": itr}

    print(
        f"   🏆 Best: Penalty={best_params['penalty']}, Max Iter={best_params['iter']} (Acc: {best_acc:.4f})"
    )

    # 5. Plotting (Iterations vs Accuracy)
    plt.figure(figsize=(10, 6))

    plt.plot(iterations, plot_data["l1"], marker="o", label="L1 (Lasso)")
    plt.plot(iterations, plot_data["l2"], marker="s", label="L2 (Ridge)")

    plt.title(f"Logistic Regression Convergence: {file_tag.capitalize()}")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.xscale("log")  # Log scale helps seeing the difference between 10 and 5000
    plt.legend()
    plt.grid(True)

    chart_path = f"images/{file_tag}_lr_study.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"   📈 Chart saved to: {chart_path}")

    # 6. Detailed Report
    print("   --- Classification Report ---")
    if best_model:
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))


# === MAIN EXECUTION ===
for ds in datasets:
    logistic_regression_study(ds["file_tag"], ds["target"])
