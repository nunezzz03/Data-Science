import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import warnings

# Ignore convergence warnings (expected on raw data)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

os.makedirs("images", exist_ok=True)

# Configuration for 2 datasets
datasets = [
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Flights", "file_tag": "flights", "target": "ArrDel15"},
]


def mlp_study(file_tag, target_col):
    print(f"\n🧠 STARTING MLP STUDY: {file_tag.upper()}")

    # 1. Load Data
    try:
        train_df = pd.read_csv(f"../data/processed/{file_tag}_train.csv")
        test_df = pd.read_csv(f"../data/processed/{file_tag}_test.csv")
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {file_tag} (Files not found)")
        return

    # Sample if dataset is too large (MLP needs scaling and is very slow on raw data)
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
    nr_max_iterations = 2500
    lag = 250  # Step size (Check accuracy every 250 iterations)

    # We test these specific learning settings
    lr_types = ["constant", "invscaling", "adaptive"]
    learning_rates = [0.5, 0.05, 0.005, 0.0005]

    nr_iterations = list(range(lag, nr_max_iterations + 1, lag))

    best_acc = 0
    best_model = None
    best_params = {}

    # 4. Main Loop (Iterating over Learning Rate Types)
    # We create one plot per Learning Rate Type and metric to keep it readable
    for metric_name in ["accuracy", "precision", "recall"]:
        for lr_type in lr_types:
            plt.figure(figsize=(10, 6))

            # Inner Loop (Different starting speeds)
            for lr in learning_rates:
                # Setup the model with warm_start=True
                # This allows us to pause, check metric, and resume training
                clf = MLPClassifier(
                    learning_rate=lr_type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=True,
                    activation="logistic",  # Using Logistic as per snippet
                    solver="sgd",  # Using SGD as per snippet
                    random_state=42,
                )

                metric_values = []

                # Iteration Loop (The "Warm Start" Trick)
                for _ in nr_iterations:
                    try:
                        clf.fit(trnX, trnY)
                        pred = clf.predict(tstX)
                        
                        if metric_name == "accuracy":
                            metric_val = accuracy_score(tstY, pred)
                        elif metric_name == "precision":
                            metric_val = precision_score(tstY, pred, average='weighted', zero_division=0)
                        else:  # recall
                            metric_val = recall_score(tstY, pred, average='weighted', zero_division=0)
                    except Exception:
                        metric_val = 0  # If it fails (NaNs in data etc), record 0

                    metric_values.append(metric_val)

                    # Save best model (only for accuracy)
                    if metric_name == "accuracy" and metric_val > best_acc:
                        best_acc = metric_val
                        best_model = clf
                        best_params = {"lr_type": lr_type, "lr": lr, "iters": clf.n_iter_}

                # Plot this line
                plt.plot(nr_iterations, metric_values, marker=".", label=f"LR={lr}")

            # Finalize Plot for this LR Type and metric
            plt.title(f"MLP {metric_name.capitalize()} ({lr_type}): {file_tag.capitalize()}")
            plt.xlabel("Iterations")
            plt.ylabel(metric_name.capitalize())
            plt.legend()
            plt.grid(True)

            chart_name = f"images/{file_tag}_mlp_{lr_type}_{metric_name}.png"
            plt.savefig(chart_name)
            plt.close()
            # print(f"   📈 Saved chart: {chart_name}")

    print(
        f"   🏆 Best Model: Type={best_params.get('lr_type')}, LR={best_params.get('lr')} (Acc: {best_acc:.4f})"
    )

    # 5. Report
    print("   --- Classification Report ---")
    if best_model:
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))


# === MAIN EXECUTION ===
for ds in datasets:
    mlp_study(ds["file_tag"], ds["target"])
