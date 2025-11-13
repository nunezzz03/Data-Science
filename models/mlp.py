import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Ignore convergence warnings (expected on raw data)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

os.makedirs("images", exist_ok=True)

# Configuration for all 4 datasets
datasets = [
    {"name": "Traffic", "file_tag": "traffic", "target": "Traffic Situation"},
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Economic", "file_tag": "economic", "target": "GDP_Category"},
    {"name": "Flights", "file_tag": "flights", "target": "ArrDel15"},
]


def mlp_study(file_tag, target_col):
    print(f"\n🧠 STARTING MLP STUDY: {file_tag.upper()}")

    # 1. Load Data
    try:
        train_df = pd.read_csv(f"data/processed/{file_tag}_train.csv")
        test_df = pd.read_csv(f"data/processed/{file_tag}_test.csv")
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {file_tag} (Files not found)")
        return

    # Skip if dataset is too large (MLP needs scaling and is very slow on raw data)
    if len(train_df) > 50000:
        print(f"   ⚠️ Skipping {file_tag} (Dataset too large: {len(train_df)} rows)")
        print(f"   💡 MLP requires feature scaling and extensive training - very slow on large unscaled datasets")
        return

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
    # We create one plot per Learning Rate Type to keep it readable
    for lr_type in lr_types:
        plt.figure(figsize=(10, 6))

        # Inner Loop (Different starting speeds)
        for lr in learning_rates:
            # Setup the model with warm_start=True
            # This allows us to pause, check accuracy, and resume training
            clf = MLPClassifier(
                learning_rate=lr_type,
                learning_rate_init=lr,
                max_iter=lag,
                warm_start=True,
                activation="logistic",  # Using Logistic as per snippet
                solver="sgd",  # Using SGD as per snippet
                random_state=42,
            )

            accuracies = []

            # Iteration Loop (The "Warm Start" Trick)
            for _ in nr_iterations:
                try:
                    clf.fit(trnX, trnY)
                    pred = clf.predict(tstX)
                    acc = accuracy_score(tstY, pred)
                except Exception:
                    acc = 0  # If it fails (NaNs in data etc), record 0

                accuracies.append(acc)

                # Save best model
                if acc > best_acc:
                    best_acc = acc
                    best_model = clf
                    best_params = {"lr_type": lr_type, "lr": lr, "iters": clf.n_iter_}

            # Plot this line
            plt.plot(nr_iterations, accuracies, marker=".", label=f"LR={lr}")

        # Finalize Plot for this LR Type
        plt.title(f"MLP Training ({lr_type}): {file_tag.capitalize()}")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        chart_name = f"images/{file_tag}_mlp_{lr_type}_study.png"
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
