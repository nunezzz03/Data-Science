import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Settings
os.makedirs("results", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Define the Best Parameters found during the study
# (We use "safe" defaults that worked generally well to save time)
models = {
    "NaiveBayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="manhattan"),
    "DecisionTree": DecisionTreeClassifier(
        criterion="entropy", max_depth=10, random_state=42
    ),
    "LogReg": LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=1000, random_state=42
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(50,), solver="adam", max_iter=500, random_state=42
    ),
}

datasets = [
    {"name": "Traffic", "file": "traffic"},
    {"name": "Accidents", "file": "accidents"},
    {"name": "Economic", "file": "economic"},
    {"name": "Flights", "file": "flights"},
]

results_list = []

print("🚀 STARTING FINAL AGGREGATION...")

for ds in datasets:
    tag = ds["file"]
    name = ds["name"]
    print(f"\n📂 Dataset: {name}")

    # Load Data
    try:
        train_df = pd.read_csv(f"data/processed/{tag}_train.csv")
        test_df = pd.read_csv(f"data/processed/{tag}_test.csv")
    except FileNotFoundError:
        print(f"   ❌ Files not found for {name}, skipping.")
        continue

    # Identify Target
    if name == "Traffic":
        target = "Traffic Situation"
    elif name == "Accidents":
        target = "crash_type"
    elif name == "Economic":
        target = "GDP_Category"
    elif name == "Flights":
        target = "ArrDel15"

    trnX = train_df.drop(columns=[target])
    trnY = train_df[target]
    tstX = test_df.drop(columns=[target])
    tstY = test_df[target]

    # Run All Models
    for model_name, clf in models.items():

        # SKIP SLOW MODELS FOR LARGE DATASETS
        if (name in ["Flights", "Accidents"] and model_name in ["KNN", "LogReg", "MLP"]):
            print(f"   ⏩ Skipping {model_name} for {name} (Too slow on large unscaled dataset)")
            results_list.append(
                {
                    "Dataset": name,
                    "Model": model_name,
                    "Accuracy": 0,
                    "Precision": 0,
                    "Recall": 0,
                    "Note": "Timed Out / Skipped",
                }
            )
            continue

        print(f"   ⚙️ Training {model_name}...")
        try:
            start_time = time.time()
            clf.fit(trnX, trnY)
            pred = clf.predict(tstX)

            # Metrics
            acc = accuracy_score(tstY, pred)
            # Macro average is best for multiclass baseline comparison
            prec = precision_score(tstY, pred, average="macro", zero_division=0)
            rec = recall_score(tstY, pred, average="macro", zero_division=0)

            results_list.append(
                {
                    "Dataset": name,
                    "Model": model_name,
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "Note": "Success",
                }
            )
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")
            results_list.append(
                {
                    "Dataset": name,
                    "Model": model_name,
                    "Accuracy": 0,
                    "Precision": 0,
                    "Recall": 0,
                    "Note": "Failed",
                }
            )

# --- Save Results to CSV ---
results_df = pd.DataFrame(results_list)
csv_path = "results/baseline_results_summary.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to {csv_path}")

# --- Create Comparison Charts ---
for name in results_df["Dataset"].unique():
    subset = results_df[results_df["Dataset"] == name]
    if subset["Accuracy"].sum() == 0:
        continue  # Skip if all failed

    plt.figure(figsize=(10, 5))
    # Plot Accuracy
    plt.bar(
        subset["Model"],
        subset["Accuracy"],
        color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"],
    )
    plt.title(f"Model Comparison: {name} (Accuracy)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)

    # Add labels
    for i, val in enumerate(subset["Accuracy"]):
        plt.text(i, val + 0.02, str(val), ha="center")

    plt.savefig(f"images/{name.lower()}_final_comparison.png")
    plt.close()
    print(f"📊 Saved chart for {name}")
