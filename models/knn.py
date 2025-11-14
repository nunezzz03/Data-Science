import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

# Configuration for 2 datasets
datasets = [
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Flights", "file_tag": "flights", "target": "ArrDel15"},
]


def knn_study(file_tag, target_col):
    print(f"\n🟣 STARTING KNN STUDY: {file_tag.upper()}")

    # 1. Load Data
    try:
        train_df = pd.read_csv(f"data/processed/{file_tag}_train.csv")
        test_df = pd.read_csv(f"data/processed/{file_tag}_test.csv")
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {file_tag} (Files not found)")
        return

    # Sample if dataset is too large (KNN is O(n²) at prediction time)
    if len(train_df) > 100000:
        sample_size = 1000
        train_df = train_df.sample(n=sample_size, random_state=42)
        test_df = test_df.sample(n=min(len(test_df), int(sample_size * 0.3)), random_state=42)
        print(f"   ⚠️ Sampled to {len(train_df)} train, {len(test_df)} test rows for performance")

    # 2. Prepare X (Features) and Y (Target)
    trnX = train_df.drop(columns=[target_col])
    trnY = train_df[target_col]
    tstX = test_df.drop(columns=[target_col])
    tstY = test_df[target_col]

    # 3. Settings for the study
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
    distances = ["manhattan", "euclidean", "chebyshev"]

    best_acc = 0
    best_model = None
    best_params = {}

    # Dictionary to store results for plotting: {'manhattan': [0.8, 0.82...], 'euclidean': ...}
    plot_data = {
        "accuracy": {d: [] for d in distances},
        "precision": {d: [] for d in distances},
        "recall": {d: [] for d in distances}
    }

    # 4. The Training Loop
    for d in distances:
        # print(f"   Testing distance: {d}...") # Uncomment if you want progress updates
        for k in k_values:
            # Train KNN
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)

            # Evaluate
            pred = clf.predict(tstX)
            acc = accuracy_score(tstY, pred)
            prec = precision_score(tstY, pred, average='weighted', zero_division=0)
            rec = recall_score(tstY, pred, average='weighted', zero_division=0)

            # Store results
            plot_data["accuracy"][d].append(acc)
            plot_data["precision"][d].append(prec)
            plot_data["recall"][d].append(rec)

            # Check if best
            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = {"k": k, "dist": d}

    print(
        f"   🏆 Best Configuration: k={best_params['k']} with {best_params['dist']} distance (Acc: {best_acc:.4f})"
    )

    # 5. Plotting the Multiline Charts for all three metrics
    markers = {"manhattan": "o", "euclidean": "s", "chebyshev": "^"}
    
    for metric in ["accuracy", "precision", "recall"]:
        plt.figure(figsize=(10, 6))
        
        for d in distances:
            plt.plot(k_values, plot_data[metric][d], marker=markers[d], label=d)

        plt.title(f"KNN {metric.capitalize()}: {file_tag.capitalize()}")
        plt.xlabel("k (Number of Neighbors)")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)

        chart_path = f"images/{file_tag}_knn_{metric}.png"
        plt.savefig(chart_path)
        plt.close()
    
    print(f"   📈 Charts saved to: images/{file_tag}_knn_*.png")

    # 6. Detailed Report
    print("   --- Classification Report ---")
    if best_model:
        # We make one final prediction with the BEST model to print the table
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))


# === MAIN EXECUTION ===
for ds in datasets:
    knn_study(ds["file_tag"], ds["target"])
