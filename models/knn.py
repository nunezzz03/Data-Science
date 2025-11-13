import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

# Configuration for all 4 datasets
datasets = [
    {"name": "Traffic", "file_tag": "traffic", "target": "Traffic Situation"},
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Economic", "file_tag": "economic", "target": "GDP_Category"},
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

    # Skip if dataset is too large (KNN is O(n²) at prediction time)
    if len(train_df) > 50000:
        print(f"   ⚠️ Skipping {file_tag} (Dataset too large: {len(train_df)} rows)")
        print(f"   💡 KNN requires computing distances to all training samples - computationally prohibitive")
        return

    # 2. Prepare X (Features) and Y (Target)
    trnX = train_df.drop(columns=[target_col])
    trnY = train_df[target_col]
    tstX = test_df.drop(columns=[target_col])
    tstY = test_df[target_col]

    # 3. Settings for the study
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    distances = ["manhattan", "euclidean", "chebyshev"]

    best_acc = 0
    best_model = None
    best_params = {}

    # Dictionary to store results for plotting: {'manhattan': [0.8, 0.82...], 'euclidean': ...}
    plot_data = {d: [] for d in distances}

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

            # Store result
            plot_data[d].append(acc)

            # Check if best
            if acc > best_acc:
                best_acc = acc
                best_model = clf
                best_params = {"k": k, "dist": d}

    print(
        f"   🏆 Best Configuration: k={best_params['k']} with {best_params['dist']} distance (Acc: {best_acc:.4f})"
    )

    # 5. Plotting the Multiline Chart
    plt.figure(figsize=(10, 6))
    markers = {"manhattan": "o", "euclidean": "s", "chebyshev": "^"}

    for d in distances:
        plt.plot(k_values, plot_data[d], marker=markers[d], label=d)

    plt.title(f"KNN Hyperparameters: {file_tag.capitalize()}")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    chart_path = f"images/{file_tag}_knn_study.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"   📈 Chart saved to: {chart_path}")

    # 6. Detailed Report
    print("   --- Classification Report ---")
    if best_model:
        # We make one final prediction with the BEST model to print the table
        print(classification_report(tstY, best_model.predict(tstX), zero_division=0))


# === MAIN EXECUTION ===
for ds in datasets:
    knn_study(ds["file_tag"], ds["target"])
