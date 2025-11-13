import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Create images folder
os.makedirs("images", exist_ok=True)

# === CONFIGURATION FOR ALL 4 DATASETS ===
datasets = [
    {"name": "Traffic", "file_tag": "traffic", "target": "Traffic Situation"},
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
    {"name": "Economic", "file_tag": "economic", "target": "GDP_Category"},
    {"name": "Flights", "file_tag": "flights", "target": "ArrDel15"},
]


def trees_study(file_tag, target_col):
    print(f"\n🔎 STARTING STUDY: {file_tag.upper()}")

    # Load Data
    try:
        train_df = pd.read_csv(f"data/processed/{file_tag}_train.csv")
        test_df = pd.read_csv(f"data/processed/{file_tag}_test.csv")
    except FileNotFoundError:
        print(f"   Skipping {file_tag} (Files not found)")
        return

    trnX = train_df.drop(columns=[target_col])
    trnY = train_df[target_col]
    tstX = test_df.drop(columns=[target_col])
    tstY = test_df[target_col]

    labels = trnY.unique()

    # Settings
    criteria = ["entropy", "gini"]
    depths = [2, 4, 6, 8, 10, 15, 20, 25]
    plot_data = {"entropy": [], "gini": []}

    best_acc = 0
    best_model = None
    best_params = {}

    # Train Loop
    for c in criteria:
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, random_state=42)
            clf.fit(trnX, trnY)
            score = accuracy_score(tstY, clf.predict(tstX))
            plot_data[c].append(score)

            if score > best_acc:
                best_acc = score
                best_model = clf
                best_params = {"c": c, "d": d}

    print(
        f"   Best: {best_params['c']} with depth={best_params['d']} (Acc: {best_acc:.2f})"
    )

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(depths, plot_data["entropy"], "-o", label="Entropy")
    plt.plot(depths, plot_data["gini"], "-s", label="Gini")
    plt.title(f"Decision Tree: {file_tag.capitalize()}")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/{file_tag}_dt_study.png")
    plt.close()

    # Print Report
    print("   --- Classification Report ---")
    print(classification_report(tstY, best_model.predict(tstX)))


# Run for all
for ds in datasets:
    trees_study(ds["file_tag"], ds["target"])
