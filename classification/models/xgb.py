import pandas as pd
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings
import itertools
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)

datasets = [
    {"name": "Accidents", "file_tag": "accidents", "target": "crash_type"},
]

def xgb_study(file_tag, target_col):
    print(f"\n🧠 STARTING XGBOOST STUDY: {file_tag.upper()}")

    # 1. Load data
    df=pd.read_csv(f"data/prepared/traffic_{file_tag}/6_selected.csv")
    try:
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    except FileNotFoundError:
        print(f"   ⚠️ Skipping {file_tag} (Files not found)")
        return

    # Drop missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    
    
    # Sampling for speed
    if len(train_df) > 100000:
        sample_size = 30000
        train_df = train_df.sample(n=sample_size, random_state=42)
        test_df = test_df.sample(n=min(len(test_df), int(sample_size * 0.3)), random_state=42)
        print(f"   ⚠️ Sampled to {len(train_df)} train, {len(test_df)} test rows")

    # # 2. Prepare X and Y
    trnX = train_df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(-999)
    tstX = test_df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(-999)

    le = LabelEncoder()
    trnY = le.fit_transform(train_df[target_col])
    tstY = le.transform(test_df[target_col])
    num_classes = len(le.classes_)

    # 3. Hyperparameter grid
    param_grid = {
        "learning_rate": [0.1, 0.05],
        "scale_pos_weight": [1, 5],
        "max_depth": [1, 3, 6, 9, 12, 15, 18, 21],
        "min_child_weight": [1, 3],
        "gamma": [0, 1],
    }

    nr_iterations = list(range(250, 2501, 250))
    metrics = ["accuracy", "precision", "recall"]
    hyperparam_results = []

    # 4. Hyperparameter search
    for params in itertools.product(*param_grid.values()):
        hp = dict(zip(param_grid.keys(), params))

        clf = XGBClassifier(
            n_estimators=max(nr_iterations),
            learning_rate=hp["learning_rate"],
            max_depth=hp["max_depth"],
            min_child_weight=hp["min_child_weight"],
            gamma=hp["gamma"],
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=hp["scale_pos_weight"],
            num_class=num_classes,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        )

        clf.fit(trnX, trnY)
        pred_probs = clf.predict(tstX)
        pred = pred_probs.argmax(axis=1)

        acc = accuracy_score(tstY, pred)
        prec = precision_score(tstY, pred, average="weighted", zero_division=0)
        rec = recall_score(tstY, pred, average="weighted", zero_division=0)

        hyperparam_results.append({**hp, "accuracy": acc, "precision": prec, "recall": rec})

    hyperparam_df = pd.DataFrame(hyperparam_results)

    # Print top 3 hyperparameters per metric
    print("\n📊 Hyperparameter Study (top 3 per metric):")
    for metric in metrics:
        print(f"\n--- Top 3 {metric.capitalize()} ---")
        print(hyperparam_df.sort_values(metric, ascending=False).head(3))

    # 5. Metric evolution for best hyperparameters
    for metric in metrics:
        best_hp = hyperparam_df.sort_values(metric, ascending=False).iloc[0]
        print(f"\n🏆 Best {metric.capitalize()} hyperparameters: {best_hp.to_dict()}")

        metric_values = []
        for n_rounds in nr_iterations:
            clf = XGBClassifier(
                n_estimators=n_rounds,
                learning_rate=best_hp["learning_rate"],
                max_depth=int(best_hp["max_depth"]),
                min_child_weight=int(best_hp["min_child_weight"]),
                gamma=best_hp["gamma"],
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=best_hp["scale_pos_weight"],
                num_class=num_classes,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
            )
            clf.fit(trnX, trnY)
            pred_probs = clf.predict(tstX)
            pred = pred_probs.argmax(axis=1)

            if metric == "accuracy":
                metric_values.append(accuracy_score(tstY, pred))
            elif metric == "precision":
                metric_values.append(precision_score(tstY, pred, average="weighted", zero_division=0))
            else:  # recall
                metric_values.append(recall_score(tstY, pred, average="weighted", zero_division=0))

        plt.figure(figsize=(10, 6))
        plt.plot(nr_iterations, metric_values, marker="o")
        plt.title(f"XGB {metric.capitalize()} Evolution ({file_tag})")
        plt.xlabel("Boosting Rounds")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        chart_path = f"images/{file_tag}_xgb_{metric}_evolution.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"   📈 Saved {metric} evolution chart: {chart_path}")

        # Classification report for best metric model
        clf = XGBClassifier(
            n_estimators=nr_iterations[-1],
            learning_rate=best_hp["learning_rate"],
            max_depth=int(best_hp["max_depth"]),
            min_child_weight=int(best_hp["min_child_weight"]),
            gamma=best_hp["gamma"],
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=best_hp["scale_pos_weight"],
            num_class=num_classes,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        )
        clf.fit(trnX, trnY)
        pred_probs = clf.predict(tstX)
        pred = pred_probs.argmax(axis=1)
        print(f"\n--- Classification Report ({metric}) ---")
        print(classification_report(tstY, pred, zero_division=0))

    # 6. Effect of max_depth on accuracy
    best_acc_hp = hyperparam_df.sort_values("accuracy", ascending=False).iloc[0]
    fixed_params = {
        "learning_rate": best_acc_hp["learning_rate"],
        "scale_pos_weight": best_acc_hp["scale_pos_weight"],
        "min_child_weight": int(best_acc_hp["min_child_weight"]),
        "gamma": best_acc_hp["gamma"],
    }

    acc_values = []
    for md in param_grid["max_depth"]:
        clf = XGBClassifier(
            n_estimators=max(nr_iterations),
            learning_rate=fixed_params["learning_rate"],
            max_depth=md,
            min_child_weight=fixed_params["min_child_weight"],
            gamma=fixed_params["gamma"],
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=fixed_params["scale_pos_weight"],
            num_class=num_classes,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        )
        clf.fit(trnX, trnY)
        pred_probs = clf.predict(tstX)
        pred = pred_probs.argmax(axis=1)
        acc_values.append(accuracy_score(tstY, pred))

    plt.figure(figsize=(8,5))
    plt.plot(param_grid["max_depth"], acc_values, marker="o")
    plt.title(f"XGB Effect of Max Depth on Accuracy ({file_tag})")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(f"images/{file_tag}_xgb_accuracy_vs_max_depth.png")
    plt.close()
    print(f"   📈 Saved hyperparameter effect chart: images/{file_tag}_accuracy_vs_max_depth.png")


# === MAIN EXECUTION ===
for ds in datasets:
    xgb_study(ds["file_tag"], ds["target"])
