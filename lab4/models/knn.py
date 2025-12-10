"""
K-Nearest Neighbors Model - Lab 4
Based on Professor Claudia Antunes' DSLabs template
"""

import os
import sys
from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, close
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", "utils"))
sys.path.insert(0, PROJECT_ROOT)

from utils.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    HEIGHT,
    read_train_test_from_files,
    plot_evaluation_results,
    plot_multiline_chart,
)
import lab4_config as config


def knn_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    k_max: int = 25,
    lag: int = 2,
    metric: str = "accuracy",
) -> tuple[KNeighborsClassifier | None, dict]:
    """
    Study KNN with different K values and distance metrics.
    Returns the best model and its parameters.
    """
    k_values: list[int] = [i for i in range(1, k_max + 1, lag)]
    dist_functions: list[str] = ["manhattan", "euclidean", "chebyshev"]

    best_model: KNeighborsClassifier | None = None
    best_params: dict = {"name": "KNN", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    cols: int = len(dist_functions)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    for i, dist in enumerate(dist_functions):
        values = {}
        y_tst_values: list[float] = []

        for k in k_values:
            clf = KNeighborsClassifier(n_neighbors=k, metric=dist)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval_score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval_score)

            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (k, dist)
                best_model = clf

        values[dist] = y_tst_values
        plot_multiline_chart(
            k_values,
            {dist: y_tst_values},
            ax=axs[0, i],
            title=f"KNN with {dist} distance",
            xlabel="k",
            ylabel=metric,
            percentage=True,
        )

    if best_params["params"]:
        print(
            f'KNN best for k={best_params["params"][0]} with {best_params["params"][1]} distance'
        )

    return best_model, best_params


def knn_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, metric):
    """Study overfitting for KNN."""
    best_k: int = params["params"][0]
    best_dist: str = params["params"][1]
    k_values: list[int] = [i for i in range(1, 51, 2)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric=best_dist)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS["accuracy"](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        k_values,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"KNN overfitting study with {best_dist} distance",
        xlabel="k",
        ylabel="accuracy",
        percentage=True,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_knn_{metric}_overfitting.png"))
    close()


def run_for_dataset(file_tag, target, eval_metric="accuracy"):
    """Run KNN study for a single dataset."""
    print(f"\n{'='*60}")
    print(f"🟣 KNN: {file_tag.upper()}")
    print(f"{'='*60}")

    train_filename = os.path.join(config.PROCESSED_DATA_DIR, f"{file_tag}_train.csv")
    test_filename = os.path.join(config.PROCESSED_DATA_DIR, f"{file_tag}_test.csv")

    if not os.path.exists(train_filename):
        print(f"   ⚠️ Data not found: {train_filename}")
        return None

    trnX, tstX, trnY, tstY, labels, vars_list = read_train_test_from_files(
        train_filename, test_filename, target
    )
    print(f"   Train#={len(trnX)} Test#={len(tstX)}")
    print(f"   Labels={labels}")

    # 1. Parameters Study
    figure()
    best_model, params = knn_study(
        trnX, trnY, tstX, tstY, k_max=25, lag=2, metric=eval_metric
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_knn_{eval_metric}_study.png"))
    close()

    if best_model is None:
        print("   ⚠️ No model found")
        return None

    # 2. Best Model Performance
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(
        os.path.join(
            config.IMAGES_DIR,
            f'{file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png',
        )
    )
    close()

    # 3. Overfitting Study
    knn_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, eval_metric)

    # Return results
    return {
        "model": "KNN",
        "params": f"k={params['params'][0]}, metric={params['params'][1]}",
        "accuracy": CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst),
        "precision": CLASS_EVAL_METRICS["precision"](tstY, prd_tst),
        "recall": CLASS_EVAL_METRICS["recall"](tstY, prd_tst),
        "f1": CLASS_EVAL_METRICS["f1"](tstY, prd_tst),
    }


def run():
    """Run KNN for all datasets."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("🟣 K-NEAREST NEIGHBORS MODELS")
    print("=" * 60)

    all_results = []

    for dataset in config.DATASETS:
        result = run_for_dataset(
            dataset["file_tag"], dataset["target"], eval_metric="accuracy"
        )
        if result:
            result["dataset"] = dataset["name"]
            all_results.append(result)

    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(config.RESULTS_DIR, "knn_results.csv"), index=False)
        print(f"\n✅ Results saved to: knn_results.csv")

    return all_results


if __name__ == "__main__":
    run()
