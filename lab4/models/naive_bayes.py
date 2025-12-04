"""
Naive Bayes Model - Lab 4
Based on Professor Claudia Antunes' DSLabs template
"""

import os
import sys
from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, close
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Add paths - utils FIRST so dslabs_functions finds utils/config.py
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
    plot_multibar_chart,
)
import lab4_config as config


def naive_bayes_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    metric: str = "accuracy",
) -> tuple[GaussianNB | BernoulliNB | None, dict]:
    """
    Study Naive Bayes with different estimators.
    Returns the best model and its parameters.
    """
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
    }

    best_model = None
    best_params: dict = {"name": "NB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}

    for key in CLASS_EVAL_METRICS:
        values[key] = []

    for clf_name, clf in estimators.items():
        try:
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)

            for key in CLASS_EVAL_METRICS:
                eval_score: float = CLASS_EVAL_METRICS[key](tstY, prdY)
                values[key].append(eval_score)

            perf = CLASS_EVAL_METRICS[metric](tstY, prdY)
            if perf - best_performance > DELTA_IMPROVE:
                best_performance = perf
                best_params["params"] = (clf_name,)
                best_model = clf

        except Exception as e:
            print(f"   ⚠️ {clf_name} failed: {e}")
            for key in CLASS_EVAL_METRICS:
                values[key].append(0)

    _, axs = subplots(1, 1, figsize=(HEIGHT * 2, HEIGHT), squeeze=False)
    plot_multibar_chart(
        list(estimators.keys()),
        values,
        ax=axs[0, 0],
        title="Naive Bayes Study",
        xlabel="Estimator",
        ylabel="Score",
        percentage=True,
    )

    if best_params["params"]:
        print(f'NB best for {best_params["params"][0]}')

    return best_model, best_params


def run_for_dataset(file_tag, target, eval_metric="accuracy"):
    """Run Naive Bayes study for a single dataset."""
    print(f"\n{'='*60}")
    print(f"🔵 NAIVE BAYES: {file_tag.upper()}")
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
    best_model, params = naive_bayes_study(trnX, trnY, tstX, tstY, metric=eval_metric)
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_nb_{eval_metric}_study.png"))
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
            f'{file_tag}_nb_{params["name"]}_best_{params["metric"]}_eval.png',
        )
    )
    close()

    # Return results
    return {
        "model": f"NaiveBayes ({params['params'][0]})",
        "params": params["params"][0],
        "accuracy": CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst),
        "precision": CLASS_EVAL_METRICS["precision"](tstY, prd_tst),
        "recall": CLASS_EVAL_METRICS["recall"](tstY, prd_tst),
        "f1": CLASS_EVAL_METRICS["f1"](tstY, prd_tst),
    }


def run():
    """Run Naive Bayes for all datasets."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("🔵 NAIVE BAYES MODELS")
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
        df.to_csv(
            os.path.join(config.RESULTS_DIR, "naive_bayes_results.csv"), index=False
        )
        print(f"\n✅ Results saved to: naive_bayes_results.csv")

    return all_results


if __name__ == "__main__":
    run()
