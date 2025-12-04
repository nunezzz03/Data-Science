"""
Gradient Boosting Model - Lab 4
Based on Professor Claudia Antunes' DSLabs template
"""

import os
import sys
from numpy import array, ndarray, std, argsort
from matplotlib.pyplot import subplots, figure, savefig, show, close
from sklearn.ensemble import GradientBoostingClassifier

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
    plot_horizontal_bar_chart,
)
import lab4_config as config


def gradient_boosting_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[GradientBoostingClassifier | None, dict]:
    """
    Study Gradient Boosting with different hyperparameters.
    Returns the best model and its parameters.
    """
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: GradientBoostingClassifier | None = None
    best_params: dict = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for lr in learning_rates:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = GradientBoostingClassifier(
                    n_estimators=n,
                    max_depth=d,
                    learning_rate=lr,
                    random_state=config.RANDOM_STATE,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval_score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval_score)
                if eval_score - best_performance > DELTA_IMPROVE:
                    best_performance = eval_score
                    best_params["params"] = (d, lr, n)
                    best_model = clf
            values[lr] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Gradient Boosting with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )

    if best_params["params"]:
        print(
            f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]})'
        )

    return best_model, best_params


def plot_gb_feature_importance(best_model, vars_list, file_tag, metric):
    """Plot feature importance for Gradient Boosting."""
    # For GB, each estimator is a list of trees
    trees_importances: list[float] = []
    for lst_trees in best_model.estimators_:
        for tree in lst_trees:
            trees_importances.append(tree.feature_importances_)

    stdevs: list[float] = list(std(trees_importances, axis=0))
    importances = best_model.feature_importances_
    indices: list[int] = list(argsort(importances)[::-1])
    elems: list[str] = []
    imp_values: list[float] = []

    print("\n   Feature Importances:")
    for f in range(len(vars_list)):
        elems.append(vars_list[indices[f]])
        imp_values.append(importances[indices[f]])
        print(f"   {f+1}. {elems[f]} ({importances[indices[f]]:.4f})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="GB variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_gb_{metric}_vars_ranking.png"))
    close()


def gb_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, metric):
    """Study overfitting for Gradient Boosting."""
    d_max: int = params["params"][0]
    lr: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric: str = "accuracy"

    for n in nr_estimators:
        clf = GradientBoostingClassifier(
            n_estimators=n,
            max_depth=d_max,
            learning_rate=lr,
            random_state=config.RANDOM_STATE,
        )
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"GB overfitting study for d={d_max} and lr={lr}",
        xlabel="nr_estimators",
        ylabel=str(metric),
        percentage=True,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_gb_{metric}_overfitting.png"))
    close()


def run_for_dataset(file_tag, target, eval_metric="accuracy"):
    """Run Gradient Boosting study for a single dataset."""
    print(f"\n{'='*60}")
    print(f"🚀 GRADIENT BOOSTING: {file_tag.upper()}")
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
    best_model, params = gradient_boosting_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_trees=1000,
        lag=250,
        metric=eval_metric,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_gb_{eval_metric}_study.png"))
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
            f'{file_tag}_gb_{params["name"]}_best_{params["metric"]}_eval.png',
        )
    )
    close()

    # 3. Feature Importance
    plot_gb_feature_importance(best_model, vars_list, file_tag, eval_metric)

    # 4. Overfitting Study
    gb_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, eval_metric)

    # Return results
    return {
        "model": "GradientBoosting",
        "params": f"d={params['params'][0]}, lr={params['params'][1]}, n={params['params'][2]}",
        "accuracy": CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst),
        "precision": CLASS_EVAL_METRICS["precision"](tstY, prd_tst),
        "recall": CLASS_EVAL_METRICS["recall"](tstY, prd_tst),
        "f1": CLASS_EVAL_METRICS["f1"](tstY, prd_tst),
    }


def run():
    """Run Gradient Boosting for all datasets."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("🚀 GRADIENT BOOSTING MODELS")
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
            os.path.join(config.RESULTS_DIR, "gradient_boosting_results.csv"),
            index=False,
        )
        print(f"\n✅ Results saved to: gradient_boosting_results.csv")

    return all_results


if __name__ == "__main__":
    run()
