"""
Decision Tree Model - Lab 4
Based on Professor Claudia Antunes' DSLabs template
"""

import os
import sys
from numpy import array, ndarray, argsort
from matplotlib.pyplot import subplots, figure, savefig, close
from sklearn.tree import DecisionTreeClassifier, plot_tree

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


def decision_tree_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    max_depth: int = 25,
    metric: str = "accuracy",
) -> tuple[DecisionTreeClassifier | None, dict]:
    """
    Study Decision Tree with different depths and criteria.
    Returns the best model and its parameters.
    """
    criteria: list[str] = ["entropy", "gini"]
    depths: list[int] = [i for i in range(2, max_depth + 1, 2)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {"name": "DT", "metric": metric, "params": ()}
    best_performance: float = 0.0

    cols: int = len(criteria)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    for i, criterion in enumerate(criteria):
        y_tst_values: list[float] = []

        for d in depths:
            clf = DecisionTreeClassifier(
                max_depth=d, criterion=criterion, random_state=config.RANDOM_STATE
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval_score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval_score)

            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (d, criterion)
                best_model = clf

        plot_multiline_chart(
            depths,
            {criterion: y_tst_values},
            ax=axs[0, i],
            title=f"Decision Tree with {criterion}",
            xlabel="max_depth",
            ylabel=metric,
            percentage=True,
        )

    if best_params["params"]:
        print(
            f'DT best for max_depth={best_params["params"][0]} with {best_params["params"][1]}'
        )

    return best_model, best_params


def dt_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, metric):
    """Study overfitting for Decision Tree."""
    best_criterion: str = params["params"][1]
    depths: list[int] = [i for i in range(2, 31, 2)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []

    for d in depths:
        clf = DecisionTreeClassifier(
            max_depth=d, criterion=best_criterion, random_state=config.RANDOM_STATE
        )
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS["accuracy"](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        depths,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"DT overfitting study with {best_criterion}",
        xlabel="max_depth",
        ylabel="accuracy",
        percentage=True,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_dt_{metric}_overfitting.png"))
    close()


def plot_dt_feature_importance(best_model, vars_list, file_tag, metric):
    """Plot feature importance for Decision Tree."""
    importances = best_model.feature_importances_
    indices: list[int] = list(argsort(importances)[::-1])
    elems: list[str] = []
    imp_values: list[float] = []

    print("\n   Feature Importances:")
    for f in range(len(vars_list)):
        elems.append(vars_list[indices[f]])
        imp_values.append(importances[indices[f]])
        if importances[indices[f]] > 0.01:  # Only print significant features
            print(f"   {f+1}. {elems[f]} ({importances[indices[f]]:.4f})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title="DT variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_dt_{metric}_vars_ranking.png"))
    close()


def plot_tree_structure(best_model, vars_list, file_tag):
    """Plot tree structure (limited depth for readability)."""
    fig, ax = subplots(1, 1, figsize=(20, 10))
    plot_tree(
        best_model,
        feature_names=vars_list,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree Structure (max_depth=3)")
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_dt_tree.png"), dpi=150)
    close()


def run_for_dataset(file_tag, target, eval_metric="accuracy"):
    """Run Decision Tree study for a single dataset."""
    print(f"\n{'='*60}")
    print(f"🌳 DECISION TREE: {file_tag.upper()}")
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
    best_model, params = decision_tree_study(
        trnX, trnY, tstX, tstY, max_depth=25, metric=eval_metric
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_dt_{eval_metric}_study.png"))
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
            f'{file_tag}_dt_{params["name"]}_best_{params["metric"]}_eval.png',
        )
    )
    close()

    # 3. Feature Importance
    plot_dt_feature_importance(best_model, vars_list, file_tag, eval_metric)

    # 4. Tree Structure
    plot_tree_structure(best_model, vars_list, file_tag)

    # 5. Overfitting Study
    dt_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, eval_metric)

    # Return results
    return {
        "model": "DecisionTree",
        "params": f"max_depth={params['params'][0]}, criterion={params['params'][1]}",
        "accuracy": CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst),
        "precision": CLASS_EVAL_METRICS["precision"](tstY, prd_tst),
        "recall": CLASS_EVAL_METRICS["recall"](tstY, prd_tst),
        "f1": CLASS_EVAL_METRICS["f1"](tstY, prd_tst),
    }


def run():
    """Run Decision Tree for all datasets."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("🌳 DECISION TREE MODELS")
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
            os.path.join(config.RESULTS_DIR, "decision_tree_results.csv"), index=False
        )
        print(f"\n✅ Results saved to: decision_tree_results.csv")

    return all_results


if __name__ == "__main__":
    run()
