"""
Logistic Regression Model - Lab 4
Based on Professor Claudia Antunes' DSLabs template
"""

import os
import sys
import warnings
from numpy import array, ndarray, argsort, abs as np_abs
from matplotlib.pyplot import subplots, figure, savefig, close
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
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


def logistic_regression_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    metric: str = "accuracy",
) -> tuple[LogisticRegression | None, dict]:
    """
    Study Logistic Regression with different C values and penalties.
    Returns the best model and its parameters.
    """
    penalties: list[str] = ["l1", "l2"]
    c_values: list[float] = [0.001, 0.01, 0.1, 1, 10, 100]

    best_model: LogisticRegression | None = None
    best_params: dict = {"name": "LR", "metric": metric, "params": ()}
    best_performance: float = 0.0

    cols: int = len(penalties)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    for i, penalty in enumerate(penalties):
        y_tst_values: list[float] = []

        for c in c_values:
            clf = LogisticRegression(
                penalty=penalty,
                C=c,
                solver="liblinear",
                max_iter=1000,
                random_state=config.RANDOM_STATE,
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval_score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval_score)

            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (c, penalty)
                best_model = clf

        plot_multiline_chart(
            c_values,
            {penalty: y_tst_values},
            ax=axs[0, i],
            title=f"Logistic Regression with {penalty.upper()} penalty",
            xlabel="C",
            ylabel=metric,
            percentage=True,
        )

    if best_params["params"]:
        print(
            f'LR best for C={best_params["params"][0]} with {best_params["params"][1]} penalty'
        )

    return best_model, best_params


def lr_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, metric):
    """Study overfitting for Logistic Regression."""
    best_penalty: str = params["params"][1]
    c_values: list[float] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []

    for c in c_values:
        clf = LogisticRegression(
            penalty=best_penalty,
            C=c,
            solver="liblinear",
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS["accuracy"](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        c_values,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"LR overfitting study with {best_penalty} penalty",
        xlabel="C",
        ylabel="accuracy",
        percentage=True,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_lr_{metric}_overfitting.png"))
    close()


def plot_lr_feature_importance(best_model, vars_list, file_tag, metric):
    """Plot feature importance (coefficients) for Logistic Regression."""
    if not hasattr(best_model, "coef_"):
        return

    coefs = np_abs(best_model.coef_).flatten()
    if len(coefs) != len(vars_list):
        return

    indices: list[int] = list(argsort(coefs)[::-1])
    elems: list[str] = []
    imp_values: list[float] = []

    print("\n   Feature Importances (Absolute Coefficients):")
    for f in range(min(20, len(vars_list))):  # Top 20
        elems.append(vars_list[indices[f]])
        imp_values.append(coefs[indices[f]])
        print(f"   {f+1}. {elems[f]} ({coefs[indices[f]]:.4f})")

    figure()
    plot_horizontal_bar_chart(
        elems[:20],
        imp_values[:20],
        title="LR variables importance (abs coefficients)",
        xlabel="importance",
        ylabel="variables",
        percentage=False,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_lr_{metric}_vars_ranking.png"))
    close()


def run_for_dataset(file_tag, target, eval_metric="accuracy"):
    """Run Logistic Regression study for a single dataset."""
    print(f"\n{'='*60}")
    print(f"🟠 LOGISTIC REGRESSION: {file_tag.upper()}")
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
    best_model, params = logistic_regression_study(
        trnX, trnY, tstX, tstY, metric=eval_metric
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_lr_{eval_metric}_study.png"))
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
            f'{file_tag}_lr_{params["name"]}_best_{params["metric"]}_eval.png',
        )
    )
    close()

    # 3. Feature Importance
    plot_lr_feature_importance(best_model, vars_list, file_tag, eval_metric)

    # 4. Overfitting Study
    lr_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, eval_metric)

    # Return results
    return {
        "model": "LogisticRegression",
        "params": f"C={params['params'][0]}, penalty={params['params'][1]}",
        "accuracy": CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst),
        "precision": CLASS_EVAL_METRICS["precision"](tstY, prd_tst),
        "recall": CLASS_EVAL_METRICS["recall"](tstY, prd_tst),
        "f1": CLASS_EVAL_METRICS["f1"](tstY, prd_tst),
    }


def run():
    """Run Logistic Regression for all datasets."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("🟠 LOGISTIC REGRESSION MODELS")
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
            os.path.join(config.RESULTS_DIR, "logistic_regression_results.csv"),
            index=False,
        )
        print(f"\n✅ Results saved to: logistic_regression_results.csv")

    return all_results


if __name__ == "__main__":
    run()
