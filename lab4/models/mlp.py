"""
Multi-Layer Perceptron Model - Lab 4
Based on Professor Claudia Antunes' DSLabs template
"""

import os
import sys
import warnings
from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, close
from sklearn.neural_network import MLPClassifier
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
)
import lab4_config as config


def mlp_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 250,
    metric: str = "accuracy",
) -> tuple[MLPClassifier | None, dict]:
    """
    Study MLP with different learning rates.
    Uses warm_start to show performance over iterations.
    Returns the best model and its parameters.
    """
    learning_rates: list[float] = [0.001, 0.01, 0.05, 0.1]
    hidden_layer_sizes: tuple = (100, 50)  # Fixed architecture for study

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0
    last_best: int = 0

    cols: int = len(learning_rates)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    for i, lr in enumerate(learning_rates):
        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        x_values: list[int] = []

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=lr,
            max_iter=lag,
            warm_start=True,
            random_state=config.RANDOM_STATE,
            activation="relu",
            solver="adam",
        )

        for n in range(lag, nr_max_iterations + 1, lag):
            clf.fit(trnX, trnY)
            prd_trn: array = clf.predict(trnX)
            prd_tst: array = clf.predict(tstX)
            y_trn_values.append(CLASS_EVAL_METRICS[metric](trnY, prd_trn))
            y_tst_values.append(CLASS_EVAL_METRICS[metric](tstY, prd_tst))
            x_values.append(n)

            eval_score = y_tst_values[-1]
            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (hidden_layer_sizes, lr, n)
                best_model = clf
                last_best = n
            elif n - last_best > 2 * lag:
                # Early stopping if no improvement
                break

        plot_multiline_chart(
            x_values,
            {"Train": y_trn_values, "Test": y_tst_values},
            ax=axs[0, i],
            title=f"MLP with lr={lr}",
            xlabel="iterations",
            ylabel=metric,
            percentage=True,
        )

    if best_params["params"]:
        print(
            f'MLP best for hidden={best_params["params"][0]} lr={best_params["params"][1]} iter={best_params["params"][2]}'
        )

    return best_model, best_params


def mlp_architecture_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    metric: str = "accuracy",
) -> tuple[MLPClassifier | None, dict]:
    """
    Study MLP with different architectures (hidden layer configurations).
    Returns the best model and its parameters.
    """
    architectures: list[tuple] = [
        (50,),
        (100,),
        (50, 25),
        (100, 50),
        (100, 50, 25),
    ]
    learning_rates: list[float] = [0.001, 0.01, 0.1]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    cols: int = len(architectures)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    for i, arch in enumerate(architectures):
        y_tst_values: list[float] = []

        for lr in learning_rates:
            clf = MLPClassifier(
                hidden_layer_sizes=arch,
                learning_rate_init=lr,
                max_iter=1000,
                random_state=config.RANDOM_STATE,
                activation="relu",
                solver="adam",
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval_score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval_score)

            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (arch, lr)
                best_model = clf

        arch_str = "x".join([str(x) for x in arch])
        plot_multiline_chart(
            learning_rates,
            {arch_str: y_tst_values},
            ax=axs[0, i],
            title=f"MLP with {arch_str} hidden",
            xlabel="learning rate",
            ylabel=metric,
            percentage=True,
        )

    if best_params["params"]:
        print(
            f'MLP best for hidden={best_params["params"][0]} lr={best_params["params"][1]}'
        )

    return best_model, best_params


def mlp_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, metric):
    """Study overfitting for MLP using iteration count."""
    best_arch = params["params"][0]
    best_lr = params["params"][1]

    nr_max_iterations = 3000
    lag = 100

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    x_values: list[int] = []

    clf = MLPClassifier(
        hidden_layer_sizes=best_arch,
        learning_rate_init=best_lr,
        max_iter=lag,
        warm_start=True,
        random_state=config.RANDOM_STATE,
        activation="relu",
        solver="adam",
    )

    for n in range(lag, nr_max_iterations + 1, lag):
        clf.fit(trnX, trnY)
        prd_trn: array = clf.predict(trnX)
        prd_tst: array = clf.predict(tstX)
        y_trn_values.append(CLASS_EVAL_METRICS["accuracy"](trnY, prd_trn))
        y_tst_values.append(CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst))
        x_values.append(n)

    figure()
    plot_multiline_chart(
        x_values,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"MLP overfitting study (hidden={best_arch}, lr={best_lr})",
        xlabel="iterations",
        ylabel="accuracy",
        percentage=True,
    )
    savefig(os.path.join(config.IMAGES_DIR, f"{file_tag}_mlp_{metric}_overfitting.png"))
    close()


def run_for_dataset(file_tag, target, eval_metric="accuracy"):
    """Run MLP study for a single dataset."""
    print(f"\n{'='*60}")
    print(f"🟣 MLP: {file_tag.upper()}")
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

    # 1. Architecture Study
    figure()
    best_model, params = mlp_architecture_study(
        trnX, trnY, tstX, tstY, metric=eval_metric
    )
    savefig(
        os.path.join(
            config.IMAGES_DIR, f"{file_tag}_mlp_{eval_metric}_architecture_study.png"
        )
    )
    close()

    if best_model is None:
        print("   ⚠️ No model found")
        return None

    # 2. Learning Rate / Iterations Study
    figure()
    best_model_iter, params_iter = mlp_study(trnX, trnY, tstX, tstY, metric=eval_metric)
    savefig(
        os.path.join(
            config.IMAGES_DIR, f"{file_tag}_mlp_{eval_metric}_iterations_study.png"
        )
    )
    close()

    # Use best model between architecture and iteration studies
    if best_model_iter is not None:
        prd_iter = best_model_iter.predict(tstX)
        prd_arch = best_model.predict(tstX)
        if CLASS_EVAL_METRICS[eval_metric](tstY, prd_iter) > CLASS_EVAL_METRICS[
            eval_metric
        ](tstY, prd_arch):
            best_model = best_model_iter
            params = params_iter

    # 3. Best Model Performance
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(
        os.path.join(
            config.IMAGES_DIR,
            f'{file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png',
        )
    )
    close()

    # 4. Overfitting Study
    mlp_overfitting_study(trnX, trnY, tstX, tstY, params, file_tag, eval_metric)

    # Return results
    return {
        "model": "MLP",
        "params": f"hidden={params['params'][0]}, lr={params['params'][1]}",
        "accuracy": CLASS_EVAL_METRICS["accuracy"](tstY, prd_tst),
        "precision": CLASS_EVAL_METRICS["precision"](tstY, prd_tst),
        "recall": CLASS_EVAL_METRICS["recall"](tstY, prd_tst),
        "f1": CLASS_EVAL_METRICS["f1"](tstY, prd_tst),
    }


def run():
    """Run MLP for all datasets."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("🟣 MLP MODELS")
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
        df.to_csv(os.path.join(config.RESULTS_DIR, "mlp_results.csv"), index=False)
        print(f"\n✅ Results saved to: mlp_results.csv")

    return all_results


if __name__ == "__main__":
    run()
