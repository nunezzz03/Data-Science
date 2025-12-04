"""
Summary and Comparison of All Models - Lab 4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab4_config as config


def load_all_results():
    """Load results from all models."""
    results = []

    model_files = [
        "naive_bayes_results.csv",
        "logistic_regression_results.csv",
        "knn_results.csv",
        "decision_tree_results.csv",
        "mlp_results.csv",
        "random_forest_results.csv",
        "gradient_boosting_results.csv",
    ]

    for file in model_files:
        filepath = os.path.join(config.RESULTS_DIR, file)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results.append(df)

    if not results:
        return None

    return pd.concat(results, ignore_index=True)


def plot_comparison_by_dataset(df):
    """Plot comparison of all models for each dataset."""
    datasets = df["dataset"].unique()

    for dataset in datasets:
        data = df[df["dataset"] == dataset]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ["accuracy", "precision", "recall", "f1"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            models = data["model"].values
            values = data[metric].values

            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            bars = ax.bar(range(len(models)), values, color=colors)

            ax.set_xlabel("Model")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{dataset} - {metric.capitalize()}")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y")

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        file_tag = dataset.lower().replace(" ", "_")
        plt.savefig(
            os.path.join(config.IMAGES_DIR, f"{file_tag}_all_models_comparison.png"),
            dpi=150,
        )
        plt.close()

        print(f"   📊 Saved comparison chart for {dataset}")


def plot_overall_comparison(df):
    """Plot overall comparison across all datasets."""
    # Group by model and calculate mean metrics
    model_means = df.groupby("model")[["accuracy", "precision", "recall", "f1"]].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_means))
    width = 0.2
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            model_means[metric],
            width,
            label=metric.capitalize(),
            color=colors[i],
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score (Mean across datasets)")
    ax.set_title("Overall Model Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_means.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(config.IMAGES_DIR, "overall_model_comparison.png"), dpi=150
    )
    plt.close()

    print("   📊 Saved overall comparison chart")


def create_summary_table(df):
    """Create and save summary table."""
    # Pivot table for each dataset
    for dataset in df["dataset"].unique():
        data = df[df["dataset"] == dataset]

        # Create summary
        summary = data[
            ["model", "params", "accuracy", "precision", "recall", "f1"]
        ].copy()
        summary = summary.sort_values("f1", ascending=False)

        file_tag = dataset.lower().replace(" ", "_")
        summary.to_csv(
            os.path.join(config.RESULTS_DIR, f"{file_tag}_summary.csv"), index=False
        )

        print(f"\n   📋 Summary for {dataset}:")
        print(summary.to_string(index=False))

    # Overall summary
    overall = df.groupby("model")[["accuracy", "precision", "recall", "f1"]].mean()
    overall = overall.sort_values("f1", ascending=False)
    overall.to_csv(os.path.join(config.RESULTS_DIR, "overall_summary.csv"))

    print("\n   📋 Overall Summary (Mean across datasets):")
    print(overall.to_string())


def find_best_models(df):
    """Find and report best models."""
    print("\n" + "=" * 60)
    print("🏆 BEST MODELS")
    print("=" * 60)

    for dataset in df["dataset"].unique():
        data = df[df["dataset"] == dataset]
        best_idx = data["f1"].idxmax()
        best = data.loc[best_idx]

        print(f"\n   {dataset}:")
        print(f"      Model: {best['model']}")
        print(f"      Params: {best['params']}")
        print(f"      Accuracy: {best['accuracy']:.4f}")
        print(f"      Precision: {best['precision']:.4f}")
        print(f"      Recall: {best['recall']:.4f}")
        print(f"      F1: {best['f1']:.4f}")


def run():
    """Generate summary and comparisons."""
    print("\n" + "=" * 60)
    print("📊 GENERATING SUMMARY AND COMPARISONS")
    print("=" * 60)

    df = load_all_results()

    if df is None or df.empty:
        print("   ⚠️ No results found. Run model scripts first.")
        return

    print(f"   Found results for {len(df)} model-dataset combinations")

    # Generate visualizations
    plot_comparison_by_dataset(df)
    plot_overall_comparison(df)

    # Create summary tables
    create_summary_table(df)

    # Find best models
    find_best_models(df)

    print("\n✅ Summary complete!")
    print(f"   Results saved to: {config.RESULTS_DIR}")
    print(f"   Images saved to: {config.IMAGES_DIR}")


if __name__ == "__main__":
    run()
