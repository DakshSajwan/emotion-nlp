"""Exploratory Data Analysis (EDA) module for Emotion Detection in Text.

Analyzes label distribution, text length profiles, and class balance across splits,
generating visualizations and summary statistics saved to reports/figures/ and reports/metrics/.
"""

from typing import Any
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, LABELS, METRICS_DIR
from src.data import get_splits


def analyze_dataset_distributions() -> dict[str, Any]:
    """Perform exploratory analysis on train, validation, and test splits.

    Returns:
        Dictionary containing summary tables, lengths, and split stats.
    """
    train_data, val_data, test_data = get_splits()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for split_name, split_dict in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        for text, label in zip(split_dict["text"], split_dict["label"]):
            word_count = len(text.split())
            char_count = len(text)
            records.append({
                "split": split_name,
                "text": text,
                "label": label,
                "word_count": word_count,
                "char_count": char_count,
            })

    df = pd.DataFrame(records)

    # 1. Class Distribution Table
    class_dist = pd.crosstab(df["label"], df["split"], margins=True)
    class_dist = class_dist.reindex(LABELS + ["All"])
    csv_path = METRICS_DIR / "class_distribution.csv"
    class_dist.to_csv(csv_path)

    # 2. Plot Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("muted", n_colors=len(LABELS))
    sns.countplot(
        data=df[df["split"] == "Train"],
        x="label",
        hue="label",
        order=LABELS,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title("Training Set Emotion Distribution (dair-ai/emotion)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 2),
            textcoords="offset points",
        )
    plt.tight_layout()
    class_dist_fig = FIGURES_DIR / "class_distribution.png"
    fig.savefig(class_dist_fig, dpi=300)
    plt.close(fig)

    # 3. Plot Text Length Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=df,
        x="word_count",
        hue="split",
        element="step",
        stat="density",
        common_norm=False,
        bins=30,
        ax=ax,
    )
    ax.set_title("Word Count Distribution Across Splits", fontsize=13, fontweight="bold")
    ax.set_xlabel("Words per Sample", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    plt.tight_layout()
    text_len_fig = FIGURES_DIR / "text_length_distribution.png"
    fig.savefig(text_len_fig, dpi=300)
    plt.close(fig)

    length_stats = {
        "mean_words": float(df["word_count"].mean()),
        "median_words": float(df["word_count"].median()),
        "max_words": int(df["word_count"].max()),
        "min_words": int(df["word_count"].min()),
        "p95_words": float(np.percentile(df["word_count"], 95)),
    }

    return {
        "class_distribution_csv": str(csv_path),
        "class_distribution_png": str(class_dist_fig),
        "text_length_png": str(text_len_fig),
        "length_stats": length_stats,
        "dataframe": df,
    }


def main() -> None:
    """CLI execution entrypoint for EDA."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    results = analyze_dataset_distributions()
    stats = results["length_stats"]
    print(f"Saved class distribution figure: {results['class_distribution_png']}")
    print(f"Saved text length figure:       {results['text_length_png']}")
    print(f"Saved distribution table:       {results['class_distribution_csv']}")
    print("\nText Length Statistics:")
    print(f"  Mean word count:   {stats['mean_words']:.2f}")
    print(f"  Median word count: {stats['median_words']:.1f}")
    print(f"  95th percentile:   {stats['p95_words']:.1f} words")
    print(f"  Max word count:    {stats['max_words']} words")
    print("=" * 60)


if __name__ == "__main__":
    main()
