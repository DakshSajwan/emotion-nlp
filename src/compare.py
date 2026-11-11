"""Model comparison and synthesis module for Emotion Detection in Text.

Reads predictions and metrics for baseline and DistilBERT models, generating:
- reports/metrics/comparison_table.csv and comparison_table.md
- reports/figures/compare_per_class_f1.png (grouped bar chart)
- reports/figures/compare_overall.png (grouped bar chart)
- reports/metrics/error_analysis.md (10 baseline wrong/DistilBERT right, 10 DistilBERT wrong/baseline right, 10 both wrong)
"""

from typing import Any, Optional
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

from src.config import FIGURES_DIR, LABELS, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR


def check_required_files(
    predictions_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
) -> tuple[bool, list[str]]:
    """Verify that required predictions and metrics files exist for baseline and distilbert.

    Strictly ignores any files named '*_smoke*'.

    Args:
        predictions_dir: Directory containing predictions CSV files.
        metrics_dir: Directory containing metrics JSON files.

    Returns:
        Tuple of (all_files_exist_bool, list_of_missing_file_descriptions).
    """
    p_dir = predictions_dir or PREDICTIONS_DIR
    m_dir = metrics_dir or METRICS_DIR

    required = {
        "Baseline Predictions": p_dir / "baseline_test_preds.csv",
        "Baseline Metrics": m_dir / "baseline.json",
        "DistilBERT Predictions": p_dir / "distilbert_test_preds.csv",
        "DistilBERT Metrics": m_dir / "distilbert.json",
    }

    missing: list[str] = []
    for desc, path in required.items():
        if not path.exists():
            missing.append(f"{desc}: {path}")

    return (len(missing) == 0), missing


def load_model_metrics(metrics_file: Path, model_file_or_dir: Optional[Path] = None) -> dict[str, Any]:
    """Load and parse metrics JSON, computing model size if not present.

    Args:
        metrics_file: Path to model metrics JSON.
        model_file_or_dir: Optional Path to model artifact for disk footprint calculation.

    Returns:
        Dictionary of model metrics.
    """
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "model_size_mb" not in data and model_file_or_dir and model_file_or_dir.exists():
        if model_file_or_dir.is_file():
            data["model_size_mb"] = round(model_file_or_dir.stat().st_size / (1024 * 1024), 2)
        elif model_file_or_dir.is_dir():
            total = sum(f.stat().st_size for f in model_file_or_dir.glob("**/*") if f.is_file())
            data["model_size_mb"] = round(total / (1024 * 1024), 2)

    return data


def df_to_markdown(data: Any, index: bool = False) -> str:
    """Format DataFrame or Series as a Markdown table without external dependencies."""
    if isinstance(data, pd.Series):
        df = data.reset_index()
        df.columns = [str(c) for c in df.columns]
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if index:
            df = df.reset_index()
    else:
        return str(data)

    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_vals = [str(val) for val in row]
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def generate_comparison_table(
    baseline_metrics: dict[str, Any],
    distilbert_metrics: dict[str, Any],
    v2_metrics: Optional[dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate comparison table CSV and Markdown files.

    Args:
        baseline_metrics: Dictionary of baseline metrics.
        distilbert_metrics: Dictionary of DistilBERT metrics.
        v2_metrics: Optional dictionary of DistilBERT v2 metrics.
        output_dir: Optional directory for saving comparison table.

    Returns:
        DataFrame containing the comparison summary.
    """
    target_dir = output_dir or METRICS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "Model": "Baseline (TF-IDF + LR)",
            "Accuracy": baseline_metrics.get("accuracy", 0.0),
            "Macro-F1": baseline_metrics.get("macro_f1", 0.0),
            "Weighted-F1": baseline_metrics.get("weighted_f1", 0.0),
            "Inference Time (ms / 1k samples)": baseline_metrics.get(
                "inference_time_per_1000_samples_ms", 0.0
            ),
            "Model Size (MB)": baseline_metrics.get("model_size_mb", 0.0),
        },
        {
            "Model": "DistilBERT v1",
            "Accuracy": distilbert_metrics.get("accuracy", 0.0),
            "Macro-F1": distilbert_metrics.get("macro_f1", 0.0),
            "Weighted-F1": distilbert_metrics.get("weighted_f1", 0.0),
            "Inference Time (ms / 1k samples)": distilbert_metrics.get(
                "inference_time_per_1000_samples_ms", 0.0
            ),
            "Model Size (MB)": distilbert_metrics.get("model_size_mb", 0.0),
        },
    ]

    if v2_metrics is not None:
        rows.append({
            "Model": "DistilBERT v2 (Improved)",
            "Accuracy": v2_metrics.get("accuracy", 0.0),
            "Macro-F1": v2_metrics.get("macro_f1", 0.0),
            "Weighted-F1": v2_metrics.get("weighted_f1", 0.0),
            "Inference Time (ms / 1k samples)": v2_metrics.get(
                "inference_time_per_1000_samples_ms", 0.0
            ),
            "Model Size (MB)": v2_metrics.get("model_size_mb", 0.0),
        })

    df = pd.DataFrame(rows)
    csv_path = target_dir / "comparison_table.csv"
    md_path = target_dir / "comparison_table.md"

    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df_to_markdown(df, index=False))
    print(f"Comparison tables saved to:\n  - {csv_path}\n  - {md_path}")
    return df


def plot_grouped_per_class_f1(
    df_base: pd.DataFrame,
    df_dist: pd.DataFrame,
    df_v2: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Plot grouped bar chart comparing per-class F1 for Baseline, DistilBERT, and optionally DistilBERT v2.

    Args:
        df_base: Predictions DataFrame for baseline.
        df_dist: Predictions DataFrame for DistilBERT.
        df_v2: Optional predictions DataFrame for DistilBERT v2.
        output_dir: Optional directory override.

    Returns:
        Path to the saved PNG figure.
    """
    target_dir = output_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / "compare_per_class_f1.png"

    f1_base = f1_score(
        df_base["true_label"], df_base["pred_label"], labels=LABELS, average=None, zero_division=0
    )
    f1_dist = f1_score(
        df_dist["true_label"], df_dist["pred_label"], labels=LABELS, average=None, zero_division=0
    )

    x = np.arange(len(LABELS))

    def _annotate_bars(bars, ax_obj):
        for bar in bars:
            height = bar.get_height()
            ax_obj.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig, ax = plt.subplots(figsize=(11, 6))

    if df_v2 is not None:
        f1_v2 = f1_score(
            df_v2["true_label"], df_v2["pred_label"], labels=LABELS, average=None, zero_division=0
        )
        width = 0.26
        bars1 = ax.bar(x - width, f1_base, width, label="Baseline (TF-IDF + LR)", color="#4c72b0")
        bars2 = ax.bar(x, f1_dist, width, label="DistilBERT v1", color="#55a868")
        bars3 = ax.bar(x + width, f1_v2, width, label="DistilBERT v2 (Improved)", color="#9b59b6")
        _annotate_bars(bars1, ax)
        _annotate_bars(bars2, ax)
        _annotate_bars(bars3, ax)
        ax.set_title("Per-Class F1 Score Comparison (Baseline vs DistilBERT v1 vs DistilBERT v2)")
    else:
        width = 0.36
        bars1 = ax.bar(x - width / 2, f1_base, width, label="Baseline (TF-IDF + LR)", color="#4c72b0")
        bars2 = ax.bar(x + width / 2, f1_dist, width, label="DistilBERT", color="#55a868")
        _annotate_bars(bars1, ax)
        _annotate_bars(bars2, ax)
        ax.set_title("Per-Class F1 Score Comparison (Baseline vs DistilBERT)")

    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0.0, 1.12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Per-class F1 comparison chart saved to: {filepath}")
    return filepath


def plot_grouped_overall_metrics(
    baseline_metrics: dict[str, Any],
    distilbert_metrics: dict[str, Any],
    v2_metrics: Optional[dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Plot grouped bar chart comparing overall Accuracy, Macro-F1, and Weighted-F1.

    Args:
        baseline_metrics: Baseline metrics dictionary.
        distilbert_metrics: DistilBERT metrics dictionary.
        v2_metrics: Optional DistilBERT v2 metrics dictionary.
        output_dir: Optional directory override.

    Returns:
        Path to the saved PNG figure.
    """
    target_dir = output_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / "compare_overall.png"

    metrics_names = ["Accuracy", "Macro-F1", "Weighted-F1"]
    base_scores = [
        baseline_metrics.get("accuracy", 0.0),
        baseline_metrics.get("macro_f1", 0.0),
        baseline_metrics.get("weighted_f1", 0.0),
    ]
    dist_scores = [
        distilbert_metrics.get("accuracy", 0.0),
        distilbert_metrics.get("macro_f1", 0.0),
        distilbert_metrics.get("weighted_f1", 0.0),
    ]

    x = np.arange(len(metrics_names))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if v2_metrics is not None:
        v2_scores = [
            v2_metrics.get("accuracy", 0.0),
            v2_metrics.get("macro_f1", 0.0),
            v2_metrics.get("weighted_f1", 0.0),
        ]
        width = 0.24
        bars1 = ax.bar(x - width, base_scores, width, label="Baseline (TF-IDF + LR)", color="#4c72b0")
        bars2 = ax.bar(x, dist_scores, width, label="DistilBERT v1", color="#c44e52")
        bars3 = ax.bar(x + width, v2_scores, width, label="DistilBERT v2 (Improved)", color="#2ca02c")
        all_bars = [bars1, bars2, bars3]
        ax.set_title("Overall Model Performance Comparison (Baseline vs v1 vs v2)")
    else:
        width = 0.32
        bars1 = ax.bar(x - width / 2, base_scores, width, label="Baseline (TF-IDF + LR)", color="#4c72b0")
        bars2 = ax.bar(x + width / 2, dist_scores, width, label="DistilBERT", color="#c44e52")
        all_bars = [bars1, bars2]
        ax.set_title("Overall Model Performance Comparison")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0.0, 1.12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Overall metrics comparison chart saved to: {filepath}")
    return filepath


def generate_error_analysis(
    df_base: pd.DataFrame,
    df_dist: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate detailed qualitative and quantitative error analysis markdown document.

    Extracts:
    - 10 examples where baseline is wrong and DistilBERT is right
    - 10 examples where DistilBERT is wrong and baseline is right
    - 10 examples where both are wrong

    Args:
        df_base: Baseline test predictions DataFrame.
        df_dist: DistilBERT test predictions DataFrame.
        output_dir: Optional directory override.

    Returns:
        Path to the saved error_analysis.md.
    """
    target_dir = output_dir or METRICS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / "error_analysis.md"

    base_correct = df_base["true_label"] == df_base["pred_label"]
    dist_correct = df_dist["true_label"] == df_dist["pred_label"]

    base_wrong_dist_right = df_base[~base_correct & dist_correct].copy()
    dist_wrong_base_right = df_dist[~dist_correct & base_correct].copy()
    both_wrong = df_base[~base_correct & ~dist_correct].copy()

    def _format_table(indices: list[int]) -> str:
        if not indices:
            return "*No examples found.*"
        lines = [
            "| # | Text | True Label | Baseline Pred (Conf) | DistilBERT Pred (Conf) |",
            "|---|---|---|---|---|",
        ]
        for num, idx in enumerate(indices, 1):
            text = df_base.loc[idx, "text"]
            true_lbl = df_base.loc[idx, "true_label"]
            b_pred = df_base.loc[idx, "pred_label"]
            b_conf = df_base.loc[idx, "confidence"]
            d_pred = df_dist.loc[idx, "pred_label"]
            d_conf = df_dist.loc[idx, "confidence"]
            # Clean newlines in text
            clean_text = text.replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {num} | {clean_text} | **{true_lbl}** | {b_pred} ({b_conf:.2f}) | {d_pred} ({d_conf:.2f}) |"
            )
        return "\n".join(lines)

    # 1. Baseline wrong, DistilBERT right
    # Sort by baseline confidence (confident baseline mistakes fixed by DistilBERT)
    idx_b_wrong = (
        base_wrong_dist_right.sort_values(by="confidence", ascending=False).head(10).index.tolist()
    )

    # 2. DistilBERT wrong, Baseline right
    idx_d_wrong = (
        dist_wrong_base_right.sort_values(by="confidence", ascending=False).head(10).index.tolist()
    )

    # 3. Both wrong
    idx_both_wrong = (
        both_wrong.sort_values(by="confidence", ascending=False).head(10).index.tolist()
    )

    # Frequent confusion pairs analysis
    base_errors = df_base[~base_correct]
    dist_errors = df_dist[~dist_correct]

    base_confusion_pairs = (
        base_errors.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False).head(5)
    )
    dist_confusion_pairs = (
        dist_errors.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False).head(5)
    )

    both_errors_df = pd.DataFrame({
        "true": df_base.loc[both_wrong.index, "true_label"],
        "base_pred": df_base.loc[both_wrong.index, "pred_label"],
        "dist_pred": df_dist.loc[both_wrong.index, "pred_label"],
    })
    common_confusions = (
        both_errors_df.groupby(["true", "base_pred", "dist_pred"])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )

    content = f"""# Error Analysis: Baseline vs DistilBERT

This document provides a comparative error breakdown on the 2,000-sample test set of `dair-ai/emotion`.

## 1. Top Confident Discrepancies

### A. Baseline Wrong, DistilBERT Right (Total: {len(base_wrong_dist_right)} instances)
Cases where transformer contextual representations successfully resolved nuanced syntactic or semantic structures that fooled bag-of-words:

{_format_table(idx_b_wrong)}

---

### B. DistilBERT Wrong, Baseline Right (Total: {len(dist_wrong_base_right)} instances)
Cases where classical lexical matching succeeded but the transformer misclassified:

{_format_table(idx_d_wrong)}

---

### C. Both Models Wrong (Total: {len(both_wrong)} instances)
Intrinsically ambiguous, overlapping, or multi-faceted emotion examples where both classifiers failed:

{_format_table(idx_both_wrong)}

---

## 2. Most Frequent Confusion Pairs

### Baseline Most Frequent Confusion Pairs (True -> Pred)
{df_to_markdown(base_confusion_pairs)}

### DistilBERT Most Frequent Confusion Pairs (True -> Pred)
{df_to_markdown(dist_confusion_pairs)}

### Common Failure Modes Where Both Models Agree on False Predictions
{df_to_markdown(common_confusions)}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Error analysis report saved to: {filepath}")
    return filepath


def run_comparison(
    predictions_dir: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> bool:
    """Execute model comparison suite.

    Returns:
        True if comparison succeeded, False if required files were missing.
    """
    p_dir = predictions_dir or PREDICTIONS_DIR
    m_dir = metrics_dir or METRICS_DIR
    f_dir = figures_dir or FIGURES_DIR

    # Check required files
    files_ok, missing = check_required_files(p_dir, m_dir)
    if not files_ok:
        print("\n" + "=" * 68)
        print("MISSING REQUIRED EVALUATION OR PREDICTION FILES:")
        for m in missing:
            print(f"  [MISSING] {m}")
        print("\nACTION REQUIRED TO PROCEED:")
        if any("DistilBERT" in m for m in missing):
            print("  Run the full DistilBERT training and evaluation first:")
            print("      .venv/bin/python -m src.transformer --epochs 3")
        if any("Baseline" in m for m in missing):
            print("  Run the baseline model training and evaluation first:")
            print("      .venv/bin/python -m src.baseline")
        print("=" * 68 + "\n")
        return False

    print("Loading prediction files (ignoring anything named '*_smoke*')...")
    df_base = pd.read_csv(p_dir / "baseline_test_preds.csv")
    df_dist = pd.read_csv(p_dir / "distilbert_test_preds.csv")

    print("Loading metrics files...")
    base_metrics = load_model_metrics(m_dir / "baseline.json", MODELS_DIR / "baseline.joblib")
    dist_metrics = load_model_metrics(m_dir / "distilbert.json", MODELS_DIR / "distilbert")

    # Optional DistilBERT v2 integration when files exist
    v2_preds_path = p_dir / "distilbert_v2_test_preds.csv"
    v2_metrics_path = m_dir / "distilbert_v2.json"
    has_v2 = v2_preds_path.exists() and v2_metrics_path.exists()

    df_v2: Optional[pd.DataFrame] = None
    v2_metrics: Optional[dict[str, Any]] = None
    if has_v2:
        print("Detected DistilBERT v2 artifacts. Including v2 in comparison...")
        df_v2 = pd.read_csv(v2_preds_path)
        v2_metrics = load_model_metrics(v2_metrics_path, MODELS_DIR / "distilbert_v2")

    # 1. Comparison table
    generate_comparison_table(base_metrics, dist_metrics, v2_metrics=v2_metrics, output_dir=m_dir)

    # 2. Grouped per-class F1 chart
    plot_grouped_per_class_f1(df_base, df_dist, df_v2=df_v2, output_dir=f_dir)

    # 3. Grouped overall metrics chart
    plot_grouped_overall_metrics(base_metrics, dist_metrics, v2_metrics=v2_metrics, output_dir=f_dir)

    # 4. Error analysis report
    generate_error_analysis(df_base, df_dist, output_dir=m_dir)

    print("\nModel comparison completed successfully.")
    return True


if __name__ == "__main__":
    success = run_comparison()
    if not success:
        sys.exit(1)
