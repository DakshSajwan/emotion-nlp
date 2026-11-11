"""Evaluation metrics, reporting, and visualization module.

Single source of truth for evaluation across baseline and transformer models.
Generates standardized predictions files, metrics JSONs, classification reports,
confusion matrices, and error analysis.
"""

from typing import Optional, Union
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import FIGURES_DIR, LABELS, METRICS_DIR, PREDICTIONS_DIR


def save_predictions(
    model_name: str,
    texts: list[str],
    y_true: list[str],
    probs: np.ndarray,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save test predictions to CSV conforming strictly to the prediction file contract.

    Args:
        model_name: Name identifier for the model (e.g. 'baseline', 'distilbert').
        texts: List of raw input text strings.
        y_true: List of ground-truth label names.
        probs: Array of predicted class probabilities of shape (n, 6) in config.LABELS order.
        output_dir: Optional directory override (defaults to config.PREDICTIONS_DIR).

    Returns:
        Path to the saved predictions CSV file.
    """
    target_dir = output_dir or PREDICTIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / f"{model_name}_test_preds.csv"

    probs_arr = np.asarray(probs)
    pred_indices = np.argmax(probs_arr, axis=1)
    pred_labels = [LABELS[idx] for idx in pred_indices]
    confidences = np.max(probs_arr, axis=1)

    data: dict[str, list] = {
        "text": list(texts),
        "true_label": list(y_true),
        "pred_label": pred_labels,
        "confidence": [float(c) for c in confidences],
    }

    # Add probability columns in exact config.LABELS order
    for idx, label in enumerate(LABELS):
        data[f"prob_{label}"] = [float(p) for p in probs_arr[:, idx]]

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return filepath


def compute_metrics(
    y_true: Union[str, list[str]],
    y_pred: list[str],
    model_name: Optional[Union[str, list[str]]] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, float]:
    """Compute accuracy, macro-F1, and weighted-F1, and update metrics JSON.

    Supports both signatures:
    - compute_metrics(y_true, y_pred, model_name=None, output_dir=None)
    - compute_metrics(model_name, y_true, y_pred, output_dir=None)

    If a metrics JSON exists, it updates it by merging keys, never overwriting
    other existing keys (such as inference time or device info).

    Returns:
        Dictionary containing 'accuracy', 'macro_f1', and 'weighted_f1'.
    """
    if isinstance(y_true, str):
        actual_model_name: Optional[str] = y_true
        actual_y_true: list[str] = y_pred
        actual_y_pred: list[str] = model_name if isinstance(model_name, list) else []
    else:
        actual_y_true = y_true
        actual_y_pred = y_pred
        actual_model_name = model_name if isinstance(model_name, str) else None

    acc = float(accuracy_score(actual_y_true, actual_y_pred))
    macro_f1 = float(f1_score(actual_y_true, actual_y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(actual_y_true, actual_y_pred, average="weighted", zero_division=0))

    metrics: dict[str, float] = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }

    if actual_model_name:
        target_dir = output_dir or METRICS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{actual_model_name}.json"

        existing_data: dict[str, object] = {}
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}

        existing_data.update(metrics)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

    return metrics


def save_classification_report(
    model_name: str,
    y_true: list[str],
    y_pred: list[str],
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate and save detailed scikit-learn classification report as CSV.

    Args:
        model_name: Name identifier for the model.
        y_true: List of ground-truth label names.
        y_pred: List of predicted label names.
        output_dir: Optional directory override (defaults to config.METRICS_DIR).

    Returns:
        DataFrame containing precision, recall, f1-score, and support.
    """
    target_dir = output_dir or METRICS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / f"{model_name}_report.csv"

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(filepath, index=True)
    return df_report


def plot_confusion_matrix(
    model_name: str,
    y_true: list[str],
    y_pred: list[str],
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate and save row-normalized confusion matrix heatmap.

    Args:
        model_name: Name identifier for the model.
        y_true: List of ground-truth label names.
        y_pred: List of predicted label names.
        output_dir: Optional directory override (defaults to config.FIGURES_DIR).

    Returns:
        Path to the saved confusion matrix PNG file.
    """
    target_dir = output_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / f"{model_name}_confusion.png"

    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Normalized Confusion Matrix - {model_name.capitalize()}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)

    return filepath


def plot_per_class_f1(
    model_name: str,
    y_true: list[str],
    y_pred: list[str],
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate and save bar chart of per-class F1-scores.

    Args:
        model_name: Name identifier for the model.
        y_true: List of ground-truth label names.
        y_pred: List of predicted label names.
        output_dir: Optional directory override (defaults to config.FIGURES_DIR).

    Returns:
        Path to the saved per-class F1 PNG file.
    """
    target_dir = output_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / f"{model_name}_per_class_f1.png"

    f1_scores = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(LABELS, f1_scores, color=sns.color_palette("deep", len(LABELS)))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Per-Class F1 Score - {model_name.capitalize()}")
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("F1 Score")

    # Add numeric value annotations atop bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)

    return filepath


def top_confident_errors(
    model_name: str,
    n: int = 10,
    predictions_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract and save top N most confident prediction errors from predictions CSV.

    Args:
        model_name: Name identifier for the model.
        n: Number of highest-confidence error rows to save.
        predictions_dir: Optional directory containing predictions CSV.
        output_dir: Optional directory for saving top errors CSV.

    Returns:
        DataFrame containing the top N confident error instances.
    """
    pred_dir = predictions_dir or PREDICTIONS_DIR
    target_dir = output_dir or METRICS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    preds_file = pred_dir / f"{model_name}_test_preds.csv"
    if not preds_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_file}")

    df_preds = pd.read_csv(preds_file)
    errors = df_preds[df_preds["true_label"] != df_preds["pred_label"]].copy()

    if not errors.empty:
        top_errors = errors.sort_values(by="confidence", ascending=False).head(n)
    else:
        top_errors = errors

    out_file = target_dir / f"{model_name}_top_errors.csv"
    top_errors.to_csv(out_file, index=False)
    return top_errors


def evaluate_model(
    model_name: str,
    texts: list[str],
    y_true: list[str],
    probs: np.ndarray,
    output_dirs: Optional[dict[str, Path]] = None,
) -> dict[str, float]:
    """Run full evaluation suite: predictions, metrics, reports, plots, and error analysis.

    Args:
        model_name: Name identifier for the model.
        texts: List of raw input text strings.
        y_true: List of ground-truth label names.
        probs: Array of shape (n, 6) in config.LABELS order.
        output_dirs: Optional dictionary mapping 'predictions', 'metrics', and 'figures'
                     to custom Path locations (useful for isolated unit testing).

    Returns:
        Dictionary of headline metrics (accuracy, macro_f1, weighted_f1).
    """
    dirs = output_dirs or {}
    pred_dir = dirs.get("predictions", PREDICTIONS_DIR)
    metrics_dir = dirs.get("metrics", METRICS_DIR)
    fig_dir = dirs.get("figures", FIGURES_DIR)

    # 1. Save predictions CSV conforming to contract
    save_predictions(model_name, texts, y_true, probs, output_dir=pred_dir)

    # 2. Derive predicted labels from probabilities
    probs_arr = np.asarray(probs)
    pred_indices = np.argmax(probs_arr, axis=1)
    y_pred = [LABELS[idx] for idx in pred_indices]

    # 3. Compute and update metrics JSON
    metrics = compute_metrics(y_true, y_pred, model_name=model_name, output_dir=metrics_dir)

    # 4. Save scikit-learn classification report
    save_classification_report(model_name, y_true, y_pred, output_dir=metrics_dir)

    # 5. Plot normalized confusion matrix
    plot_confusion_matrix(model_name, y_true, y_pred, output_dir=fig_dir)

    # 6. Plot per-class F1 bar chart
    plot_per_class_f1(model_name, y_true, y_pred, output_dir=fig_dir)

    # 7. Extract top confident errors
    top_confident_errors(model_name, n=10, predictions_dir=pred_dir, output_dir=metrics_dir)

    # 8. Print headline metrics
    print(f"[{model_name.upper()}] Evaluation Complete:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")

    return metrics
