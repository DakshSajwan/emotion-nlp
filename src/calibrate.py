"""Model calibration module for temperature scaling and abstain threshold selection.

Provides temperature scaling to optimize expected calibration error (ECE) on multiclass logits,
generates reliability diagrams, and tunes confidence-based abstain thresholds using out-of-domain
probe distributions.
"""

from typing import Any, Optional
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    DATA_DIR,
    FIGURES_DIR,
    LABELS,
    MAX_LENGTH,
    MODELS_DIR,
    NUM_LABELS,
    SEED,
    label2id,
    set_seed,
)
from src.preprocess import minimal_clean


def extract_logits(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Run batched forward passes to extract unnormalized logits from text inputs.

    Args:
        model: Loaded sequence classification model.
        tokenizer: Loaded sequence tokenizer.
        texts: List of raw input text strings.
        device: PyTorch device (CPU or MPS/CUDA).
        batch_size: Batch size for tokenization and forward pass.

    Returns:
        PyTorch tensor of shape (N, NUM_LABELS) containing model logits on CPU.
    """
    model.eval()
    cleaned = [minimal_clean(t) for t in texts]
    all_logits: list[torch.Tensor] = []

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.cpu()
        all_logits.append(logits)

    return torch.cat(all_logits, dim=0)


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
    """Fit a scalar temperature parameter T > 0 on validation set logits using NLL.

    Args:
        val_logits: Tensor of shape (N, num_classes) of unscaled logits.
        val_labels: LongTensor of ground truth integer class IDs.

    Returns:
        Optimal scalar temperature float.
    """
    def nll_objective(t: float) -> float:
        scaled_logits = val_logits / t
        return float(F.cross_entropy(scaled_logits, val_labels).item())

    res = minimize_scalar(nll_objective, bounds=(0.05, 5.0), method="bounded")
    optimal_t = float(res.x)
    return optimal_t


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) across M equally spaced bins.

    Args:
        logits: Tensor of shape (N, num_classes) of logits.
        labels: LongTensor of ground truth integer class IDs.
        n_bins: Number of confidence bins (default: 10).

    Returns:
        Tuple of (ece_float, bin_statistics_list).
    """
    probs = F.softmax(logits, dim=1).numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.numpy()).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details: list[dict[str, Any]] = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = float(np.mean(in_bin))
        count = int(np.sum(in_bin))

        if count > 0:
            acc_bin = float(np.mean(accuracies[in_bin]))
            conf_bin = float(np.mean(confidences[in_bin]))
            ece += abs(conf_bin - acc_bin) * prop_in_bin
            bin_details.append({
                "bin_index": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": count,
                "prop": prop_in_bin,
                "accuracy": acc_bin,
                "confidence": conf_bin,
            })
        else:
            bin_details.append({
                "bin_index": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": 0,
                "prop": 0.0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
            })

    return round(float(ece), 4), bin_details


def plot_reliability_diagram(
    uncal_details: list[dict[str, Any]],
    cal_details: list[dict[str, Any]],
    uncal_ece: float,
    cal_ece: float,
    temperature: float,
    output_path: Path,
    model_name: str,
) -> Path:
    """Generate and save side-by-side reliability diagrams before and after calibration.

    Args:
        uncal_details: Per-bin statistics before temperature scaling.
        cal_details: Per-bin statistics after temperature scaling.
        uncal_ece: ECE score before scaling.
        cal_ece: ECE score after scaling.
        temperature: Learned temperature value T.
        output_path: File path to save the generated diagram.
        model_name: Name of model for titling.

    Returns:
        Path to saved PNG diagram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (axes[0], uncal_details, uncal_ece, f"Uncalibrated (T = 1.0000)\nECE = {uncal_ece:.4f}", "#1f77b4"),
        (axes[1], cal_details, cal_ece, f"Temperature Scaled (T = {temperature:.4f})\nECE = {cal_ece:.4f}", "#2ca02c"),
    ]

    for ax, details, ece_val, title, bar_color in panels:
        bin_centers = [(d["bin_lower"] + d["bin_upper"]) / 2.0 for d in details]
        accuracies = [d["accuracy"] for d in details]
        counts = [d["count"] for d in details]

        # Draw perfect calibration reference
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Perfect Calibration")

        # Draw accuracy bars
        ax.bar(
            bin_centers,
            accuracies,
            width=0.08,
            color=bar_color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
            label="Empirical Accuracy",
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_ylabel("Empirical Accuracy", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)

        # Annotate non-empty bins with count
        for bc, acc, cnt in zip(bin_centers, accuracies, counts):
            if cnt > 0:
                ax.text(bc, min(acc + 0.03, 1.0), f"n={cnt}", ha="center", va="bottom", fontsize=8, rotation=45)

    plt.suptitle(f"Reliability Diagram & Calibration: {model_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def choose_abstain_threshold(
    calibrated_probs: np.ndarray,
    label_names: list[str],
    min_coverage: float = 0.70,
    thresholds: Optional[list[float]] = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Search confidence thresholds on the probe dev split to maximize selective accuracy.

    Searches 0.40 to 0.90 in steps of 0.05 to find the threshold that maximizes accuracy on
    answered 6-class rows, subject to answering at least 70% of 6-class rows, while
    rejecting as many neutral rows as possible.

    Args:
        calibrated_probs: Array of shape (N, 6) of temperature-scaled probability predictions.
        label_names: List of ground-truth label names for each row in the dev split.
        min_coverage: Minimum fraction of 6-class samples answered (default: 0.70).
        thresholds: List of candidate thresholds (default: 0.40 to 0.90 step 0.05).

    Returns:
        Tuple of (best_threshold, trade_off_table_rows).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.40, 0.90 + 0.001, 0.05)]

    confidences = np.max(calibrated_probs, axis=1)
    pred_indices = np.argmax(calibrated_probs, axis=1)
    preds = [LABELS[i] for i in pred_indices]

    is_emo = [l in LABELS for l in label_names]
    is_neu = [l == "neutral" for l in label_names]

    n_emo = sum(is_emo)
    n_neu = sum(is_neu)

    table: list[dict[str, Any]] = []
    best_threshold = thresholds[0]
    best_acc = -1.0
    best_neu_rej = -1.0

    print("\n" + "=" * 67)
    print("  ABSTAIN THRESHOLD TRADE-OFF TABLE (PROBE SET: DEV SPLIT ONLY)")
    print("=" * 67)
    print(f"{'threshold':>9} | {'% answered':>10} | {'accuracy on answered':>20} | {'% neutral rejected':>18}")
    print("-" * 67)

    for t in thresholds:
        ans_mask = [is_emo[i] and confidences[i] >= t for i in range(len(label_names))]
        n_ans = sum(ans_mask)
        pct_ans = (n_ans / n_emo) * 100.0 if n_emo > 0 else 0.0

        if n_ans > 0:
            correct = sum(
                preds[i] == label_names[i] for i in range(len(label_names)) if ans_mask[i]
            )
            acc_ans = correct / n_ans
        else:
            acc_ans = 0.0

        neu_rej_mask = [is_neu[i] and confidences[i] < t for i in range(len(label_names))]
        n_neu_rej = sum(neu_rej_mask)
        pct_neu_rej = (n_neu_rej / n_neu) * 100.0 if n_neu > 0 else 0.0

        is_feasible = (pct_ans >= min_coverage * 100.0)

        row = {
            "threshold": t,
            "pct_answered": round(pct_ans, 2),
            "accuracy_on_answered": round(acc_ans, 4),
            "pct_neutral_rejected": round(pct_neu_rej, 2),
            "feasible": is_feasible,
        }
        table.append(row)

        print(f"{t:>9.2f} | {pct_ans:>9.1f}% | {acc_ans:>20.4f} | {pct_neu_rej:>17.1f}%")

        # Pick threshold that maximizes accuracy on answered rows, subject to coverage >= 70%,
        # tie-breaking on highest neutral rejected percentage, then higher threshold.
        if is_feasible:
            if (acc_ans > best_acc) or (
                np.isclose(acc_ans, best_acc, atol=1e-5) and pct_neu_rej > best_neu_rej
            ) or (
                np.isclose(acc_ans, best_acc, atol=1e-5)
                and np.isclose(pct_neu_rej, best_neu_rej, atol=1e-5)
                and t > best_threshold
            ):
                best_acc = acc_ans
                best_neu_rej = pct_neu_rej
                best_threshold = t

    print("=" * 67)
    print(f"Optimal Abstain Threshold Selected: {best_threshold:.2f}\n")
    return best_threshold, table


def calibrate_model(model_name: str = "distilbert_v2") -> dict[str, Any]:
    """Execute full calibration pipeline for the designated model architecture.

    1. Fits temperature T on original data/val.csv logits.
    2. Evaluates ECE (10 bins) on data/test.csv before and after scaling, saving a reliability diagram.
    3. Finds the optimal confidence abstain threshold using the 'dev' split of data/probe_set.csv.
    4. Serializes calibration metadata to models/<model_name>/calibration.json.

    Args:
        model_name: Model subfolder name inside models/ (default: 'distilbert_v2').

    Returns:
        Dictionary containing calibration hyperparameters and evaluation statistics.
    """
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"\n==========================================")
    print(f"Calibrating Model: {model_name.upper()}")
    print(f"Model Directory: {model_dir}")
    print(f"Device: {device}")
    print(f"==========================================\n")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    # 1. Load logits on original validation split (data/val.csv)
    val_csv = DATA_DIR / "val.csv"
    print(f"Loading validation split from {val_csv}...")
    df_val = pd.read_csv(val_csv)
    val_labels = torch.tensor([label2id[l] for l in df_val["label"]], dtype=torch.long)
    val_logits = extract_logits(model, tokenizer, df_val["text"].tolist(), device=device)

    print("Fitting optimal temperature T via negative log-likelihood minimization...")
    optimal_t = fit_temperature(val_logits, val_labels)
    nll_uncal = float(F.cross_entropy(val_logits, val_labels).item())
    nll_cal = float(F.cross_entropy(val_logits / optimal_t, val_labels).item())
    print(f"Optimal Temperature T: {optimal_t:.4f} (Val NLL: {nll_uncal:.4f} -> {nll_cal:.4f})")

    # 2. Evaluate Expected Calibration Error on data/test.csv before & after scaling
    test_csv = DATA_DIR / "test.csv"
    print(f"Evaluating ECE on test split from {test_csv}...")
    df_test = pd.read_csv(test_csv)
    test_labels = torch.tensor([label2id[l] for l in df_test["label"]], dtype=torch.long)
    test_logits = extract_logits(model, tokenizer, df_test["text"].tolist(), device=device)

    ece_before, details_before = compute_ece(test_logits, test_labels, n_bins=10)
    ece_after, details_after = compute_ece(test_logits / optimal_t, test_labels, n_bins=10)
    print(f"ECE (10 bins) on test.csv: Before = {ece_before:.4f} | After = {ece_after:.4f}")

    diagram_path = FIGURES_DIR / f"{model_name}_calibration.png"
    plot_reliability_diagram(
        details_before,
        details_after,
        ece_before,
        ece_after,
        optimal_t,
        diagram_path,
        model_name,
    )
    print(f"Reliability diagram saved to: {diagram_path}")

    # 3. Choose abstain threshold using ONLY the 'dev' split of data/probe_set.csv
    probe_csv = DATA_DIR / "probe_set.csv"
    print(f"Loading probe dev split from {probe_csv} (excluding test split)...")
    df_probe = pd.read_csv(probe_csv)
    df_dev = df_probe[df_probe["split"] == "dev"].copy()

    dev_texts = df_dev["text"].tolist()
    dev_labels = df_dev["label_name"].tolist()
    dev_logits = extract_logits(model, tokenizer, dev_texts, device=device)
    dev_cal_probs = F.softmax(dev_logits / optimal_t, dim=1).numpy()

    best_threshold, trade_off_table = choose_abstain_threshold(dev_cal_probs, dev_labels)

    # 4. Save calibration hyperparameters to models/<model_name>/calibration.json
    selected_row = next(r for r in trade_off_table if r["threshold"] == best_threshold)
    cal_data: dict[str, Any] = {
        "model_name": model_name,
        "temperature": round(optimal_t, 4),
        "T": round(optimal_t, 4),
        "abstain_threshold": float(best_threshold),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "val_nll_before": round(nll_uncal, 4),
        "val_nll_after": round(nll_cal, 4),
        "dev_pct_answered": selected_row["pct_answered"],
        "dev_accuracy_on_answered": selected_row["accuracy_on_answered"],
        "dev_pct_neutral_rejected": selected_row["pct_neutral_rejected"],
    }

    cal_path = model_dir / "calibration.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(cal_data, f, indent=2)

    print(f"Calibration metadata successfully saved to: {cal_path}")
    print(json.dumps(cal_data, indent=2))
    return cal_data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model calibration."""
    parser = argparse.ArgumentParser(description="Calibrate emotion detection model.")
    parser.add_argument(
        "model_pos",
        nargs="?",
        default=None,
        help="Model directory name inside models/ (positional, default: 'distilbert_v2').",
    )
    parser.add_argument(
        "--model_name",
        "--model",
        dest="model_flag",
        type=str,
        default=None,
        help="Model directory name inside models/ (default: 'distilbert_v2').",
    )
    parsed = parser.parse_args()
    target_model = parsed.model_flag or parsed.model_pos or "distilbert_v2"
    parsed.model_name = target_model
    return parsed


if __name__ == "__main__":
    args = parse_args()
    calibrate_model(model_name=args.model_name)
