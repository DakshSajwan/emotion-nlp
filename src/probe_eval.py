"""Independent probe evaluation module for Emotion Detection in Text.

Evaluates trained baseline and DistilBERT models on independent probe sentences
simulating live faculty testing, including out-of-domain neutral sentence detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import DATA_DIR, LABELS, METRICS_DIR
from src.predict import EmotionPredictor


def evaluate_probe(
    model_type: str,
    split: str = "test",
    probe_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate an emotion predictor on the independent probe dataset.

    Computes multi-class classification accuracy, macro-F1, and per-class F1
    scores on emotional sentences, exports error analysis CSVs, and computes
    the falsely confident prediction rate on neutral sentences.

    Args:
        model_type: Classifier architecture ('baseline' or 'distilbert').
        split: Dataset split to evaluate ('dev', 'test', or 'all').
        probe_csv_path: Optional path to probe_set.csv. Defaults to DATA_DIR / 'probe_set.csv'.

    Returns:
        Dictionary containing computed metrics and neutral false-confidence rate.
    """
    valid_models = ["baseline", "distilbert", "distilbert_v2"]
    if model_type not in valid_models:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {valid_models}.")

    if split not in ["dev", "test", "all"]:
        raise ValueError(f"Invalid split '{split}'. Must be 'dev', 'test', or 'all'.")

    target_path = Path(probe_csv_path) if probe_csv_path else DATA_DIR / "probe_set.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"Probe dataset not found at: {target_path}")

    df = pd.read_csv(target_path)

    # Filter by split if not 'all'
    if split != "all":
        df = df[df["split"] == split].copy()

    # Separate emotional rows and neutral rows
    df_emo = df[df["label_name"].isin(LABELS)].copy()
    df_neutral = df[df["label_name"] == "neutral"].copy()

    predictor = EmotionPredictor(model_type=model_type)

    # 1. Evaluate emotional rows
    emo_texts = df_emo["text"].tolist()
    y_true = df_emo["label_name"].tolist()

    emo_preds = predictor.predict_batch(emo_texts)
    y_pred = [p["label"] for p in emo_preds]
    confidences = [p["confidence"] for p in emo_preds]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    per_class_f1 = {label: float(report_dict[label]["f1-score"]) for label in LABELS}

    # Extract misclassified emotional rows
    errors = []
    for text, true_lbl, pred_lbl, conf in zip(emo_texts, y_true, y_pred, confidences):
        if true_lbl != pred_lbl:
            errors.append({
                "text": text,
                "true": true_lbl,
                "pred": pred_lbl,
                "confidence": round(conf, 4),
            })

    errors_df = pd.DataFrame(errors)
    errors_csv_path = METRICS_DIR / f"probe_{model_type}_{split}_errors.csv"
    errors_df.to_csv(errors_csv_path, index=False)

    # 2. Evaluate neutral rows (OOD false confidence test)
    neutral_texts = df_neutral["text"].tolist()
    if neutral_texts:
        neutral_preds = predictor.predict_batch(neutral_texts)
        neutral_confs = [p["confidence"] for p in neutral_preds]
        has_is_confident = any("is_confident" in p for p in neutral_preds)
        if has_is_confident:
            falsely_confident_count = sum(1 for p in neutral_preds if p.get("is_confident", False))
        else:
            falsely_confident_count = sum(1 for c in neutral_confs if c >= 0.80)
        falsely_confident_rate = float(falsely_confident_count / len(neutral_confs))
        neutral_mean_conf = float(np.mean(neutral_confs))
    else:
        has_is_confident = False
        falsely_confident_count = 0
        falsely_confident_rate = 0.0
        neutral_mean_conf = 0.0

    # Confident-only evaluation if model supports abstention
    has_emo_confident = any("is_confident" in p for p in emo_preds)
    confident_stats: dict[str, Any] = {}
    if has_emo_confident:
        ans_mask = [p.get("is_confident", True) for p in emo_preds]
        n_ans = sum(ans_mask)
        pct_ans = float(n_ans / len(emo_preds)) * 100.0 if emo_preds else 0.0
        if n_ans > 0:
            y_t_ans = [y for y, a in zip(y_true, ans_mask) if a]
            y_p_ans = [y for y, a in zip(y_pred, ans_mask) if a]
            acc_ans = float(accuracy_score(y_t_ans, y_p_ans))
            f1_ans = float(f1_score(y_t_ans, y_p_ans, average="macro", zero_division=0))
        else:
            acc_ans = 0.0
            f1_ans = 0.0
        confident_stats = {
            "confident_only_accuracy": round(acc_ans, 4),
            "confident_only_macro_f1": round(f1_ans, 4),
            "pct_answered": round(pct_ans, 2),
            "answered_samples": n_ans,
        }

    # Package output metrics
    results: dict[str, Any] = {
        "model_type": model_type,
        "split": split,
        "emotional_samples": len(df_emo),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
        "neutral_samples": len(df_neutral),
        "falsely_confident_neutral_count": falsely_confident_count,
        "falsely_confident_neutral_rate": round(falsely_confident_rate, 4),
        "neutral_mean_confidence": round(neutral_mean_conf, 4),
        **confident_stats,
    }

    # Save metrics JSON
    metrics_json_path = METRICS_DIR / f"probe_{model_type}_{split}.json"
    metrics_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print summary to console
    print("\n" + "=" * 65)
    print(f"PROBE EVALUATION: {model_type.upper()} on '{split}' split")
    print("=" * 65)
    print(f"Emotional Samples : {len(df_emo)}")
    print(f"Accuracy          : {acc * 100:.2f}%")
    print(f"Macro-F1          : {macro_f1 * 100:.2f}%")
    print(f"Weighted-F1       : {weighted_f1 * 100:.2f}%")
    if confident_stats:
        print(f"Answered Samples  : {confident_stats['answered_samples']}/{len(df_emo)} ({confident_stats['pct_answered']:.1f}%)")
        print(f"Confident Accuracy: {confident_stats['confident_only_accuracy'] * 100:.2f}%")
        print(f"Confident Macro-F1: {confident_stats['confident_only_macro_f1'] * 100:.2f}%")
    print("\nPer-Class F1 Scores:")
    for label in LABELS:
        print(f"  - {label:<10}: {per_class_f1[label] * 100:.2f}%")
    print(f"\nNeutral Samples   : {len(df_neutral)}")
    print(f"Falsely Confident : {falsely_confident_count}/{len(df_neutral)} ({falsely_confident_rate * 100:.2f}%)")
    print(f"Mean Neutral Conf : {neutral_mean_conf * 100:.2f}%")
    print(f"Saved Metrics     : {metrics_json_path}")
    print(f"Saved Errors      : {errors_csv_path} ({len(errors)} errors)")
    print("=" * 65 + "\n")

    return results


def main() -> None:
    """CLI entry point for independent probe set evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Emotion Detection models on independent probe set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "distilbert", "distilbert_v2"],
        default="baseline",
        help="Classifier model architecture to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test", "all"],
        default="test",
        help="Probe dataset partition to evaluate",
    )

    args = parser.parse_args()
    evaluate_probe(model_type=args.model, split=args.split)


if __name__ == "__main__":
    main()
