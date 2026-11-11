"""Unit tests for the evaluation module using isolated temporary directories."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.config import LABELS
from src.evaluate import compute_metrics, evaluate_model


def test_evaluate_model_pipeline_with_dummy_data(tmp_path: Path) -> None:
    """Verify evaluate_model runs full pipeline and creates all contracted artifacts."""
    output_dirs = {
        "predictions": tmp_path / "predictions",
        "metrics": tmp_path / "metrics",
        "figures": tmp_path / "figures",
    }

    dummy_texts = [
        "I feel so down today",
        "Today is a wonderful sunny day",
        "I love spending time with family",
        "This makes me furious and annoyed",
        "I am scared of the dark",
        "What a shocking turn of events",
    ]
    dummy_y_true = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # Construct probabilities (6 samples, 6 classes) with one intentional misclassification
    # Sample 4 ('fear') intentionally predicted as 'anger' with high confidence
    dummy_probs = np.array([
        [0.85, 0.03, 0.02, 0.04, 0.03, 0.03],  # sadness -> sadness
        [0.02, 0.90, 0.03, 0.02, 0.01, 0.02],  # joy -> joy
        [0.05, 0.05, 0.80, 0.04, 0.03, 0.03],  # love -> love
        [0.01, 0.02, 0.02, 0.88, 0.04, 0.03],  # anger -> anger
        [0.02, 0.01, 0.02, 0.75, 0.15, 0.05],  # fear -> anger (error)
        [0.04, 0.03, 0.03, 0.02, 0.03, 0.85],  # surprise -> surprise
    ])

    metrics = evaluate_model(
        "dummy_test_model",
        dummy_texts,
        dummy_y_true,
        dummy_probs,
        output_dirs=output_dirs,
    )

    # 1. Check returned metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert metrics["accuracy"] == pytest.approx(5 / 6, rel=1e-3)

    # 2. Check predictions CSV and contract columns
    preds_file = output_dirs["predictions"] / "dummy_test_model_test_preds.csv"
    assert preds_file.exists()
    df_preds = pd.read_csv(preds_file)

    expected_cols = ["text", "true_label", "pred_label", "confidence"] + [f"prob_{l}" for l in LABELS]
    assert list(df_preds.columns) == expected_cols
    assert len(df_preds) == 6

    # 3. Check metrics JSON
    metrics_file = output_dirs["metrics"] / "dummy_test_model.json"
    assert metrics_file.exists()
    with open(metrics_file, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["accuracy"] == metrics["accuracy"]

    # 4. Check classification report CSV
    report_file = output_dirs["metrics"] / "dummy_test_model_report.csv"
    assert report_file.exists()

    # 5. Check figures
    conf_fig = output_dirs["figures"] / "dummy_test_model_confusion.png"
    f1_fig = output_dirs["figures"] / "dummy_test_model_per_class_f1.png"
    assert conf_fig.exists() and conf_fig.stat().st_size > 0
    assert f1_fig.exists() and f1_fig.stat().st_size > 0

    # 6. Check top confident errors CSV
    errors_file = output_dirs["metrics"] / "dummy_test_model_top_errors.csv"
    assert errors_file.exists()
    df_errors = pd.read_csv(errors_file)
    assert len(df_errors) == 1
    assert df_errors.iloc[0]["true_label"] == "fear"
    assert df_errors.iloc[0]["pred_label"] == "anger"
    assert df_errors.iloc[0]["confidence"] == pytest.approx(0.75)


def test_compute_metrics_merges_without_overwriting(tmp_path: Path) -> None:
    """Verify compute_metrics preserves existing keys in metrics JSON."""
    metrics_file = tmp_path / "merge_model.json"
    initial_metadata = {
        "inference_time_ms": 14.25,
        "device": "cpu",
        "num_parameters": 66362886,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(initial_metadata, f, indent=2)

    y_true = ["joy", "sadness", "anger"]
    y_pred = ["joy", "sadness", "sadness"]

    computed = compute_metrics(y_true, y_pred, model_name="merge_model", output_dir=tmp_path)

    with open(metrics_file, "r", encoding="utf-8") as f:
        updated_data = json.load(f)

    # Verify existing metadata was preserved
    assert updated_data["inference_time_ms"] == 14.25
    assert updated_data["device"] == "cpu"
    assert updated_data["num_parameters"] == 66362886

    # Verify new metrics were merged in
    assert updated_data["accuracy"] == computed["accuracy"]
    assert updated_data["macro_f1"] == computed["macro_f1"]
    assert updated_data["weighted_f1"] == computed["weighted_f1"]
