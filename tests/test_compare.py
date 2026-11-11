"""Unit tests for src/compare.py using isolated temporary directory."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.config import LABELS
from src.compare import (
    check_required_files,
    generate_comparison_table,
    generate_error_analysis,
    plot_grouped_overall_metrics,
    plot_grouped_per_class_f1,
    run_comparison,
)


def test_check_required_files_detects_missing(tmp_path: Path) -> None:
    """Verify check_required_files returns missing files cleanly."""
    p_dir = tmp_path / "predictions"
    m_dir = tmp_path / "metrics"
    p_dir.mkdir()
    m_dir.mkdir()

    # Empty dirs -> missing
    ok, missing = check_required_files(p_dir, m_dir)
    assert not ok
    assert len(missing) == 4

    # Create dummy baseline files
    (p_dir / "baseline_test_preds.csv").write_text("dummy")
    (m_dir / "baseline.json").write_text("{}")
    ok, missing = check_required_files(p_dir, m_dir)
    assert not ok
    assert len(missing) == 2
    assert any("DistilBERT" in m for m in missing)


def test_compare_generates_all_artifacts(tmp_path: Path) -> None:
    """Verify run_comparison produces tables, plots, and error analysis."""
    p_dir = tmp_path / "predictions"
    m_dir = tmp_path / "metrics"
    f_dir = tmp_path / "figures"
    p_dir.mkdir()
    m_dir.mkdir()
    f_dir.mkdir()

    # Create dummy predictions for 12 samples covering various discrepancy types
    texts = [f"Sample text {i}" for i in range(12)]
    true_labels = [
        "sadness", "joy", "love", "anger", "fear", "surprise",
        "sadness", "joy", "love", "anger", "fear", "surprise",
    ]

    # Baseline predictions
    base_preds = [
        "anger", "joy", "joy", "anger", "fear", "surprise",   # 0 and 2 wrong
        "sadness", "sadness", "love", "anger", "fear", "fear", # 7 and 11 wrong
    ]
    # DistilBERT predictions
    dist_preds = [
        "sadness", "joy", "love", "anger", "fear", "surprise", # 0 and 2 right (base wrong, dist right)
        "anger", "joy", "love", "anger", "fear", "surprise",   # 6 wrong (dist wrong, base right)
    ]

    def _make_df(preds: list[str]) -> pd.DataFrame:
        data = {
            "text": texts,
            "true_label": true_labels,
            "pred_label": preds,
            "confidence": [0.85] * len(texts),
        }
        for l in LABELS:
            data[f"prob_{l}"] = [0.1] * len(texts)
        return pd.DataFrame(data)

    df_b = _make_df(base_preds)
    df_d = _make_df(dist_preds)

    df_b.to_csv(p_dir / "baseline_test_preds.csv", index=False)
    df_d.to_csv(p_dir / "distilbert_test_preds.csv", index=False)

    base_metrics = {
        "accuracy": 0.6667,
        "macro_f1": 0.6500,
        "weighted_f1": 0.6600,
        "inference_time_per_1000_samples_ms": 5.2,
        "model_size_mb": 1.2,
    }
    dist_metrics = {
        "accuracy": 0.9167,
        "macro_f1": 0.9000,
        "weighted_f1": 0.9100,
        "inference_time_per_1000_samples_ms": 2300.0,
        "model_size_mb": 256.0,
    }

    with open(m_dir / "baseline.json", "w") as f:
        json.dump(base_metrics, f)
    with open(m_dir / "distilbert.json", "w") as f:
        json.dump(dist_metrics, f)

    success = run_comparison(predictions_dir=p_dir, metrics_dir=m_dir, figures_dir=f_dir)
    assert success

    # Verify all 4 required outputs exist
    assert (m_dir / "comparison_table.csv").exists()
    assert (m_dir / "comparison_table.md").exists()
    assert (f_dir / "compare_per_class_f1.png").exists()
    assert (f_dir / "compare_overall.png").exists()
    assert (m_dir / "error_analysis.md").exists()

    # Verify error analysis content
    error_md = (m_dir / "error_analysis.md").read_text()
    assert "Baseline Wrong, DistilBERT Right" in error_md
    assert "DistilBERT Wrong, Baseline Right" in error_md
    assert "Both Models Wrong" in error_md
