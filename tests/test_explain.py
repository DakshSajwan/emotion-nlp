"""Unit tests for LIME explainability module in src/explain.py."""

from pathlib import Path
import pytest
from src.config import LABELS
from src.explain import (
    explain_text,
    plot_explanation_barchart,
    predict_proba,
    save_explanation,
)
from tests.conftest import requires_model


def test_empty_or_whitespace_input_returns_error() -> None:
    """Verify empty or whitespace strings return error value without crashing."""
    res_none = explain_text(None)
    assert res_none["error"] is not None
    assert res_none["predicted_label"] == ""
    assert res_none["weights"] == []

    res_empty = explain_text("")
    assert res_empty["error"] is not None
    assert res_empty["weights"] == []

    res_ws = explain_text("    \n\t   ")
    assert res_ws["error"] is not None
    assert res_ws["weights"] == []


@requires_model("lime")
def test_predict_proba_handles_empty_strings() -> None:
    """Verify batched predict_proba handles empty strings perturbed by LIME."""
    batch = ["I feel happy", "", "   ", "this is wonderful"]
    probs = predict_proba(batch, batch_size=2)
    assert probs.shape == (4, len(LABELS))
    # Probabilities should sum to approximately 1.0 per row
    for row in probs:
        assert pytest.approx(float(row.sum()), abs=1e-4) == 1.0


@requires_model("lime")
def test_explain_text_and_saving(tmp_path: Path) -> None:
    """Verify explain_text generates valid weights and saves HTML and PNG files."""
    text = "i am so grateful and happy for your kind support"
    res = explain_text(text, num_samples=50, num_features=4)

    assert res["error"] is None
    assert res["predicted_label"] in LABELS
    assert len(res["probabilities"]) == len(LABELS)
    assert len(res["weights"]) <= 4

    # Test save_explanation HTML
    html_file = tmp_path / "test_lime.html"
    save_explanation(res["explanation"], html_file)
    assert html_file.exists() and html_file.stat().st_size > 0

    # Test plot_explanation_barchart PNG
    png_file = tmp_path / "test_lime.png"
    plot_explanation_barchart(res["weights"], res["predicted_label"], 0.85, png_file)
    assert png_file.exists() and png_file.stat().st_size > 0
