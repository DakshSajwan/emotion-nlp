"""Pytest configuration and test skipping fixtures for models and datasets."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable
import pytest

from src.config import DATA_DIR, MODELS_DIR


def is_model_present(name: str) -> bool:
    """Check if model folder or file exists in models/ with weights present."""
    name_clean = name.lower().strip()
    if name_clean == "baseline":
        p = MODELS_DIR / "baseline.joblib"
        return p.is_file() and p.stat().st_size > 0
    elif name_clean in (
        "distilbert",
        "distilbert_v2",
        "roberta_v3",
        "distilbert_smoke",
        "distilbert_v2_smoke",
    ):
        p = MODELS_DIR / name_clean
        w_safe = p / "model.safetensors"
        w_bin = p / "pytorch_model.bin"
        return p.is_dir() and (
            (w_safe.is_file() and w_safe.stat().st_size > 0)
            or (w_bin.is_file() and w_bin.stat().st_size > 0)
        )
    elif name_clean in ("calibration", "calibration.json"):
        p = MODELS_DIR / "distilbert_v2" / "calibration.json"
        return p.is_file() and p.stat().st_size > 0
    elif name_clean in ("lime", "explain"):
        from src.config import ACTIVE_DISTILBERT
        return is_model_present(ACTIVE_DISTILBERT)
    else:
        p = MODELS_DIR / name_clean
        return p.exists()


def is_data_present(filename: str) -> bool:
    """Check if a data file exists in data/ and is non-empty."""
    p = DATA_DIR / filename
    return p.is_file() and p.stat().st_size > 0


def requires_model(name: str) -> Any:
    """Decorator or test-level helper that skips tests if model weights are missing.

    Usage:
        @requires_model("distilbert")
        def test_something(): ...

        def test_another():
            requires_model("distilbert_v2")
            ...
    """
    caller_frame = sys._getframe(1)
    caller_func = caller_frame.f_code.co_name
    if caller_func.startswith("test_") or "test" in caller_func:
        if not is_model_present(name):
            pytest.skip(f"Required model '{name}' is missing in models/ directory.")
        return None

    return pytest.mark.requires_model(name)


def requires_data(filename: str) -> Any:
    """Decorator or test-level helper that skips tests if data file is missing.

    Usage:
        @requires_data("train_v2.csv")
        def test_something(): ...

        def test_another():
            requires_data("val.csv")
            ...
    """
    caller_frame = sys._getframe(1)
    caller_func = caller_frame.f_code.co_name
    if caller_func.startswith("test_") or "test" in caller_func:
        if not is_data_present(filename):
            pytest.skip(f"Required data file '{filename}' is missing from data/.")
        return None

    return pytest.mark.requires_data(filename)


# Aliases for convenience
require_model = requires_model
require_data = requires_data


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers in pytest."""
    config.addinivalue_line(
        "markers", "requires_model(name): skip test if the specified model is missing"
    )
    config.addinivalue_line(
        "markers", "requires_data(filename): skip test if the specified data file is missing"
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Evaluate custom requires_model and requires_data markers prior to test execution."""
    for mark in item.iter_markers(name="requires_model"):
        model_name = mark.args[0] if mark.args else "distilbert"
        if not is_model_present(model_name):
            pytest.skip(f"Required model '{model_name}' is missing from models/ directory.")

    for mark in item.iter_markers(name="requires_data"):
        filename = mark.args[0] if mark.args else ""
        if filename and not is_data_present(filename):
            pytest.skip(f"Required data file '{filename}' is missing from data/ directory.")

