"""Tests for independent faculty probe evaluation dataset and pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATA_DIR, LABELS
from src.probe_eval import evaluate_probe
from tests.conftest import requires_data


def test_probe_set_file_exists() -> None:
    """Verify that data/probe_set.csv exists, is non-empty, and has expected columns."""
    probe_path = DATA_DIR / "probe_set.csv"
    assert probe_path.exists(), f"Probe dataset file does not exist at {probe_path}"
    assert probe_path.stat().st_size > 0, "Probe dataset file is empty"

    df = pd.read_csv(probe_path)
    assert set(["text", "label_name", "split"]).issubset(df.columns), (
        f"Missing required columns in probe_set.csv. Found: {df.columns.tolist()}"
    )


def test_six_labels_have_twenty_rows_each() -> None:
    """Verify that all 6 canonical labels have exactly 20 rows and neutral has 24."""
    df = pd.read_csv(DATA_DIR / "probe_set.csv")

    for label in LABELS:
        count = (df["label_name"] == label).sum()
        assert count == 20, f"Expected exactly 20 rows for label '{label}', got {count}"

    neutral_count = (df["label_name"] == "neutral").sum()
    assert neutral_count == 24, f"Expected 24 neutral rows, got {neutral_count}"
    assert len(df) == 144, f"Expected total 144 rows, got {len(df)}"


@requires_data("train.csv")
@requires_data("val.csv")
@requires_data("test.csv")
def test_no_overlap_with_train_val_test_csvs() -> None:
    """Verify that no probe sentence overlaps with train, val, or test splits."""
    probe_df = pd.read_csv(DATA_DIR / "probe_set.csv")
    probe_texts = set(probe_df["text"].str.strip().str.lower())

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    existing_texts = (
        set(train_df["text"].str.strip().str.lower())
        | set(val_df["text"].str.strip().str.lower())
        | set(test_df["text"].str.strip().str.lower())
    )

    overlaps = probe_texts.intersection(existing_texts)
    assert len(overlaps) == 0, f"Found overlapping sentences between probe set and corpus: {overlaps}"


def test_splits_are_balanced() -> None:
    """Verify that dev and test splits are strictly balanced across all classes."""
    df = pd.read_csv(DATA_DIR / "probe_set.csv")

    for label in LABELS:
        dev_cnt = ((df["label_name"] == label) & (df["split"] == "dev")).sum()
        test_cnt = ((df["label_name"] == label) & (df["split"] == "test")).sum()
        assert dev_cnt == 10, f"Expected 10 dev rows for '{label}', got {dev_cnt}"
        assert test_cnt == 10, f"Expected 10 test rows for '{label}', got {test_cnt}"

    dev_neutral = ((df["label_name"] == "neutral") & (df["split"] == "dev")).sum()
    test_neutral = ((df["label_name"] == "neutral") & (df["split"] == "test")).sum()
    assert dev_neutral == 12, f"Expected 12 dev neutral rows, got {dev_neutral}"
    assert test_neutral == 12, f"Expected 12 test neutral rows, got {test_neutral}"

    assert (df["split"] == "dev").sum() == 72
    assert (df["split"] == "test").sum() == 72


def test_probe_sentence_word_counts() -> None:
    """Verify that all probe sentences adhere to 5-25 words constraint."""
    df = pd.read_csv(DATA_DIR / "probe_set.csv")
    word_counts = df["text"].apply(lambda t: len(t.split()))
    assert word_counts.min() >= 5, f"Found sentence with < 5 words: {word_counts.min()}"
    assert word_counts.max() <= 25, f"Found sentence with > 25 words: {word_counts.max()}"
