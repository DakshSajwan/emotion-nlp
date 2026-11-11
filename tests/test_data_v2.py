"""Unit tests for dataset v2 (data/train_v2.csv) construction and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATA_DIR, LABELS, label2id
from tests.conftest import requires_data


@requires_data("train_v2.csv")
def test_train_v2_exists_and_has_required_columns() -> None:
    """Verify that data/train_v2.csv exists, is non-empty, and has expected columns."""
    path = DATA_DIR / "train_v2.csv"
    assert path.exists(), f"File does not exist: {path}"
    assert path.stat().st_size > 0, f"File is empty: {path}"

    df = pd.read_csv(path)
    assert list(df.columns) == ["text", "label", "label_name"], (
        f"Columns mismatch. Expected ['text', 'label', 'label_name'], got {list(df.columns)}"
    )
    assert len(df) > 16000, f"Expected more than original 16,000 samples, got {len(df)}"


@requires_data("train_v2.csv")
@requires_data("val.csv")
@requires_data("test.csv")
def test_no_overlap_with_val_test_probe() -> None:
    """Strictly verify zero data leakage between train_v2 and val/test/probe datasets."""
    train_v2_df = pd.read_csv(DATA_DIR / "train_v2.csv")
    train_texts = set(train_v2_df["text"].dropna().str.strip().str.lower())

    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    probe_df = pd.read_csv(DATA_DIR / "probe_set.csv")

    forbidden_texts = (
        set(val_df["text"].dropna().str.strip().str.lower())
        | set(test_df["text"].dropna().str.strip().str.lower())
        | set(probe_df["text"].dropna().str.strip().str.lower())
    )

    overlaps = train_texts.intersection(forbidden_texts)
    assert len(overlaps) == 0, f"Found {len(overlaps)} leaking sentences in train_v2.csv: {overlaps}"


@requires_data("train_v2.csv")
def test_all_labels_valid() -> None:
    """Verify that all labels in train_v2 belong to the canonical 6 emotion classes."""
    df = pd.read_csv(DATA_DIR / "train_v2.csv")

    # Check label_name validity
    unique_names = set(df["label_name"].unique())
    assert unique_names.issubset(set(LABELS)), f"Unexpected label names found: {unique_names - set(LABELS)}"

    # Check label integer ID validity
    unique_ids = set(df["label"].unique())
    assert unique_ids.issubset(set(range(len(LABELS)))), f"Unexpected label IDs found: {unique_ids}"

    # Verify ID to Name alignment
    for _, row in df.head(500).iterrows():
        assert row["label"] == label2id[row["label_name"]], (
            f"ID/Name mismatch: {row['label']} vs {row['label_name']}"
        )


@requires_data("train_v2.csv")
def test_no_empty_texts() -> None:
    """Verify that no text in train_v2 is null, empty, or whitespace-only."""
    df = pd.read_csv(DATA_DIR / "train_v2.csv")

    assert df["text"].isna().sum() == 0, "Found null/NaN values in text column"

    trimmed_lengths = df["text"].astype(str).str.strip().str.len()
    assert (trimmed_lengths == 0).sum() == 0, "Found empty or whitespace-only text entries"

    word_counts = df["text"].astype(str).apply(lambda t: len(t.split()))
    assert (word_counts == 0).sum() == 0, "Found entries with zero words"
