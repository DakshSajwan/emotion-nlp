"""Tests for preset sentence pool management and sampling queue logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import LABELS
from src.presets import FALLBACK_POOLS, build_pools, next_sample


def test_no_immediate_repetition() -> None:
    """Verify that 200 consecutive next_sample calls never return the same sentence twice in a row."""
    state: dict[str, Any] = {}
    prev: str | None = None
    for i in range(200):
        curr = next_sample("joy", state)
        assert curr != prev, f"Immediate repetition detected at step {i}: {curr}"
        prev = curr


def test_unique_draws_until_exhaustion() -> None:
    """Verify that the first min(pool_size, 50) draws are all unique without duplicates."""
    pools = build_pools()
    for label in LABELS:
        state: dict[str, Any] = {}
        pool_size = len(pools[label])
        num_draws = min(pool_size, 50)
        draws = [next_sample(label, state, pools=pools) for _ in range(num_draws)]
        assert len(set(draws)) == num_draws, (
            f"Expected {num_draws} unique draws for label '{label}', got {len(set(draws))}"
        )


def test_every_label_returns_non_empty_string() -> None:
    """Verify that every emotion label returns a valid, non-empty text string."""
    state: dict[str, Any] = {}
    for label in LABELS:
        sample = next_sample(label, state)
        assert isinstance(sample, str)
        assert len(sample.strip()) > 0
        words = len(sample.split())
        assert 6 <= words <= 40, f"Sample word count ({words}) outside 6-40 range: '{sample}'"


def test_fallback_when_csv_missing(tmp_path: Path) -> None:
    """Verify fallback mechanism when prediction CSV is absent."""
    missing_csv = tmp_path / "does_not_exist.csv"
    pools = build_pools(missing_csv)

    # Check that all canonical labels exist with at least 15 fallback sentences
    assert len(pools) == len(LABELS)
    state: dict[str, Any] = {}

    for label in LABELS:
        assert len(pools[label]) >= 15
        assert set(pools[label]) == set(FALLBACK_POOLS[label])

        sample = next_sample(label, state, pools=pools)
        assert isinstance(sample, str)
        assert sample in FALLBACK_POOLS[label]
        assert len(sample.strip()) > 0

    # Verify that consecutive draws from fallback pool (size 15) also never repeat consecutively
    prev: str | None = None
    fb_state: dict[str, Any] = {}
    for _ in range(100):
        curr = next_sample("surprise", fb_state, pools=pools)
        assert curr != prev, f"Immediate repetition in fallback pool: {curr}"
        prev = curr
