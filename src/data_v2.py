"""Dataset V2 builder module for Emotion Detection in Text.

Constructs an augmented, multi-source training dataset (data/train_v2.csv) combining:
1. The original dair-ai/emotion training split.
2. Filtered, single-label mapped examples from Google's GoEmotions benchmark.
3. Light prefix-stripping and rephrasing augmentation on 'i feel' expressions.
Strictly removes duplicates and enforces a zero-leakage filter against val, test, and probe sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any, Tuple

from datasets import load_dataset
import pandas as pd

from src.config import DATA_DIR, GOEMOTIONS_MAP, LABELS, SEED, label2id

MAX_EXTERNAL_PER_CLASS: int = 4000
PREFIXES: tuple[str, ...] = ("i feel like", "i am feeling", "i m feeling", "i feel")


def rephrase_prefix(text: str, rng: random.Random) -> str:
    """Remove or naturally rephrase leading emotional state prefixes.

    Args:
        text: Input text string.
        rng: Deterministic random generator.

    Returns:
        Rephrased text string with leading 'i feel' removed or contextualized.
    """
    clean_text = text.strip()
    lower_text = clean_text.lower()
    remainder = clean_text

    for prefix in PREFIXES:
        if lower_text == prefix or lower_text.startswith(prefix + " "):
            remainder = clean_text[len(prefix) :].lstrip(" ,;:-")
            break

    # Natural rephrasing variations
    options = [
        remainder,
        remainder,  # weighted 40% for pure removal
        f"honestly, {remainder}",
        f"really, {remainder}",
        f"truly, {remainder}",
    ]
    return rng.choice(options)


def build_train_v2(
    max_external_per_class: int = MAX_EXTERNAL_PER_CLASS,
    seed: int = SEED,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build, clean, augment, and export data/train_v2.csv.

    Args:
        max_external_per_class: Maximum GoEmotions samples to add per emotion class.
        seed: Random seed for reproducible sampling and shuffling.
        output_path: Optional path for destination CSV. Defaults to DATA_DIR / 'train_v2.csv'.

    Returns:
        Final prepared pandas DataFrame of train_v2.
    """
    rng = random.Random(seed)
    destination = Path(output_path) if output_path else DATA_DIR / "train_v2.csv"

    print("=" * 70)
    print("  BUILDING DATASET V2 (AUGMENTED + GOEMOTIONS)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 0: Assemble zero-leakage exclusion set
    # -----------------------------------------------------------------------
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"
    probe_path = DATA_DIR / "probe_set.csv"

    leakage_set: set[str] = set()
    for p in [val_path, test_path, probe_path]:
        if p.exists():
            df_leak = pd.read_csv(p)
            leakage_set.update(df_leak["text"].dropna().str.strip().str.lower().tolist())

    print(f"Loaded {len(leakage_set):,} forbidden sentences for zero-leakage protection.")

    # -----------------------------------------------------------------------
    # Step 1: Ingest original dair-ai training data
    # -----------------------------------------------------------------------
    train_orig_path = DATA_DIR / "train.csv"
    if not train_orig_path.exists():
        raise FileNotFoundError(f"Original train.csv not found at: {train_orig_path}")

    df_orig = pd.read_csv(train_orig_path)
    before_class_counts = df_orig["label"].value_counts().to_dict()

    rows: list[dict[str, Any]] = []
    for _, row in df_orig.iterrows():
        lbl_name = str(row["label"]).strip()
        if lbl_name in label2id:
            rows.append({
                "text": str(row["text"]).strip(),
                "label": label2id[lbl_name],
                "label_name": lbl_name,
                "source": "dair_original",
            })

    print(f"Loaded {len(rows):,} original DAIR training sentences.")

    # -----------------------------------------------------------------------
    # Step 2: Ingest and map external GoEmotions examples
    # -----------------------------------------------------------------------
    print(f"\nIngesting GoEmotions dataset (cap = {max_external_per_class:,} per class)...")
    try:
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception:
        ds = load_dataset("go_emotions", "simplified")

    feature_names = ds["train"].features["labels"].feature.names
    external_per_class: dict[str, list[dict[str, Any]]] = {c: [] for c in LABELS}

    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            labels = ex.get("labels", [])
            # Keep only rows with exactly ONE label
            if len(labels) == 1:
                label_idx = labels[0]
                go_label_name = feature_names[label_idx]

                if go_label_name in GOEMOTIONS_MAP:
                    target_class = GOEMOTIONS_MAP[go_label_name]
                    if len(external_per_class[target_class]) < max_external_per_class:
                        raw_text = str(ex["text"]).strip()
                        word_count = len(raw_text.split())
                        # Length constraint: 4 to 40 words
                        if 4 <= word_count <= 40:
                            external_per_class[target_class].append({
                                "text": raw_text,
                                "label": label2id[target_class],
                                "label_name": target_class,
                                "source": "go_emotions",
                            })

    total_go_added = sum(len(v) for v in external_per_class.values())
    for target_class in LABELS:
        rows.extend(external_per_class[target_class])
    print(f"Added {total_go_added:,} single-label mapped GoEmotions sentences.")

    # -----------------------------------------------------------------------
    # Step 3: Light augmentation of 15% prefix-matching DAIR rows
    # -----------------------------------------------------------------------
    matching_orig_indices: list[int] = []
    for idx, r in enumerate(df_orig["text"]):
        t_low = str(r).lower().strip()
        if any(t_low == p or t_low.startswith(p + " ") for p in PREFIXES):
            matching_orig_indices.append(idx)

    sample_size = int(round(len(matching_orig_indices) * 0.15))
    selected_indices = rng.sample(matching_orig_indices, sample_size)

    augmented_rows: list[dict[str, Any]] = []
    augmented_preview: list[Tuple[str, str]] = []

    for idx in selected_indices:
        orig_row = df_orig.iloc[idx]
        orig_text = str(orig_row["text"]).strip()
        lbl_name = str(orig_row["label"]).strip()

        rephrased_text = rephrase_prefix(orig_text, rng)
        augmented_rows.append({
            "text": rephrased_text,
            "label": label2id[lbl_name],
            "label_name": lbl_name,
            "source": "dair_augmented",
        })
        if len(augmented_preview) < 10:
            augmented_preview.append((orig_text, rephrased_text))

    rows.extend(augmented_rows)
    print(f"Generated {len(augmented_rows):,} augmented prefix-stripped copies (15% of matching DAIR rows).")

    # -----------------------------------------------------------------------
    # Step 4: Leakage filtering and deduplication
    # -----------------------------------------------------------------------
    df_combined = pd.DataFrame(rows)
    initial_count = len(df_combined)

    # Check leakage against val, test, and probe
    is_leak = df_combined["text"].str.strip().str.lower().isin(leakage_set)
    leakage_removed = int(is_leak.sum())
    df_clean = df_combined[~is_leak].copy()

    # Check and drop exact duplicates
    dup_count = int(df_clean.duplicated(subset=["text"]).sum())
    df_clean = df_clean.drop_duplicates(subset=["text"], keep="first").copy()

    # Drop any blank/empty strings if any
    df_clean = df_clean[df_clean["text"].str.strip().str.len() > 0].copy()

    # -----------------------------------------------------------------------
    # Step 5: Shuffle and Export
    # -----------------------------------------------------------------------
    df_clean = df_clean.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save target dataset with exact columns: text, label, label_name
    output_df = df_clean[["text", "label", "label_name"]].copy()
    output_df.to_csv(destination, index=False)

    after_class_counts = df_clean["label_name"].value_counts().to_dict()

    # -----------------------------------------------------------------------
    # Summary Reporting
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  SUMMARY STATISTICS FOR DATASET V2")
    print("-" * 70)
    print(f"Total rows before cleaning : {initial_count:,}")
    print(f"Leakage rows removed       : {leakage_removed:,} (guaranteeing zero overlap with val/test/probe)")
    print(f"Duplicate rows dropped     : {dup_count:,}")
    print(f"Final training rows        : {len(df_clean):,}")

    print("\nRows per Data Source:")
    source_counts = df_clean["source"].value_counts()
    for src, cnt in source_counts.items():
        print(f"  - {src:<16}: {cnt:>6,} ({cnt / len(df_clean) * 100:.1f}%)")

    print("\nClass Distribution (Before vs. After):")
    print(f"  {'Class':<12} {'Original DAIR':<15} {'Train V2':<15} {'Growth'}")
    print(f"  {'-'*10:<12} {'-'*13:<15} {'-'*13:<15} {'-'*10}")
    for c in LABELS:
        before = before_class_counts.get(c, 0)
        after = after_class_counts.get(c, 0)
        growth = f"+{after - before:,}"
        print(f"  {c:<12} {before:>8,}        {after:>8,}        {growth:>8}")

    print("\n10 Sample Augmented Rephrasings (Prefix Breakdown):")
    for idx, (orig, aug) in enumerate(augmented_preview, 1):
        print(f"  [{idx:2d}] Original : \"{orig}\"")
        print(f"       Augmented: \"{aug}\"\n")

    print(f"Saved dataset to: {destination}")
    print("=" * 70 + "\n")

    return output_df


def main() -> None:
    """CLI entry point for dataset v2 generation."""
    parser = argparse.ArgumentParser(
        description="Build augmented, multi-source training dataset (train_v2.csv).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-external",
        type=int,
        default=MAX_EXTERNAL_PER_CLASS,
        help="Maximum GoEmotions external sentences to import per class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible augmentation and shuffling",
    )
    args = parser.parse_args()

    build_train_v2(max_external_per_class=args.max_external, seed=args.seed)


if __name__ == "__main__":
    main()
