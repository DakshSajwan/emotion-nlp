"""Data loading and splitting module for Emotion Detection in Text."""

from typing import Any
from datasets import DatasetDict, load_dataset
from src.config import id2label


def load_emotion_dataset() -> DatasetDict:
    """Load the Hugging Face dair-ai/emotion dataset.

    Returns:
        DatasetDict containing 'train', 'validation', and 'test' splits.
    """
    return load_dataset("dair-ai/emotion")


def get_splits() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Retrieve train, validation, and test splits with raw texts and label names.

    Returns:
        Tuple of (train_data, val_data, test_data), each containing:
            - 'text': list of raw text strings
            - 'label': list of label names (e.g., 'sadness', 'joy')
            - 'label_id': list of integer class IDs (0..5)
    """
    dataset = load_emotion_dataset()
    splits: dict[str, dict[str, Any]] = {}

    for split in ["train", "validation", "test"]:
        texts: list[str] = list(dataset[split]["text"])
        label_ids: list[int] = list(dataset[split]["label"])
        label_names: list[str] = [id2label[idx] for idx in label_ids]
        splits[split] = {
            "text": texts,
            "label": label_names,
            "label_id": label_ids,
        }

    return splits["train"], splits["validation"], splits["test"]


def save_splits_to_csv() -> None:
    """Save canonical train, val, and test splits to data/*.csv if missing."""
    import pandas as pd
    from src.config import DATA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_emotion_dataset()
    for split_name, file_name in [
        ("train", "train.csv"),
        ("validation", "val.csv"),
        ("test", "test.csv"),
    ]:
        target_csv = DATA_DIR / file_name
        if not target_csv.exists():
            texts = list(dataset[split_name]["text"])
            label_ids = list(dataset[split_name]["label"])
            label_names = [id2label[idx] for idx in label_ids]
            df = pd.DataFrame({"text": texts, "label": label_names})
            df.to_csv(target_csv, index=False)
            print(f"Saved {len(df):,} samples to {file_name}")


if __name__ == "__main__":
    print("Loading dair-ai/emotion dataset splits...")
    save_splits_to_csv()
    train_data, val_data, test_data = get_splits()
    print(f"Train samples:      {len(train_data['text']):,}")
    print(f"Validation samples: {len(val_data['text']):,}")
    print(f"Test samples:       {len(test_data['text']):,}")
    print("Data loading completed successfully.")
