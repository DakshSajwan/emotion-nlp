"""DistilBERT transformer fine-tuning and evaluation module.

Uses HuggingFace Trainer to fine-tune distilbert-base-uncased for multiclass emotion detection.
Provides automatic evaluation strategy and tokenizer parameter adaptation across transformers versions.
"""

from typing import Any, Optional
import argparse
import inspect
from pathlib import Path
import json
import shutil
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    FIGURES_DIR,
    LABELS,
    MAX_LENGTH,
    METRICS_DIR,
    MODEL_NAME,
    MODELS_DIR,
    NUM_LABELS,
    PROJECT_ROOT,
    SEED,
    id2label,
    label2id,
    set_seed,
)
from src.data import get_splits
from src.evaluate import evaluate_model
from src.preprocess import minimal_clean


class WeightedTrainer(Trainer):
    """Custom HuggingFace Trainer supporting inverse-frequency class-weighted loss."""

    def __init__(
        self,
        *args: Any,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute class-weighted cross-entropy loss with optional label smoothing."""
        if self.class_weights is None or "labels" not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        labels = inputs.get("labels")
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs_no_labels)
        logits = outputs.get("logits")
        weights = self.class_weights.to(model.device)
        label_smoothing = getattr(self.args, "label_smoothing_factor", 0.0)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing,
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_train_data(filepath: Path | str) -> tuple[list[str], list[int]]:
    """Load text and integer label IDs from a training CSV file."""
    df = pd.read_csv(filepath)
    texts = df["text"].astype(str).tolist()
    if "label_id" in df.columns:
        labels = df["label_id"].astype(int).tolist()
    elif "label_name" in df.columns:
        labels = [label2id[str(x)] for x in df["label_name"]]
    elif "label" in df.columns:
        first_val = df["label"].iloc[0]
        if isinstance(first_val, (int, np.integer)) or (isinstance(first_val, str) and first_val.isdigit()):
            labels = df["label"].astype(int).tolist()
        else:
            labels = [label2id[str(x)] for x in df["label"]]
    else:
        raise ValueError(f"Could not find label column in {filepath}")
    return texts, labels


def load_val_data(filepath: Path | str) -> tuple[list[str], list[int]]:
    """Load validation split from CSV."""
    df = pd.read_csv(filepath)
    texts = df["text"].astype(str).tolist()
    if "label_id" in df.columns:
        labels = df["label_id"].astype(int).tolist()
    elif "label" in df.columns:
        first_val = df["label"].iloc[0]
        if isinstance(first_val, (int, np.integer)) or (isinstance(first_val, str) and first_val.isdigit()):
            labels = df["label"].astype(int).tolist()
        else:
            labels = [label2id[str(x)] for x in df["label"]]
    else:
        raise ValueError(f"Could not find label column in {filepath}")
    return texts, labels


def load_test_data(filepath: Path | str) -> tuple[list[str], list[str], list[int]]:
    """Load test split from CSV, returning (texts, label_names, label_ids)."""
    df = pd.read_csv(filepath)
    texts = df["text"].astype(str).tolist()
    if "label_name" in df.columns:
        label_names = [str(x) for x in df["label_name"]]
        label_ids = [label2id[name] for name in label_names]
    elif "label" in df.columns:
        first_val = df["label"].iloc[0]
        if isinstance(first_val, (int, np.integer)) or (isinstance(first_val, str) and first_val.isdigit()):
            label_ids = df["label"].astype(int).tolist()
            label_names = [id2label[i] for i in label_ids]
        else:
            label_names = [str(x) for x in df["label"]]
            label_ids = [label2id[name] for name in label_names]
    else:
        raise ValueError(f"Could not find label column in {filepath}")
    return texts, label_names, label_ids


class EmotionDataset(Dataset):
    """PyTorch Dataset for tokenized emotion texts."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_eval_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute accuracy, macro-F1, and weighted-F1 for evaluation during training."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def get_dir_size_mb(directory: Path) -> float:
    """Calculate total size of all files in directory in megabytes."""
    total_bytes = sum(f.stat().st_size for f in directory.glob("**/*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def plot_training_curves(
    log_history: list[dict[str, Any]],
    output_path: Path,
    model_name: str,
) -> None:
    """Plot training/validation loss and evaluation metric curves across epochs."""
    epochs_train: list[float] = []
    train_loss: list[float] = []
    epochs_eval: list[float] = []
    eval_loss: list[float] = []
    eval_acc: list[float] = []
    eval_f1: list[float] = []

    for entry in log_history:
        ep = entry.get("epoch")
        if ep is None:
            continue
        if "loss" in entry:
            epochs_train.append(ep)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            epochs_eval.append(ep)
            eval_loss.append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                eval_acc.append(entry["eval_accuracy"])
            if "eval_macro_f1" in entry:
                eval_f1.append(entry["eval_macro_f1"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    if epochs_train and train_loss:
        axes[0].plot(epochs_train, train_loss, label="Train Loss", marker="o", color="#1f77b4")
    if epochs_eval and eval_loss:
        axes[0].plot(epochs_eval, eval_loss, label="Validation Loss", marker="s", color="#d62728")
    axes[0].set_title(f"Loss Curves ({model_name})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Metrics subplot
    if epochs_eval and eval_acc:
        axes[1].plot(epochs_eval, eval_acc, label="Val Accuracy", marker="o", color="#2ca02c")
    if epochs_eval and eval_f1:
        axes[1].plot(epochs_eval, eval_f1, label="Val Macro-F1", marker="s", color="#9467bd")
    axes[1].set_title(f"Validation Performance ({model_name})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def train_transformer(
    subset: Optional[int] = None,
    epochs: int = 3,
    batch_size: int = 32,
    train_file: str | Path = DATA_DIR / "train.csv",
    run_name: str = "distilbert",
    label_smoothing: float = 0.0,
    class_weights: bool = False,
) -> dict[str, Any]:
    """Train and evaluate DistilBERT on emotion dataset.

    Args:
        subset: If set, trains on N train samples and N//4 validation samples.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        train_file: Path to training CSV dataset (default: data/train.csv).
        run_name: Base run name for model and artifact outputs (default: 'distilbert').
        label_smoothing: Epsilon label smoothing factor for CrossEntropyLoss (default: 0.0).
        class_weights: Whether to apply inverse-frequency class weights in loss (default: False).

    Returns:
        Dictionary of final test metrics and benchmark measurements.
    """
    set_seed(SEED)

    # Use '<run_name>_smoke' when subset is active to avoid collision with full runs
    model_name = f"{run_name}_smoke" if subset is not None else run_name
    print(f"\n==========================================")
    print(f"Starting Fine-Tuning: {model_name.upper()}")
    print(f"Run Name: {run_name} | Model Target: {model_name}")
    print(f"Train File: {train_file}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Subset: {subset}")
    print(f"Label Smoothing: {label_smoothing} | Class Weights: {class_weights}")
    print(f"==========================================\n")

    # 1. Load Data
    train_path = Path(train_file)
    if not train_path.is_absolute():
        if not train_path.exists() and (DATA_DIR / train_file).exists():
            train_path = DATA_DIR / train_file
        elif not train_path.exists() and (PROJECT_ROOT / train_file).exists():
            train_path = PROJECT_ROOT / train_file

    print(f"Loading training data from: {train_path}")
    train_texts_full, train_labels_full = load_train_data(train_path)

    # Validation is ALWAYS loaded from original data/val.csv for comparability
    val_path = DATA_DIR / "val.csv"
    val_texts_full, val_labels_full = load_val_data(val_path)

    # Test set is ALWAYS evaluated in full from original data/test.csv
    test_path = DATA_DIR / "test.csv"
    test_texts_raw, test_labels_names, test_labels_ids = load_test_data(test_path)

    if subset is not None:
        train_texts_raw = train_texts_full[:subset]
        train_labels = train_labels_full[:subset]
        val_subset = max(1, subset // 4)
        val_texts_raw = val_texts_full[:val_subset]
        val_labels = val_labels_full[:val_subset]
    else:
        train_texts_raw = train_texts_full
        train_labels = train_labels_full
        val_texts_raw = val_texts_full
        val_labels = val_labels_full

    # Compute inverse class frequency weights if flag is enabled
    if class_weights:
        print(f"Computing inverse-frequency class weights from {train_path}...")
        classes = np.arange(NUM_LABELS)
        weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=np.array(train_labels_full),
        )
        computed_class_weights: Optional[torch.Tensor] = torch.tensor(weights_arr, dtype=torch.float)
        weights_summary = {LABELS[i]: round(float(weights_arr[i]), 4) for i in range(NUM_LABELS)}
        print(f"Class weights computed: {weights_summary}")
    else:
        computed_class_weights = None

    # 2. Preprocess with minimal_clean
    print("Applying minimal_clean preprocessing...")
    train_texts = [minimal_clean(t) for t in train_texts_raw]
    val_texts = [minimal_clean(t) for t in val_texts_raw]
    test_texts = [minimal_clean(t) for t in test_texts_raw]

    # 3. Tokenize
    print(f"Tokenizing with {MODEL_NAME} (max_length={MAX_LENGTH})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, max_length=MAX_LENGTH)

    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels_ids)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Model instantiation
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    # 5. TrainingArguments version compatibility check
    tmp_ckpt_dir = MODELS_DIR / f"{model_name}_checkpoints"
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(tmp_ckpt_dir),
        "learning_rate": 5e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "fp16": False,  # CPU compatibility
        "report_to": "none",
        "seed": SEED,
        "save_strategy": "epoch",
        eval_strategy_key: "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_strategy": "steps",
        "logging_steps": max(1, len(train_dataset) // (batch_size * 2)),
        "label_smoothing_factor": label_smoothing,
    }

    training_args = TrainingArguments(**training_args_kwargs)

    # 6. Trainer instantiation version compatibility check
    tr_params = inspect.signature(Trainer.__init__).parameters
    tokenizer_arg_name = "processing_class" if "processing_class" in tr_params else "tokenizer"

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_eval_metrics,
        "class_weights": computed_class_weights,
        tokenizer_arg_name: tokenizer,
    }

    trainer = WeightedTrainer(**trainer_kwargs)

    # 7. Train
    print("Starting training...")
    start_train = time.perf_counter()
    train_result = trainer.train()
    training_time_s = time.perf_counter() - start_train
    print(f"Training completed in {training_time_s:.2f}s ({training_time_s/60:.2f} min).")

    # 8. Save final model and tokenizer (clean folder, no nested checkpoints)
    final_model_dir = MODELS_DIR / model_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"Final model and tokenizer saved to: {final_model_dir}")

    # Remove temporary checkpoints folder
    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)

    # Calculate model disk footprint
    model_size_mb = get_dir_size_mb(final_model_dir)

    # 9. Save training curves and train log
    log_history = trainer.state.log_history
    log_file = METRICS_DIR / f"{model_name}_train_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    curves_file = FIGURES_DIR / f"{model_name}_training_curves.png"
    plot_training_curves(log_history, curves_file, model_name)
    print(f"Training curves saved to: {curves_file}")

    # 10. Predict on full test set and benchmark inference
    print("Evaluating on full test set...")
    start_infer = time.perf_counter()
    test_predictions = trainer.predict(test_dataset)
    infer_time_total_s = time.perf_counter() - start_infer

    logits = test_predictions.predictions
    # Apply softmax to obtain probability distributions in config.LABELS order
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    n_test = len(test_dataset)
    infer_time_per_1000_s = (infer_time_total_s / n_test) * 1000
    infer_time_per_1000_ms = infer_time_per_1000_s * 1000

    # 11. Run full evaluation suite
    metrics = evaluate_model(model_name, test_texts_raw, test_labels_names, probs)

    # 12. Merge benchmark metrics into reports/metrics/<model_name>.json
    metrics_file = METRICS_DIR / f"{model_name}.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    benchmark_meta = {
        "training_time_seconds": round(training_time_s, 4),
        "inference_time_per_1000_samples_ms": round(infer_time_per_1000_ms, 4),
        "inference_time_per_1000_samples_seconds": round(infer_time_per_1000_s, 6),
        "total_test_inference_time_seconds": round(infer_time_total_s, 4),
        "model_size_mb": model_size_mb,
        "epochs": epochs,
        "batch_size": batch_size,
        "subset": subset,
        "train_file": str(train_path),
        "run_name": run_name,
        "label_smoothing": label_smoothing,
        "class_weights": class_weights,
    }
    saved_metrics.update(benchmark_meta)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(saved_metrics, f, indent=2)

    print(f"\nFinal {model_name} Benchmark Summary:")
    print(json.dumps(saved_metrics, indent=2))

    return saved_metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for transformer training."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on emotion dataset.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train.csv",
        help="Path to training CSV file (default: data/train.csv; use data/train_v2.csv for v2).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="distilbert",
        help="Run name identifier for models, predictions, and reports (default: 'distilbert'; use 'distilbert_v2' for v2).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for TrainingArguments (default: 0.0; for v2 use 0.1).",
    )
    parser.add_argument(
        "--class_weights",
        action="store_true",
        default=False,
        help="If set, compute inverse-frequency class weights from training file and use custom loss.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Number of train samples (val will be subset // 4). If provided, outputs to <run_name>_smoke.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-device batch size (default: 32).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_transformer(
        subset=args.subset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_file=args.train_file,
        run_name=args.run_name,
        label_smoothing=args.label_smoothing,
        class_weights=args.class_weights,
    )
