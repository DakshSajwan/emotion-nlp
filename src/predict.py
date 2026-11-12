"""Inference API and prediction utilities for Emotion Detection in Text.

Provides the EmotionPredictor class supporting both classical baseline (TF-IDF + LR)
and fine-tuned DistilBERT models with lazy loading, batched inference, and CLI access.
"""

from typing import Any, Optional
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    ACTIVE_DISTILBERT,
    LABELS,
    MAX_LENGTH,
    MODEL_DIRS,
    available_models,
)
from src.preprocess import heavy_clean, minimal_clean


class EmotionPredictor:
    """Inference predictor for emotion classification in text.

    Supports 'baseline' (TF-IDF + Logistic Regression), 'distilbert', and 'distilbert_v2' models.
    Models are loaded lazily on the first prediction call.
    """

    def __init__(self, model_type: str = "baseline") -> None:
        """Initialize EmotionPredictor.

        Args:
            model_type: Classifier architecture ('baseline', 'distilbert', or 'distilbert_v2').

        Raises:
            ValueError: If model_type is not 'baseline', 'distilbert', or 'distilbert_v2'.
        """
        model_type_clean = model_type.lower().strip()
        if model_type_clean not in MODEL_DIRS:
            raise ValueError(
                f"Invalid model_type: '{model_type}'. Expected one of {list(MODEL_DIRS.keys())}."
            )

        self.model_type: str = model_type_clean
        self._model: Optional[Any] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._temperature: Optional[float] = None
        self._abstain_threshold: Optional[float] = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the requested model, tokenizer, and calibration parameters on first invocation.

        Raises:
            FileNotFoundError: If the corresponding model artifact cannot be found.
        """
        if self._model is not None:
            return

        if self.model_type == "baseline":
            model_path = MODEL_DIRS["baseline"]
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Baseline model not found at {model_path}.\n"
                    "Please download the pre-trained models by running:\n"
                    "    python -m src.download_models\n"
                    "or train the baseline model from scratch with:\n"
                    "    python -m src.data\n"
                    "    python -m src.baseline"
                )
            try:
                self._model = joblib.load(model_path)
            except Exception as err:
                raise RuntimeError(
                    f"Failed to load baseline model from {model_path}: {err}\n"
                    "This may be caused by a scikit-learn version mismatch.\n"
                    "Please retrain the baseline model by running:\n"
                    "    python -m src.data\n"
                    "    python -m src.baseline"
                ) from err

        elif self.model_type in ("distilbert", "distilbert_v2"):
            target_dir = MODEL_DIRS[self.model_type]

            if not target_dir.exists():
                raise FileNotFoundError(
                    f"{self.model_type} model not found in {target_dir}.\n"
                    "Please download the pre-trained models by running:\n"
                    "    python -m src.download_models\n"
                    "or train from scratch with:\n"
                    f"    python -m src.transformer --run_name {self.model_type}"
                )

            self._tokenizer = AutoTokenizer.from_pretrained(str(target_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(target_dir))
            model.eval()
            self._model = model

            # Load optional calibration file (temperature scaling & abstain threshold)
            cal_file = target_dir / "calibration.json"
            if cal_file.exists():
                try:
                    with open(cal_file, "r", encoding="utf-8") as f:
                        cal_data = json.load(f)
                    t_val = cal_data.get("temperature", cal_data.get("T"))
                    if t_val is not None:
                        self._temperature = float(t_val)
                    thresh_val = cal_data.get("abstain_threshold")
                    if thresh_val is not None:
                        self._abstain_threshold = float(thresh_val)
                except Exception:
                    self._temperature = None
                    self._abstain_threshold = None

    def predict(self, text: str) -> dict[str, Any]:
        """Classify a single input text string.

        Args:
            text: Non-empty, non-whitespace string to classify.

        Returns:
            Dictionary containing:
                - 'label': Predicted emotion name string.
                - 'confidence': Highest prediction probability float.
                - 'all_probs': Dict of {label: prob} for all 6 classes in config.LABELS order.
                - 'is_confident' (optional): Bool indicating whether confidence meets abstain threshold.

        Raises:
            ValueError: If input text is None, empty, or whitespace-only.
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify a batch of input text strings.

        Args:
            texts: List of non-empty, non-whitespace strings to classify.

        Returns:
            List of prediction dictionaries with keys 'label', 'confidence', 'all_probs',
            and optionally 'is_confident'.

        Raises:
            ValueError: If any item in texts is None, empty, or whitespace-only.
        """
        if not texts:
            return []

        # Validate all inputs
        for idx, t in enumerate(texts):
            if t is None or not isinstance(t, str) or not t.strip():
                raise ValueError(
                    f"Empty or whitespace-only input encountered at index {idx}. "
                    "Input text must contain non-whitespace characters."
                )

        self._ensure_loaded()

        if self.model_type == "baseline":
            cleaned_texts = [heavy_clean(t) for t in texts]
            raw_probs = self._model.predict_proba(cleaned_texts)

            # Reorder probability columns to match canonical config.LABELS order
            class_to_idx = {cls_name: i for i, cls_name in enumerate(self._model.classes_)}
            reorder_indices = [class_to_idx[lbl] for lbl in LABELS]
            probs_arr = raw_probs[:, reorder_indices]

        else:  # distilbert and distilbert_v2
            cleaned_texts = [minimal_clean(t) for t in texts]
            encoded = self._tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
                if self._temperature is not None and self._temperature > 0:
                    logits = logits / self._temperature
                probs_tensor = torch.softmax(logits, dim=1)
                probs_arr = probs_tensor.cpu().numpy()

        results: list[dict[str, Any]] = []
        for i in range(len(texts)):
            row_probs = probs_arr[i]
            pred_idx = int(np.argmax(row_probs))
            pred_label = LABELS[pred_idx]
            confidence = float(row_probs[pred_idx])
            conf_rounded = round(confidence, 4)

            all_probs = {
                label: round(float(row_probs[j]), 4) for j, label in enumerate(LABELS)
            }

            item: dict[str, Any] = {
                "label": pred_label,
                "confidence": conf_rounded,
                "all_probs": all_probs,
            }
            if self._abstain_threshold is not None:
                item["is_confident"] = bool(conf_rounded >= self._abstain_threshold)

            results.append(item)

        return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prediction entrypoint."""
    parser = argparse.ArgumentParser(description="Predict emotion from text.")
    parser.add_argument("text", type=str, help="Text string to classify.")
    parser.add_argument(
        "--model",
        type=str,
        default=ACTIVE_DISTILBERT,
        choices=list(MODEL_DIRS.keys()),
        help=f"Model architecture to use ('baseline', 'distilbert', or 'distilbert_v2'; default: {ACTIVE_DISTILBERT}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    model_type = args.model
    avail = available_models()
    if model_type == ACTIVE_DISTILBERT and ACTIVE_DISTILBERT not in avail:
        print(f"Note: Active model '{ACTIVE_DISTILBERT}' is not installed; falling back to 'baseline'.")
        model_type = "baseline"

    predictor = EmotionPredictor(model_type=model_type)
    result = predictor.predict(args.text)

    print("\n" + "=" * 50)
    print("EMOTION DETECTION INFERENCE")
    print("=" * 50)
    print(f"Input:        \"{args.text}\"")
    print(f"Model:        {model_type}")
    print(f"Prediction:   {result['label']} (Confidence: {result['confidence']:.4f})")
    if "is_confident" in result:
        status_text = "CONFIDENT" if result["is_confident"] else "ABSTAIN (Uncertain / Low Confidence)"
        print(f"Confidence:   {status_text} (is_confident: {result['is_confident']})")
    print("-" * 50)
    print("Class Probabilities:")
    for label, prob in result["all_probs"].items():
        bar = "█" * int(prob * 30)
        print(f"  {label:10s} : {prob:.4f}  {bar}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
