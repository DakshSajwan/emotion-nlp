"""Project-wide configuration and single source of truth.

No other file may hardcode paths or labels.
"""

from pathlib import Path
import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic algorithms where possible
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Project Paths (Absolute Paths via pathlib)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
PREDICTIONS_DIR: Path = REPORTS_DIR / "predictions"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

# Ensure runtime directories exist
for path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, METRICS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset & Model Configuration
# ---------------------------------------------------------------------------
# HuggingFace dair-ai/emotion labels in standard index order
LABELS: list[str] = ["sadness", "joy", "love", "anger", "fear", "surprise"]

id2label: dict[int, str] = {i: label for i, label in enumerate(LABELS)}
label2id: dict[str, int] = {label: i for i, label in enumerate(LABELS)}

NUM_LABELS: int = len(LABELS)

MODEL_NAME: str = "distilbert-base-uncased"
MAX_LENGTH: int = 64

# ---------------------------------------------------------------------------
# GoEmotions to 6-Class Canonical Mapping
# ---------------------------------------------------------------------------
GOEMOTIONS_MAP: dict[str, str] = {
    # joy <- joy, amusement, excitement, optimism, relief, pride, gratitude
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "optimism": "joy",
    "relief": "joy",
    "pride": "joy",
    "gratitude": "joy",
    # love <- love, caring, admiration, desire
    "love": "love",
    "caring": "love",
    "admiration": "love",
    "desire": "love",
    # sadness <- sadness, grief, disappointment, remorse
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    # anger <- anger, annoyance, disapproval, disgust
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",
    # fear <- fear, nervousness
    "fear": "fear",
    "nervousness": "fear",
    # surprise <- surprise, realization
    "surprise": "surprise",
    "realization": "surprise",
}

# ---------------------------------------------------------------------------
# Model Artifact Paths Mapping
# ---------------------------------------------------------------------------
MODEL_DIRS: dict[str, Path] = {
    "baseline": MODELS_DIR / "baseline.joblib",
    "distilbert": MODELS_DIR / "distilbert",
    "distilbert_v2": MODELS_DIR / "distilbert_v2",
}

# ---------------------------------------------------------------------------
# Active Production Transformer Model Configuration
# ---------------------------------------------------------------------------
ACTIVE_DISTILBERT: str = "distilbert_v2"
ACTIVE_DISTILBERT_DIR: Path = MODEL_DIRS[ACTIVE_DISTILBERT]
ACTIVE_DISTILBERT_PREDS: Path = PREDICTIONS_DIR / f"{ACTIVE_DISTILBERT}_test_preds.csv"

# ---------------------------------------------------------------------------
# Remote Model Distribution & Availability
# ---------------------------------------------------------------------------
MODELS_URL = "https://github.com/DakshSajwan/emotion-nlp/releases/download/v1.0/models.zip"
MODELS_SHA256: str = "f4544265c5016f75f1bd064af492ae6fd5275ef3c31959b8dd0ef4c88cd90dc1"


def available_models() -> list[str]:
    """Return model types whose folder or file exists in models/ with weights present."""
    available: list[str] = []
    for model_type, path in MODEL_DIRS.items():
        if model_type == "baseline":
            if path.is_file() and path.stat().st_size > 0:
                available.append(model_type)
        else:
            weights_file = path / "model.safetensors"
            if path.is_dir() and weights_file.is_file() and weights_file.stat().st_size > 0:
                available.append(model_type)
    return available
