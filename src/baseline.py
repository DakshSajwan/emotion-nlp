"""Classical baseline model for Emotion Detection in Text.

Pipeline:
- Feature Extraction: TfidfVectorizer(max_features=20000)
- Classification: LogisticRegression(class_weight='balanced', max_iter=1000)
- Text Preprocessing: preprocess.heavy_clean
- Hyperparameter Tuning: GridSearchCV with PredefinedSplit over fixed train/val splits
- Evaluation: evaluate.evaluate_model conforming to project contract
"""

from typing import Any
import json
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.config import LABELS, METRICS_DIR, MODELS_DIR, SEED, set_seed
from src.data import get_splits
from src.evaluate import evaluate_model
from src.preprocess import heavy_clean


def build_pipeline(C: float = 1.0, ngram_range: tuple[int, int] = (1, 1)) -> Pipeline:
    """Construct the TF-IDF + Logistic Regression scikit-learn pipeline.

    Args:
        C: Inverse regularization strength parameter for LogisticRegression.
        ngram_range: Lower and upper boundary of the range of n-values for n-grams.

    Returns:
        Configured scikit-learn Pipeline instance.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=ngram_range,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED,
                solver="lbfgs",
            ),
        ),
    ])


def tune_baseline(
    X_train_clean: list[str],
    y_train: list[str],
    X_val_clean: list[str],
    y_val: list[str],
) -> tuple[dict[str, Any], float]:
    """Tune C and ngram_range using GridSearchCV with PredefinedSplit over train+val.

    Args:
        X_train_clean: Preprocessed training text strings.
        y_train: Ground truth training labels.
        X_val_clean: Preprocessed validation text strings.
        y_val: Ground truth validation labels.

    Returns:
        Tuple of (best_params dictionary, best validation macro-F1 score).
    """
    param_grid = {
        "clf__C": [0.5, 1.0, 5.0, 10.0],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
    }

    # Fixed validation split: -1 for train, 0 for validation
    test_fold = [-1] * len(X_train_clean) + [0] * len(X_val_clean)
    ps = PredefinedSplit(test_fold=test_fold)

    X_combined = X_train_clean + X_val_clean
    y_combined = y_train + y_val

    base_pipeline = build_pipeline()

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=ps,
        scoring="f1_macro",
        refit=False,
        n_jobs=-1,
        verbose=1,
    )

    print("Beginning hyperparameter tuning over fixed validation split...")
    grid.fit(X_combined, y_combined)

    print(f"Hyperparameter tuning complete. Best params: {grid.best_params_}")
    print(f"Best validation macro-F1: {grid.best_score_:.4f}")

    return grid.best_params_, float(grid.best_score_)


def train_baseline(
    X_train_clean: list[str],
    y_train: list[str],
    best_params: dict[str, Any],
) -> tuple[Pipeline, float]:
    """Fit the best pipeline configuration strictly on the training set.

    Args:
        X_train_clean: Preprocessed training text strings.
        y_train: Ground truth training labels.
        best_params: Best hyperparameter dictionary returned by tuning.

    Returns:
        Tuple of (fitted Pipeline, training duration in seconds).
    """
    pipeline = build_pipeline(
        C=best_params["clf__C"],
        ngram_range=best_params["tfidf__ngram_range"],
    )

    print("Refitting best model on training set only...")
    start_time = time.perf_counter()
    pipeline.fit(X_train_clean, y_train)
    training_time_s = time.perf_counter() - start_time
    print(f"Training completed in {training_time_s:.2f} seconds.")

    return pipeline, training_time_s


def run_baseline() -> dict[str, Any]:
    """Execute full baseline workflow: data loading, preprocessing, tuning, refit, eval, and export.

    Returns:
        Dictionary of final evaluation and benchmark metrics.
    """
    set_seed(SEED)

    print("Loading dair-ai/emotion dataset splits...")
    train_data, val_data, test_data = get_splits()

    X_train_raw: list[str] = train_data["text"]
    y_train: list[str] = train_data["label"]

    X_val_raw: list[str] = val_data["text"]
    y_val: list[str] = val_data["label"]

    X_test_raw: list[str] = test_data["text"]
    y_test: list[str] = test_data["label"]

    print("Preprocessing texts with heavy_clean...")
    X_train_clean: list[str] = [heavy_clean(t) for t in tqdm(X_train_raw, desc="Cleaning train")]
    X_val_clean: list[str] = [heavy_clean(t) for t in tqdm(X_val_raw, desc="Cleaning val")]
    X_test_clean: list[str] = [heavy_clean(t) for t in tqdm(X_test_raw, desc="Cleaning test")]

    # 1. Hyperparameter Tuning
    best_params, best_val_score = tune_baseline(
        X_train_clean, y_train, X_val_clean, y_val
    )

    # 2. Refit on train set only
    model, training_time_s = train_baseline(X_train_clean, y_train, best_params)

    # 3. Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "baseline.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # 4. Measure inference time on test set
    start_infer = time.perf_counter()
    raw_probs = model.predict_proba(X_test_clean)
    infer_time_total_s = time.perf_counter() - start_infer

    n_test = len(X_test_clean)
    infer_time_per_1000_s = (infer_time_total_s / n_test) * 1000
    infer_time_per_1000_ms = infer_time_per_1000_s * 1000

    # 5. Reorder probability columns to match canonical config.LABELS order
    class_to_idx = {cls_name: i for i, cls_name in enumerate(model.classes_)}
    reorder_indices = [class_to_idx[lbl] for lbl in LABELS]
    probs = raw_probs[:, reorder_indices]

    # 6. Evaluate and save all prediction/metrics/figure artifacts
    metrics = evaluate_model("baseline", X_test_raw, y_test, probs)

    # 7. Merge benchmark and tuning metadata into reports/metrics/baseline.json
    metrics_file = METRICS_DIR / "baseline.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    benchmark_metadata = {
        "training_time_seconds": round(training_time_s, 4),
        "inference_time_per_1000_samples_ms": round(infer_time_per_1000_ms, 4),
        "inference_time_per_1000_samples_seconds": round(infer_time_per_1000_s, 6),
        "total_test_inference_time_seconds": round(infer_time_total_s, 4),
        "best_params": {
            "C": best_params["clf__C"],
            "ngram_range": list(best_params["tfidf__ngram_range"]),
        },
        "best_val_macro_f1": round(best_val_score, 4),
    }
    saved_metrics.update(benchmark_metadata)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(saved_metrics, f, indent=2)

    print("\nBenchmark Metadata Added to reports/metrics/baseline.json:")
    print(json.dumps(benchmark_metadata, indent=2))

    return saved_metrics


if __name__ == "__main__":
    run_baseline()
