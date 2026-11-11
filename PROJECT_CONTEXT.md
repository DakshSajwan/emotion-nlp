# Project Context: Emotion Detection in Text

## Project Goal
Compare a classical baseline (TF-IDF + Logistic Regression) with a fine-tuned DistilBERT transformer on the HuggingFace `dair-ai/emotion` dataset (6 labels: sadness, joy, love, anger, fear, surprise), augment with LIME explainability, and build an interactive Streamlit demo. The project is designed to run efficiently on CPU with free, open-source libraries.

---

## Project Structure
```
NLP Project/
├── .gitignore
├── .venv/
├── PROJECT_CONTEXT.md
├── README.md
├── requirements.txt
├── app.py
├── data/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── baseline.py
│   ├── transformer.py
│   ├── compare.py
│   ├── explain.py
│   └── predict.py
└── tests/
    └── __init__.py
```

---

## The PY Command
Always execute Python scripts and commands explicitly through the virtual environment's interpreter from the workspace root:

```bash
.venv/bin/python
```

Execute all project modules from the workspace root via the `-m` flag:
```bash
.venv/bin/python -m src.<module>
```

---

## The `src/config.py` Single Source of Truth Rule
- `src/config.py` is the **single source of truth** for all project configurations:
  - `SEED = 42`
  - Absolute paths constructed using `pathlib` relative to `PROJECT_ROOT` (`DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `FIGURES_DIR`, `PREDICTIONS_DIR`, `METRICS_DIR`, etc.)
  - Dataset labels in canonical order: `LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]`
  - Mapping dictionaries: `id2label` and `label2id`
  - Model constants: `MODEL_NAME = "distilbert-base-uncased"`, `MAX_LENGTH = 64`
  - Reproducibility helper: `set_seed(seed=SEED)` seeding Python's `random`, `numpy`, and `torch`
  - **Active Model Setting**: `ACTIVE_DISTILBERT = "distilbert_v2"` designating the promoted production transformer
  - Dynamic artifact routing: `ACTIVE_DISTILBERT_DIR = MODEL_DIRS[ACTIVE_DISTILBERT]` and `ACTIVE_DISTILBERT_PREDS = PREDICTIONS_DIR / f"{ACTIVE_DISTILBERT}_test_preds.csv"`
- **No other file may hardcode paths, label names, label mappings, or seeds.**

---

## Coding Rules
1. **Type Hints**: All functions and methods must include explicit type hints for arguments and return values.
2. **Docstrings**: Provide clear, standard docstrings for every module, class, and public function.
3. **Reproducibility**: Use `SEED = 42` consistently across all models, splits, and stochastic components.
4. **No Hardcoding**: Import paths, labels, and constants strictly from `src.config`.

---

## Working Rules
1. **Be Concise**: Keep explanations direct and avoid unnecessary verbosity.
2. **No Whole-File Reprints**: Do not reprint full files in chat replies; use targeted diffs or line ranges.
3. **Execution Verification**: Always execute code after writing it to verify correctness.
4. **Self-Healing**: Fix any errors immediately before reporting back.
5. **No Invented Metrics**: Record and report only actual computed metrics from dataset splits.
6. **No Blocking Commands**: Never run servers or training jobs that block the terminal without a timeout or background management.
7. **Checklist Progression**: Update and check off items in the checklist after completing each step.

---

## Prediction File Contract
Every trained model must save its test set predictions to:
`reports/predictions/<model>_test_preds.csv`

The file must contain exactly the following columns:
- `text`: Raw input string.
- `true_label`: Ground truth label string (label NAME, e.g., `"joy"`, not integer ID).
- `pred_label`: Predicted label string (label NAME, e.g., `"sadness"`, not integer ID).
- `confidence`: Highest class probability (float between 0.0 and 1.0).
- `prob_sadness`: Predicted probability for sadness.
- `prob_joy`: Predicted probability for joy.
- `prob_love`: Predicted probability for love.
- `prob_anger`: Predicted probability for anger.
- `prob_fear`: Predicted probability for fear.
- `prob_surprise`: Predicted probability for surprise.

---

## Project Checklist
- [x] 0 setup (Python 3.10+, venv, dependencies, folder structure, config, git init)
- [x] 1 data+EDA
- [x] 2 preprocess
- [x] 3 evaluate module
- [x] 4 baseline
- [x] 5 DistilBERT
- [x] 6 compare
- [x] 7 LIME
- [x] 8 predict API+tests
- [x] 9 Streamlit
- [x] 10 README/report/reproducibility
- [x] 11 probe set (independent faculty evaluation set with 144 sentences and neutral OOD detection)
- [x] 12 data_v2 (multi-source GoEmotions ingestion, prefix stripping, zero-leakage filter)
- [x] 13 v2 training (DistilBERT v2 trained on train_v2 with label smoothing 0.1)
- [x] 14 calibration (temperature scaling T=0.6775 on val, abstain threshold 0.90 on probe dev)
- [x] 15 promotion (probe test evaluation audit; promoted ACTIVE_DISTILBERT = "distilbert_v2")
