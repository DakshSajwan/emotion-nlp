#!/usr/bin/env python3
"""
package_submission.py

Pure Python, cross-platform, standard library only script to package the
Emotion Detection NLP project for formal submission and deployment.

It creates a 'submission/' directory containing:
1. emotion-nlp-code.zip: The complete project codebase, documentation,
   datasets (including data/probe_set.csv and data/train_v2.csv), excluding
   virtual environments, caches, git history, models, smoke artifacts,
   and previous archives.
2. emotion-nlp-models.zip: Serialized trained models (models/baseline.joblib,
   models/distilbert/, and models/distilbert_v2/ with calibration.json).
3. HOW_TO_RUN.txt: Standalone, comprehensive instructions for environment setup,
   model placement, test verification, app launching, and demonstration commands.

Usage:
    python package_submission.py
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from typing import Set, Tuple

# Project root and submission directory paths
PROJECT_ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = PROJECT_ROOT / "submission"
CODE_ZIP_PATH = SUBMISSION_DIR / "emotion-nlp-code.zip"
MODELS_ZIP_PATH = SUBMISSION_DIR / "emotion-nlp-models.zip"
HOW_TO_RUN_PATH = SUBMISSION_DIR / "HOW_TO_RUN.txt"

# Exclusion sets for the codebase archive
EXCLUDED_DIR_NAMES: Set[str] = {
    ".venv",
    "venv",
    "env",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "models",
    "submission",
}

EXCLUDED_FILE_SUFFIXES: Set[str] = {
    ".zip",
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store",
}


def should_exclude_path(rel_path: Path) -> bool:
    """Determine whether a relative path should be excluded from emotion-nlp-code.zip."""
    parts = rel_path.parts

    # Exclude if any folder component matches known excluded folder names
    for part in parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        if "smoke" in part.lower():
            return True

    # Exclude if file itself contains 'smoke' or has excluded file extensions
    filename = rel_path.name
    if "smoke" in filename.lower():
        return True
    if any(filename.endswith(suffix) for suffix in EXCLUDED_FILE_SUFFIXES):
        return True

    return False


def build_code_zip(project_root: Path, output_zip_path: Path) -> Tuple[int, int]:
    """
    Package the project codebase into a compressed zip archive.
    
    Returns:
        (file_count, uncompressed_bytes)
    """
    file_count = 0
    total_uncompressed = 0

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_root):
            # Prune excluded directories in-place to avoid traversing them
            dirs[:] = [
                d
                for d in dirs
                if d not in EXCLUDED_DIR_NAMES and "smoke" not in d.lower()
            ]

            for file in sorted(files):
                abs_path = Path(root) / file
                try:
                    rel_path = abs_path.relative_to(project_root)
                except ValueError:
                    continue

                if should_exclude_path(rel_path):
                    continue

                zf.write(abs_path, arcname=str(rel_path))
                file_count += 1
                total_uncompressed += abs_path.stat().st_size

    return file_count, total_uncompressed


def build_models_zip(project_root: Path, output_zip_path: Path) -> Tuple[int, int]:
    """
    Package only models/baseline.joblib and models/distilbert/ into emotion-nlp-models.zip.
    
    Returns:
        (file_count, uncompressed_bytes)
    """
    models_dir = project_root / "models"
    file_count = 0
    total_uncompressed = 0

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found at: {models_dir}")

    targets = [
        models_dir / "baseline.joblib",
        models_dir / "distilbert",
        models_dir / "distilbert_v2",
    ]

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for target in targets:
            if not target.exists():
                print(f"  [!] Warning: Model path does not exist: {target}")
                continue

            if target.is_file():
                rel_path = target.relative_to(project_root)
                zf.write(target, arcname=str(rel_path))
                file_count += 1
                total_uncompressed += target.stat().st_size
            elif target.is_dir():
                for root, _, files in os.walk(target):
                    for file in sorted(files):
                        abs_file = Path(root) / file
                        if "smoke" in file.lower() or file.endswith(".pyc") or file == ".DS_Store":
                            continue
                        rel_path = abs_file.relative_to(project_root)
                        zf.write(abs_file, arcname=str(rel_path))
                        file_count += 1
                        total_uncompressed += abs_file.stat().st_size

    return file_count, total_uncompressed


def write_how_to_run(destination: Path) -> None:
    """Write the comprehensive, self-contained HOW_TO_RUN.txt file."""
    content = """================================================================================
           EMOTION DETECTION IN TEXT: PROJECT REPRODUCTION GUIDE
================================================================================

This project implements an end-to-end Natural Language Processing system that
empirically benchmarks a classical Machine Learning pipeline (TF-IDF with Logistic
Regression) against fine-tuned Transformers (DistilBERT v1 and DistilBERT v2)
on the dair-ai/emotion benchmark (6 affective classes: sadness, joy, love,
anger, fear, surprise) and an out-of-distribution stress probe set. It includes
temperature scaling calibration, selective prediction abstention, Local
Interpretable Model-agnostic Explanations (LIME), and an interactive 5-tab
Streamlit web application.

ACTIVE PRODUCTION MODEL:
  - DistilBERT v2 (improved with GoEmotions data augmentation, label smoothing
    alpha=0.1, temperature scaling T=0.6775, and confidence abstention tau=0.90).
  - DistilBERT v1 and Baseline (TF-IDF + LR) remain packaged and fully selectable.

--------------------------------------------------------------------------------
1. PREREQUISITES & PYTHON VERSION
--------------------------------------------------------------------------------
- Python Version: Python 3.10, 3.11, or 3.12 (Python 3.10+ required)
- Operating System: macOS, Windows, or Linux (cross-platform compatible)
- Hardware Requirements: Standard modern CPU with at least 4 GB RAM.
  (All training and inference can execute purely on CPU without a GPU).
- Estimated Disk Space: ~1.8 GB (code + virtual environment dependencies + models).

--------------------------------------------------------------------------------
2. PROJECT & MODEL EXTRACTION
--------------------------------------------------------------------------------
The project submission is split into two zip archives:
  a) emotion-nlp-code.zip   -> Full source code, tests, documentation, figures,
                               metrics, and datasets (including data/probe_set.csv
                               and data/train_v2.csv).
  b) emotion-nlp-models.zip -> Serialized pre-trained model weights for Baseline,
                               DistilBERT v1, and DistilBERT v2 (with calibration.json).

STEP 1: Extract 'emotion-nlp-code.zip' to your working directory:
        e.g., /path/to/NLP-Project/ or C:\\Projects\\NLP-Project\\

STEP 2: Extract 'emotion-nlp-models.zip' directly into the project root directory
        so that models are located under 'models/':

        Expected folder layout after extraction:
        NLP Project/
        ├── models/
        │   ├── baseline.joblib
        │   ├── distilbert/
        │   │   ├── config.json
        │   │   ├── model.safetensors
        │   │   ├── tokenizer.json
        │   │   ├── tokenizer_config.json
        │   │   └── training_args.bin
        │   └── distilbert_v2/
        │       ├── calibration.json
        │       ├── config.json
        │       ├── model.safetensors
        │       ├── tokenizer.json
        │       ├── tokenizer_config.json
        │       └── training_args.bin

--------------------------------------------------------------------------------
3. VIRTUAL ENVIRONMENT & DEPENDENCY INSTALLATION
--------------------------------------------------------------------------------
Open a terminal / command prompt, change directory to the project root, and run:

[ macOS / Linux ]
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt

[ Windows (Command Prompt / PowerShell) ]
  python -m venv .venv
  .venv\\Scripts\\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt

--------------------------------------------------------------------------------
4. HOW TO RUN THE AUTOMATED TESTS
--------------------------------------------------------------------------------
To verify data integrity, preprocessing contracts, model architectures, calibration,
selective abstention, and inference endpoints across the whole project, run:

  python -m pytest -q

Expected result:
  35 passed in ~7 seconds (100% test pass rate).

--------------------------------------------------------------------------------
5. HOW TO LAUNCH THE INTERACTIVE STREAMLIT APPLICATION
--------------------------------------------------------------------------------
Launch the Streamlit web dashboard using the top-level orchestrator:

  python run_all.py app

or directly via Streamlit:

  python -m streamlit run app.py

Once running, navigate your web browser to:
  http://localhost:8501

Interactive Tabs:
  - Tab 1: Analyze (Single-text inference with emotion badge, calibrated probability bars,
           and selective abstention banner "Not sure: low confidence" when confidence < 0.90)
  - Tab 2: Explain (LIME token-attribution heatmaps and perturbation weights)
  - Tab 3: Compare Models (Side-by-side Baseline vs. DistilBERT v1 vs. DistilBERT v2)
  - Tab 4: Batch Processing (Upload CSV, predict all rows, export results)
  - Tab 5: Results & Architecture (Full test benchmark tables and confusion matrices)

--------------------------------------------------------------------------------
6. FINAL COMMAND LIST FOR DEMO & CLI EVALUATION
--------------------------------------------------------------------------------
1. Single-text prediction via CLI:
   # Active promoted model (DistilBERT v2 with calibrated probabilities and abstention):
   python -m src.predict "I just got my dream job! Truly thankful!" --model distilbert_v2

   # Low-confidence / neutral input demonstrating abstention (is_confident: False):
   python -m src.predict "The flight arrives at gate B12." --model distilbert_v2

   # DistilBERT v1 prediction:
   python -m src.predict "I just got my dream job! Truly thankful!" --model distilbert

   # Baseline prediction:
   python -m src.predict "I just got my dream job! Truly thankful!" --model baseline

2. Calibration and Robustness Evaluation CLI:
   python -m src.calibrate         # Computes temperature scaling & ECE on test split
   python -m src.probe_eval        # Evaluates Baseline, v1, and v2 on stress probe set

3. Pipeline CLI Orchestrator:
   python run_all.py eda          # Regenerates EDA distributions & charts
   python run_all.py evaluate     # Runs full evaluation on 2,000 test samples
   python run_all.py compare      # Generates comparison tables & error analysis
   python run_all.py explain      # Generates LIME explanation artifacts

4. Demo & Presentation Guides:
   - reports/demo_script.md      # Timed 5-minute presentation script & 10-slide deck
   - reports/project_report.md   # Comprehensive academic & technical report
   - reports/code_walkthrough.md # Architecture walkthrough & module-by-module guide
   - reports/viva_questions.md   # 43 viva examination questions and answers
================================================================================
"""
    destination.write_text(content.strip() + "\n", encoding="utf-8")


def format_size(bytes_val: int) -> str:
    """Format bytes into a human-readable string (KB or MB)."""
    if bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val / (1024 * 1024):.2f} MB"


def main() -> None:
    """Execute packaging routine for project submission."""
    print("=" * 70)
    print("  EMOTION DETECTION NLP: PACKAGING SUBMISSION")
    print("=" * 70)
    print(f"Project Root   : {PROJECT_ROOT}")
    print(f"Submission Dir : {SUBMISSION_DIR}")

    # Ensure clean submission directory
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build emotion-nlp-code.zip
    print("\n[1/3] Packaging project codebase -> emotion-nlp-code.zip ...")
    code_count, code_uncompressed = build_code_zip(PROJECT_ROOT, CODE_ZIP_PATH)
    code_zip_size = CODE_ZIP_PATH.stat().st_size
    print(f"  ✓ Files included        : {code_count}")
    print(f"  ✓ Uncompressed size     : {format_size(code_uncompressed)}")
    print(f"  ✓ Compressed zip size   : {format_size(code_zip_size)}")
    print(f"  ✓ Output location       : {CODE_ZIP_PATH}")

    # 2. Build emotion-nlp-models.zip
    print("\n[2/3] Packaging serialized models -> emotion-nlp-models.zip ...")
    model_count, model_uncompressed = build_models_zip(PROJECT_ROOT, MODELS_ZIP_PATH)
    model_zip_size = MODELS_ZIP_PATH.stat().st_size
    print(f"  ✓ Files included        : {model_count}")
    print(f"  ✓ Uncompressed size     : {format_size(model_uncompressed)}")
    print(f"  ✓ Compressed zip size   : {format_size(model_zip_size)}")
    print(f"  ✓ Output location       : {MODELS_ZIP_PATH}")

    # 3. Write HOW_TO_RUN.txt
    print("\n[3/3] Generating submission documentation -> HOW_TO_RUN.txt ...")
    write_how_to_run(HOW_TO_RUN_PATH)
    print(f"  ✓ Instructions written  : {HOW_TO_RUN_PATH} ({HOW_TO_RUN_PATH.stat().st_size} bytes)")

    # Summary
    print("\n" + "=" * 70)
    print("  SUBMISSION PACKAGING COMPLETE")
    print("=" * 70)
    print(f"  1. Code Zip   : {CODE_ZIP_PATH.name} ({format_size(code_zip_size)})")
    print(f"  2. Models Zip : {MODELS_ZIP_PATH.name} ({format_size(model_zip_size)})")
    print(f"  3. Guide      : {HOW_TO_RUN_PATH.name}")
    print(f"  Location      : {SUBMISSION_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
