"""Cross-platform workflow orchestrator for Emotion Detection in Text.

Provides a unified CLI interface to execute pipeline steps: setup, data, eda,
baseline, transformer, compare, explain, test, app, and all.
Executes pure Python subprocesses using the current active virtual environment interpreter.
"""

from typing import Optional
import argparse
import os
from pathlib import Path
import subprocess
import sys

# Workspace Root
PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(args: list[str], desc: str) -> None:
    """Execute a subprocess command with clear logging and error handling.

    Args:
        args: Command arguments list.
        desc: Short human-readable task description.

    Raises:
        SystemExit: If the subprocess command exits with a non-zero code.
    """
    print("\n" + "=" * 60)
    print(f"STAGE: {desc}")
    print(f"COMMAND: {' '.join(args)}")
    print("=" * 60 + "\n")

    try:
        res = subprocess.run(args, cwd=str(PROJECT_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ Error during '{desc}' (Exit code: {exc.returncode})", file=sys.stderr)
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\n❌ Execution failure during '{desc}': {exc}", file=sys.stderr)
        sys.exit(1)


def step_setup() -> None:
    """Verify environment, dependencies, and download required NLTK corpora."""
    print("Verifying Python environment...")
    v = sys.version_info
    print(f"Current Python interpreter: {sys.executable}")
    print(f"Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("❌ Error: Python 3.10+ is required.", file=sys.stderr)
        sys.exit(1)

    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            "Installing project requirements",
        )

    # Trigger NLTK download helper in preprocess
    run_command(
        [
            sys.executable,
            "-c",
            "import src.preprocess as p; p.ensure_nltk_resources(); print('NLTK resources verified.')",
        ],
        "Verifying NLTK corpora",
    )
    print("\n✅ Environment setup successfully completed.")


def step_data() -> None:
    """Download and prepare the Hugging Face emotion dataset."""
    run_command([sys.executable, "-m", "src.data"], "Data Ingestion & Verification")


def step_eda() -> None:
    """Execute exploratory data analysis and generate dataset visualizations."""
    run_command([sys.executable, "-m", "src.eda"], "Exploratory Data Analysis (EDA)")


def step_baseline() -> None:
    """Train and evaluate classical TF-IDF + Logistic Regression baseline."""
    run_command([sys.executable, "-m", "src.baseline"], "Baseline Model Training & Evaluation")


def step_transformer(epochs: Optional[int] = None, subset: Optional[int] = None) -> None:
    """Train and evaluate DistilBERT sequence classifier."""
    cmd = [sys.executable, "-m", "src.transformer"]
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if subset is not None:
        cmd.extend(["--subset", str(subset)])
    run_command(cmd, "DistilBERT Model Training & Evaluation")


def step_compare() -> None:
    """Compare predictions, metrics, and generate synthesis reports."""
    run_command([sys.executable, "-m", "src.compare"], "Model Comparative Synthesis")


def step_explain() -> None:
    """Generate LIME explanations for benchmark examples."""
    run_command([sys.executable, "-m", "src.explain"], "LIME Interpretability Generation")


def step_test() -> None:
    """Run full pytest automated test suite."""
    run_command([sys.executable, "-m", "pytest", "-v"], "Pytest Test Suite Execution")


def step_app() -> None:
    """Launch interactive Streamlit web application."""
    run_command([sys.executable, "-m", "streamlit", "run", "app.py"], "Streamlit Web Demo")


def step_all() -> None:
    """Execute complete end-to-end pipeline."""
    print("\n" + "#" * 60)
    print("EXECUTING COMPLETE END-TO-END NLP PIPELINE")
    print("#" * 60)
    step_setup()
    step_data()
    step_eda()
    step_baseline()
    step_transformer()
    step_compare()
    step_explain()
    step_test()
    print("\n" + "#" * 60)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("#" * 60 + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Cross-platform orchestrator for Emotion Detection in Text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Pipeline subcommand to execute")

    subparsers.add_parser("setup", help="Verify environment, install requirements, download NLTK data")
    subparsers.add_parser("data", help="Download and verify dataset splits")
    subparsers.add_parser("eda", help="Run exploratory data analysis and generate figures")
    subparsers.add_parser("baseline", help="Train and evaluate TF-IDF + Logistic Regression baseline")

    trans_p = subparsers.add_parser("transformer", help="Train and evaluate DistilBERT model")
    trans_p.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: 3)")
    trans_p.add_argument("--subset", type=int, default=None, help="Optional subset size for fast smoke testing")

    subparsers.add_parser("compare", help="Compare models, generate comparison table and error analysis")
    subparsers.add_parser("explain", help="Generate LIME explanations for benchmark examples")
    subparsers.add_parser("test", help="Run full pytest test suite")
    subparsers.add_parser("app", help="Launch Streamlit interactive web application")
    subparsers.add_parser("all", help="Execute all pipeline stages sequentially")

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    command_map = {
        "setup": step_setup,
        "data": step_data,
        "eda": step_eda,
        "baseline": step_baseline,
        "transformer": lambda: step_transformer(epochs=args.epochs, subset=args.subset),
        "compare": step_compare,
        "explain": step_explain,
        "test": step_test,
        "app": step_app,
        "all": step_all,
    }

    fn = command_map.get(args.subcommand)
    if fn:
        fn()


if __name__ == "__main__":
    main()
