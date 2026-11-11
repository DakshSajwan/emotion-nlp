"""Model download and installation utility.

Uses only the Python standard library to download, verify, and unpack pre-trained
model artifacts with retry logic, progress reporting, and zip-slip security guards.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

from src.config import (
    ACTIVE_DISTILBERT,
    MODEL_DIRS,
    MODELS_DIR,
    MODELS_SHA256,
    MODELS_URL,
    PROJECT_ROOT,
    available_models,
)

MANUAL_ALTERNATIVE_MSG: str = """
================================================================================
MANUAL INSTALLATION INSTRUCTIONS:
================================================================================
1. Download 'models.zip' directly from the project's GitHub Releases page:
   https://github.com/<DakshSajwan>/<emotion-nlp>/releases

2. Install the local zip archive using:
   python -m src.download_models --from-file <path_to_models.zip>

Alternatively, train models from scratch:
   python -m src.data
   python -m src.baseline
   python -m src.transformer --run_name distilbert_v2
================================================================================
"""


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Verify SHA256 checksum of a file against expected hash.

    Args:
        file_path: Path to target file.
        expected_hash: Hexadecimal SHA256 string.

    Returns:
        True if hashes match or expected_hash is empty; False otherwise.
    """
    if not expected_hash or not expected_hash.strip():
        return True

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    computed = sha256.hexdigest().lower()
    return computed == expected_hash.strip().lower()


def download_with_progress(
    url: str,
    destination: Path,
    timeout_sec: int = 60,
    max_retries: int = 3,
) -> None:
    """Download a URL to destination path with progress reporting and retries.

    Args:
        url: Remote HTTP/HTTPS URL.
        destination: Target local file path.
        timeout_sec: Socket timeout in seconds.
        max_retries: Maximum download attempts.

    Raises:
        RuntimeError: If download fails after max_retries attempts.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; EmotionNLP-Downloader/1.0)",
    }

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading models archive (attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                total_bytes = response.getheader("Content-Length")
                total_size = int(total_bytes) if total_bytes and total_bytes.isdigit() else 0
                total_mb = total_size / (1024 * 1024) if total_size > 0 else 0.0

                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunk

                with open(destination, "wb") as out_f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        dl_mb = downloaded / (1024 * 1024)

                        if total_mb > 0:
                            percent = (downloaded / total_size) * 100
                            print(
                                f"\r  Progress: {dl_mb:.2f} MB / {total_mb:.2f} MB ({percent:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r  Progress: {dl_mb:.2f} MB downloaded",
                                end="",
                                flush=True,
                            )

                print("\n  Download complete.")
                return

        except Exception as err:
            last_error = err
            print(f"\n  [!] Attempt {attempt} failed: {err}")
            if destination.exists():
                destination.unlink(missing_ok=True)
            if attempt < max_retries:
                backoff = attempt * 2
                print(f"  Retrying in {backoff} seconds...")
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to download models from {url} after {max_retries} attempts: {last_error}"
    )


def extract_archive_safe(zip_path: Path, target_models_dir: Path) -> list[Path]:
    """Safely extract zip archive into models directory with zip-slip protection.

    Args:
        zip_path: Path to the zip file.
        target_models_dir: Destination directory (models/).

    Returns:
        List of extracted root model directories/files.

    Raises:
        RuntimeError: If zip-slip path traversal is detected or zip is invalid.
    """
    target_models_dir.mkdir(parents=True, exist_ok=True)
    models_dir_resolved = target_models_dir.resolve()
    extracted_items: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = member.filename

            # Determine extraction relative destination
            if member_path.startswith("models/"):
                rel_dest = member_path[len("models/") :]
            elif member_path.startswith("models\\"):
                rel_dest = member_path[len("models\\") :]
            else:
                rel_dest = member_path

            if not rel_dest or rel_dest.endswith("/"):
                # Directory entry
                dest_dir = (target_models_dir / rel_dest).resolve()
                if not str(dest_dir).startswith(str(models_dir_resolved)):
                    raise RuntimeError(f"Zip-slip traversal detected in directory: {member.filename}")
                dest_dir.mkdir(parents=True, exist_ok=True)
                continue

            dest_file = (target_models_dir / rel_dest).resolve()

            # Zip-slip security guard: ensure target file is strictly inside models/
            try:
                dest_file.relative_to(models_dir_resolved)
            except ValueError:
                raise RuntimeError(
                    f"Zip-slip security violation! Member {member.filename} resolves "
                    f"outside target directory: {dest_file}"
                )

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest_file, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted_items.append(dest_file)

    return extracted_items


def validate_extracted_models(models_dir: Path) -> None:
    """Validate that extracted transformer models have required configuration and weights.

    Args:
        models_dir: Path to models/ directory.

    Raises:
        RuntimeError: If any transformer model folder lacks config, weights, or tokenizer.
    """
    for entry in models_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            # Check for config.json
            config_file = entry / "config.json"
            if not config_file.exists():
                raise RuntimeError(
                    f"Model validation failed for '{entry.name}': Missing config.json"
                )

            # Check for model weights
            weights_file = entry / "model.safetensors"
            bin_file = entry / "pytorch_model.bin"
            if not (weights_file.exists() or bin_file.exists()):
                raise RuntimeError(
                    f"Model validation failed for '{entry.name}': Missing model weights (model.safetensors)"
                )

            # Check for tokenizer files
            tok_json = entry / "tokenizer.json"
            tok_config = entry / "tokenizer_config.json"
            if not (tok_json.exists() or tok_config.exists()):
                raise RuntimeError(
                    f"Model validation failed for '{entry.name}': Missing tokenizer files"
                )


def run_download_or_install(
    from_file: str | None = None,
    force: bool = False,
) -> int:
    """Download, verify, and unpack model artifacts.

    Returns:
        0 on success, non-zero exit code on failure.
    """
    # Check if models are already present
    current_models = available_models()
    if not force and not from_file and ACTIVE_DISTILBERT in current_models and "baseline" in current_models:
        print(f"Models already installed: {current_models}. Use --force to re-install.")
        return 0

    temp_zip: Path | None = None
    try:
        if from_file:
            archive_path = Path(from_file).resolve()
            if not archive_path.is_file():
                print(f"Error: Specified local archive file does not exist: {archive_path}")
                print(MANUAL_ALTERNATIVE_MSG)
                return 1
            print(f"Using local model archive: {archive_path}")
        else:
            temp_fd, temp_path_str = tempfile.mkstemp(prefix="emotion_models_", suffix=".zip")
            os.close(temp_fd)
            temp_zip = Path(temp_path_str)

            print(f"Connecting to model repository: {MODELS_URL}")
            download_with_progress(MODELS_URL, temp_zip, timeout_sec=60, max_retries=3)
            archive_path = temp_zip

        # Verify SHA-256 Checksum if specified
        if MODELS_SHA256:
            print("Verifying SHA256 checksum...")
            if not verify_sha256(archive_path, MODELS_SHA256):
                print("Error: SHA256 checksum verification failed!")
                print(MANUAL_ALTERNATIVE_MSG)
                return 1
            print("  ✓ Checksum verified successfully.")

        # Extract archive safely
        print(f"Extracting model files into {MODELS_DIR}...")
        extract_archive_safe(archive_path, MODELS_DIR)

        # Validate extracted files
        print("Validating extracted model artifacts...")
        validate_extracted_models(MODELS_DIR)

        print("\n✓ Model installation successful!")
        print(f"  Available models: {available_models()}")
        return 0

    except Exception as err:
        print(f"\n[!] Model download and extraction failed: {err}")
        print(MANUAL_ALTERNATIVE_MSG)
        return 1

    finally:
        if temp_zip and temp_zip.exists():
            temp_zip.unlink(missing_ok=True)


def main() -> None:
    """CLI entrypoint for python -m src.download_models."""
    parser = argparse.ArgumentParser(
        description="Download and install pre-trained emotion detection models."
    )
    parser.add_argument(
        "--from-file",
        type=str,
        default=None,
        help="Path to a local models.zip file, skipping remote download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction even if models exist.",
    )
    args = parser.parse_args()

    exit_code = run_download_or_install(from_file=args.from_file, force=args.force)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
