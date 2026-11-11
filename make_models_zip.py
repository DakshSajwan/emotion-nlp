#!/usr/bin/env python3
"""Build release_assets/models.zip containing the active production model.

Standard library only, cross-platform script to bundle the active transformer model
for GitHub release distribution, compute its SHA256 checksum, and update src/config.py.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys
import zipfile

from src.config import ACTIVE_DISTILBERT, MODELS_DIR, PROJECT_ROOT

RELEASE_DIR: Path = PROJECT_ROOT / "release_assets"
OUTPUT_ZIP: Path = RELEASE_DIR / "models.zip"
CONFIG_FILE: Path = PROJECT_ROOT / "src" / "config.py"

MAX_RECOMMENDED_BYTES: int = int(1.9 * 1024 * 1024 * 1024)  # 1.9 GB


def compute_sha256(path: Path) -> str:
    """Compute hexadecimal SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()


def update_config_sha256(config_path: Path, new_sha: str) -> None:
    """Update only MODELS_SHA256 in src/config.py without modifying any other lines."""
    content = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r'^(MODELS_SHA256:\s*str\s*=\s*)[^\n]*', re.MULTILINE)

    if not pattern.search(content):
        # Fallback pattern without type annotation
        pattern = re.compile(r'^(MODELS_SHA256\s*=\s*)[^\n]*', re.MULTILINE)
        if not pattern.search(content):
            raise RuntimeError(f"Could not locate MODELS_SHA256 definition in {config_path}")
        new_content = pattern.sub(rf'\g<1>"{new_sha}"', content)
    else:
        new_content = pattern.sub(rf'\g<1>"{new_sha}"', content)

    config_path.write_text(new_content, encoding="utf-8")


def verify_zip_contents(zip_path: Path, expected_folder_name: str) -> list[str]:
    """Verify that the zip archive contains required model and tokenizer artifacts."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

    expected_prefix = f"{expected_folder_name}/"
    model_members = [m for m in members if m.startswith(expected_prefix)]

    if not model_members:
        raise RuntimeError(
            f"Verification failed: Zip archive does not contain files under '{expected_prefix}'"
        )

    # Required files
    has_config = any(m == f"{expected_prefix}config.json" for m in members)
    has_weights = any(
        m in (f"{expected_prefix}model.safetensors", f"{expected_prefix}pytorch_model.bin")
        for m in members
    )
    has_tokenizer = any(
        m in (f"{expected_prefix}tokenizer.json", f"{expected_prefix}tokenizer_config.json")
        for m in members
    )

    if not has_config:
        raise RuntimeError(f"Verification failed: Missing '{expected_prefix}config.json' in zip.")
    if not has_weights:
        raise RuntimeError(f"Verification failed: Missing weights (model.safetensors) in zip.")
    if not has_tokenizer:
        raise RuntimeError(f"Verification failed: Missing tokenizer files in zip.")

    return members


def build_models_archive() -> None:
    """Package the active model folder into release_assets/models.zip."""
    print("=" * 70)
    print("  BUILDING RELEASE ASSET: models.zip")
    print("=" * 70)

    active_model_name = ACTIVE_DISTILBERT
    active_model_dir = MODELS_DIR / active_model_name

    print(f"Active Model      : {active_model_name}")
    print(f"Source Directory  : {active_model_dir}")

    if not active_model_dir.is_dir():
        raise FileNotFoundError(
            f"Active model directory not found at: {active_model_dir}\n"
            f"Please ensure model weights exist before building the release archive."
        )

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Packaging files into {OUTPUT_ZIP.relative_to(PROJECT_ROOT)} ...")
    file_count = 0
    total_uncompressed = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(active_model_dir):
            # Prune checkpoint directories and smoke items in-place
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith("checkpoint-") and "smoke" not in d.lower()
            ]

            for file in sorted(files):
                if "smoke" in file.lower() or file.endswith(".pyc") or file == ".DS_Store":
                    continue

                abs_file = Path(root) / file
                # The archive name must preserve the model folder at the top level
                rel_inside_model = abs_file.relative_to(active_model_dir)
                arcname = str(Path(active_model_name) / rel_inside_model)

                zf.write(abs_file, arcname=arcname)
                file_count += 1
                total_uncompressed += abs_file.stat().st_size
                print(f"  + Added: {arcname} ({abs_file.stat().st_size / (1024*1024):.2f} MB)")

    zip_size_bytes = OUTPUT_ZIP.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)

    print(f"\n  ✓ Files packed        : {file_count}")
    print(f"  ✓ Uncompressed size   : {total_uncompressed / (1024 * 1024):.2f} MB")
    print(f"  ✓ Compressed size     : {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")

    # Size warning check
    if zip_size_bytes > MAX_RECOMMENDED_BYTES:
        print(f"\n  [!] WARNING: Archive size ({zip_size_mb:.2f} MB) exceeds recommended 1.9 GB limit!")
    else:
        print(f"  ✓ Size limit check    : OK (< 1.9 GB)")

    # 2. Compute SHA256 and update config.py
    print("\n[2/3] Computing SHA256 checksum and updating src/config.py ...")
    sha256_hash = compute_sha256(OUTPUT_ZIP)
    print(f"  ✓ SHA256 Checksum     : {sha256_hash}")

    update_config_sha256(CONFIG_FILE, sha256_hash)
    print(f"  ✓ Updated config      : {CONFIG_FILE.relative_to(PROJECT_ROOT)} (MODELS_SHA256 = \"{sha256_hash}\")")

    # 3. Re-open and verify contents
    print("\n[3/3] Re-opening and verifying zip integrity ...")
    members = verify_zip_contents(OUTPUT_ZIP, active_model_name)
    print(f"  ✓ Verification passed : Archive correctly contains model.safetensors, config.json, and tokenizer files.")
    print("  ✓ Members in archive  :")
    for m in members:
        print(f"      - {m}")

    print("\n" + "=" * 70)
    print("  MODEL RELEASE ZIP GENERATION COMPLETE")
    print("=" * 70)
    print(f"Folder in Zip : {active_model_name}/")
    print(f"Archive Size  : {zip_size_mb:.2f} MB")
    print(f"SHA256 Hash   : {sha256_hash}")
    print(f"Output File   : {OUTPUT_ZIP}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    build_models_archive()

