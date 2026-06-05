"""Download and extract the DsPCBSD+ dataset from figshare.

Usage:
    python src/data/download_dataset.py [--url URL] [--dest DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.settings import DATA_RAW_DIR

# figshare.com/ndownloader returns HTTP 202 (queue); use ndownloader.figshare.com directly
DATASET_URL = "https://ndownloader.figshare.com/files/44069552"
from src.utils.logging import get_logger

log = get_logger(__name__)

ARCHIVE_NAME = "DsPCBSD+.zip"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        print(
            f"\r  Downloading ... {pct:5.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)",
            end="",
            flush=True,
        )


def download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / ARCHIVE_NAME

    if archive_path.exists():
        log.info("Archive already exists at %s — skipping download.", archive_path)
        return archive_path

    log.info("Downloading dataset from %s ...", url)
    urllib.request.urlretrieve(url, archive_path, reporthook=_progress_hook)
    print()  # newline after progress
    log.info("Saved to %s  (%.1f MB)", archive_path, archive_path.stat().st_size / 1_048_576)
    return archive_path


def extract(archive_path: Path, dest_dir: Path) -> Path:
    extract_marker = dest_dir / ".extracted"
    if extract_marker.exists():
        log.info("Dataset already extracted — skipping.")
        return dest_dir

    log.info("Extracting %s ...", archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest_dir)

    extract_marker.touch()
    log.info("Extraction complete -> %s", dest_dir)
    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DsPCBSD+ dataset")
    parser.add_argument("--url", default=DATASET_URL, help="Direct download URL")
    parser.add_argument(
        "--dest", type=Path, default=DATA_RAW_DIR, help="Destination directory"
    )
    args = parser.parse_args()

    archive = download(args.url, args.dest)
    extract(archive, args.dest)
    log.info("Dataset ready at %s", args.dest)


if __name__ == "__main__":
    main()
