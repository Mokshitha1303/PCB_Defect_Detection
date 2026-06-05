"""Verify dataset integrity: image count, label count, class distribution.

Usage:
    python src/data/verify_dataset.py [--raw-dir DIR]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.settings import CLASS_NAMES, DATA_RAW_DIR
from src.utils.logging import get_logger

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_dataset_root(raw_dir: Path) -> Path:
    """Return the folder that actually contains 'images' and 'labels' subdirs."""
    # Figshare zip may nest under DsPCBSD+/ or similar
    for candidate in [raw_dir, *raw_dir.iterdir()]:
        if candidate.is_dir():
            if (candidate / "images").exists() or (candidate / "labels").exists():
                return candidate
            # Check one more level
            for sub in candidate.iterdir():
                if sub.is_dir() and ((sub / "images").exists() or (sub / "labels").exists()):
                    return sub
    return raw_dir


def verify(raw_dir: Path) -> dict:
    dataset_root = _find_dataset_root(raw_dir)
    log.info("Dataset root detected: %s", dataset_root)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if not images_dir.exists():
        log.warning("No 'images' directory found under %s", dataset_root)
        # Try flat layout
        images_dir = dataset_root

    image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    label_files = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else []

    log.info("Images found : %d", len(image_files))
    log.info("Labels found : %d", len(label_files))

    # Class distribution
    class_counts: dict[int, int] = collections.defaultdict(int)
    corrupt_labels: list[Path] = []
    total_boxes = 0

    for lf in label_files:
        try:
            lines = lf.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                class_counts[cls_id] += 1
                total_boxes += 1
        except Exception as exc:
            log.warning("Corrupt label %s: %s", lf, exc)
            corrupt_labels.append(lf)

    log.info("Total bounding boxes : %d", total_boxes)
    log.info("Corrupt label files  : %d", len(corrupt_labels))

    log.info("Class distribution:")
    for cls_id in sorted(class_counts):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        log.info("  [%d] %-6s : %d", cls_id, name, class_counts[cls_id])

    return {
        "dataset_root": str(dataset_root),
        "num_images": len(image_files),
        "num_labels": len(label_files),
        "total_boxes": total_boxes,
        "class_counts": dict(class_counts),
        "corrupt_labels": len(corrupt_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DsPCBSD+ dataset")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    args = parser.parse_args()
    stats = verify(args.raw_dir)
    log.info("Verification complete: %s", stats)


if __name__ == "__main__":
    main()
