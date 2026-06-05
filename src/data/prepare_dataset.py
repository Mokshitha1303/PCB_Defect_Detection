"""Prepare DsPCBSD+ dataset into Ultralytics YOLO layout and write dataset.yaml.

Expected output layout:
    data/processed/
        images/
            train/  *.jpg ...
            val/    *.jpg ...
        labels/
            train/  *.txt ...
            val/    *.txt ...

Usage:
    python src/data/prepare_dataset.py [--raw-dir DIR] [--processed-dir DIR]
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.settings import CLASS_NAMES, DATA_PROCESSED_DIR, DATA_RAW_DIR, DATASET_YAML
from src.utils.logging import get_logger

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VAL_RATIO = 0.20  # fallback split if no predefined split found
RANDOM_SEED = 42


def _find_dataset_root(raw_dir: Path) -> Path:
    for candidate in [raw_dir, *sorted(raw_dir.rglob("*"))]:
        if candidate.is_dir():
            if (candidate / "images").exists() and (candidate / "labels").exists():
                return candidate
    return raw_dir


def _collect_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    """Return list of (image_path, label_path) for all matched pairs."""
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        # Find corresponding label file (same stem, .txt)
        label = labels_dir / img.relative_to(images_dir).with_suffix(".txt")
        if not label.exists():
            # Try flat labels dir
            label = labels_dir / (img.stem + ".txt")
        if label.exists():
            pairs.append((img, label))
        else:
            log.debug("No label for %s — skipping", img.name)

    log.info("Found %d image-label pairs", len(pairs))
    return pairs


def _detect_existing_split(dataset_root: Path) -> tuple[list, list] | None:
    """Check if the dataset already has train/val sub-folders."""
    images_dir = dataset_root / "images"
    train_dir = images_dir / "train"
    val_dir = images_dir / "val"
    if train_dir.exists() and val_dir.exists():
        log.info("Pre-existing train/val split detected — using it.")
        return None  # signal to use existing layout directly
    return None


def _copy_pairs(
    pairs: list[tuple[Path, Path]],
    split: str,
    processed_dir: Path,
) -> None:
    img_out = processed_dir / "images" / split
    lbl_out = processed_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)


def prepare(raw_dir: Path, processed_dir: Path) -> Path:
    dataset_root = _find_dataset_root(raw_dir)
    log.info("Dataset root: %s", dataset_root)

    images_dir = dataset_root / "images"

    # Check for pre-existing train/val split
    has_split = (images_dir / "train").exists() and (images_dir / "val").exists()

    if has_split:
        log.info("Using existing train/val split from dataset.")
        train_imgs = list((images_dir / "train").rglob("*"))
        train_imgs = [p for p in train_imgs if p.suffix.lower() in IMAGE_EXTS]
        val_imgs = list((images_dir / "val").rglob("*"))
        val_imgs = [p for p in val_imgs if p.suffix.lower() in IMAGE_EXTS]
        log.info("Train: %d  Val: %d", len(train_imgs), len(val_imgs))

        # Symlink/copy dataset_root as-is if structure matches
        # Just write yaml pointing there
        yaml_data = {
            "path": str(dataset_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
    else:
        log.info("No split found — creating %.0f%%/%.0f%% train/val split.", (1 - VAL_RATIO) * 100, VAL_RATIO * 100)
        pairs = _collect_pairs(dataset_root)

        if not pairs:
            log.error("No image-label pairs found. Check dataset extraction.")
            sys.exit(1)

        random.seed(RANDOM_SEED)
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - VAL_RATIO))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        log.info("Copying train split (%d pairs) ...", len(train_pairs))
        _copy_pairs(train_pairs, "train", processed_dir)
        log.info("Copying val split (%d pairs) ...", len(val_pairs))
        _copy_pairs(val_pairs, "val", processed_dir)

        yaml_data = {
            "path": str(processed_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }

    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(yaml_data, fh, default_flow_style=False, allow_unicode=True)

    log.info("Wrote dataset config -> %s", DATASET_YAML)
    return DATASET_YAML


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DsPCBSD+ for Ultralytics training")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR)
    args = parser.parse_args()
    prepare(args.raw_dir, args.processed_dir)


if __name__ == "__main__":
    main()
