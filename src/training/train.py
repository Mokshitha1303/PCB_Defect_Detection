"""Train YOLO26 on the DsPCBSD+ dataset.

Usage:
    python src/training/train.py [--config configs/train.yaml] [--epochs N] [--batch N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.settings import DATASET_YAML
from src.utils.logging import get_logger

log = get_logger(__name__)


def _load_train_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train(config_path: Path, overrides: dict) -> None:
    from ultralytics import YOLO

    cfg = _load_train_config(config_path)
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    # Auto-detect device
    if "device" not in cfg or cfg.get("device") is None:
        cfg["device"] = "0" if torch.cuda.is_available() else "cpu"

    log.info("Device: %s", cfg["device"])
    log.info("Model : %s", cfg.get("model", "yolo26n.pt"))
    log.info("Epochs: %s  Batch: %s  Imgsz: %s", cfg.get("epochs"), cfg.get("batch"), cfg.get("imgsz"))

    if not DATASET_YAML.exists():
        log.error("dataset.yaml not found at %s. Run `make prepare` first.", DATASET_YAML)
        sys.exit(1)

    model = YOLO(cfg.pop("model", "yolo26n.pt"))

    results = model.train(
        data=str(DATASET_YAML),
        **cfg,
    )

    log.info("Training complete.")
    log.info("Best weights: %s", results.save_dir / "weights" / "best.pt")
    return results


def main() -> None:
    default_cfg = ROOT / "configs" / "train.yaml"

    parser = argparse.ArgumentParser(description="Train YOLO26 on DsPCBSD+")
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda device index or 'cpu'")
    parser.add_argument("--name", type=str, default=None, help="run name override")
    args = parser.parse_args()

    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "name": args.name,
    }

    train(args.config, overrides)


if __name__ == "__main__":
    main()
