"""Evaluate a trained YOLO26 model on the validation set.

Usage:
    python src/training/evaluate.py [--weights PATH] [--imgsz N] [--conf F]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.settings import CLASS_NAMES, CONF_THRESHOLD, DATASET_YAML, IMGSZ, PT_MODEL_PATH
from src.utils.logging import get_logger

log = get_logger(__name__)


def evaluate(weights: Path, imgsz: int, conf: float, save_dir: Path | None = None) -> dict:
    from ultralytics import YOLO

    if not weights.exists():
        log.error("Weights not found: %s", weights)
        sys.exit(1)

    if not DATASET_YAML.exists():
        log.error("dataset.yaml not found: %s", DATASET_YAML)
        sys.exit(1)

    model = YOLO(str(weights))
    log.info("Evaluating %s on val set ...", weights.name)

    results = model.val(
        data=str(DATASET_YAML),
        imgsz=imgsz,
        conf=conf,
        save_json=True,
        plots=True,
        verbose=True,
        project=str(save_dir.parent) if save_dir else "runs/eval",
        name=save_dir.name if save_dir else "yolo26n_pcb_eval",
    )

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    # Per-class AP if available
    if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
        per_class: dict[str, float] = {}
        for idx, ap in zip(results.box.ap_class_index, results.box.ap50):
            name = CLASS_NAMES[int(idx)] if int(idx) < len(CLASS_NAMES) else f"cls_{idx}"
            per_class[name] = float(ap)
        metrics["per_class_ap50"] = per_class

    log.info("mAP50    : %.4f", metrics["mAP50"])
    log.info("mAP50-95 : %.4f", metrics["mAP50_95"])
    log.info("Precision: %.4f", metrics["precision"])
    log.info("Recall   : %.4f", metrics["recall"])

    if "per_class_ap50" in metrics:
        log.info("Per-class AP50:")
        for cls_name, ap in metrics["per_class_ap50"].items():
            log.info("  %-6s : %.4f", cls_name, ap)

    # Save summary JSON
    out_dir = Path(results.save_dir) if hasattr(results, "save_dir") else ROOT / "runs" / "eval"
    summary_path = out_dir / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Saved eval summary -> %s", summary_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 model")
    parser.add_argument("--weights", type=Path, default=PT_MODEL_PATH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    evaluate(args.weights, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
