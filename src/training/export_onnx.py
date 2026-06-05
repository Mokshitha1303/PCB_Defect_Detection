"""Export trained YOLO26 .pt weights to ONNX for CPU deployment.

Usage:
    python src/training/export_onnx.py [--weights PATH] [--imgsz N] [--opset N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.settings import IMGSZ, PT_MODEL_PATH
from src.utils.logging import get_logger

log = get_logger(__name__)


def export_onnx(weights: Path, imgsz: int, opset: int, simplify: bool) -> Path:
    from ultralytics import YOLO

    if not weights.exists():
        log.error("Weights not found: %s", weights)
        sys.exit(1)

    model = YOLO(str(weights))
    log.info("Exporting %s to ONNX (opset=%d, imgsz=%d, simplify=%s) ...", weights.name, opset, imgsz, simplify)

    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,
        half=False,  # keep FP32 for CPU compatibility
    )

    onnx_path = Path(export_path)
    log.info("ONNX model saved -> %s  (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1_048_576)
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO26 best.pt -> best.onnx")
    parser.add_argument("--weights", type=Path, default=PT_MODEL_PATH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", default=True)
    args = parser.parse_args()

    export_onnx(args.weights, args.imgsz, args.opset, args.simplify)


if __name__ == "__main__":
    main()
