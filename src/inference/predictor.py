"""PyTorch / Ultralytics inference predictor (fallback path)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.postprocess import detections_from_ultralytics
from src.inference.schemas import Detection
from src.settings import CONF_THRESHOLD, IOU_THRESHOLD, IMGSZ, PT_MODEL_PATH
from src.utils.logging import get_logger

log = get_logger(__name__)


class Predictor:
    """Wraps Ultralytics YOLO for PyTorch inference."""

    def __init__(
        self,
        weights: Path = PT_MODEL_PATH,
        conf: float = CONF_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        imgsz: int = IMGSZ,
    ) -> None:
        from ultralytics import YOLO

        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found: {weights}")

        log.info("Loading PyTorch model from %s ...", weights)
        self._model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        log.info("PyTorch predictor ready.")

    def predict(self, image: Image.Image) -> tuple[list[Detection], float]:
        """Run inference on a PIL Image. Returns (detections, inference_ms)."""
        img_arr = np.array(image)
        t0 = time.perf_counter()
        results = self._model(
            img_arr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        detections = detections_from_ultralytics(results)
        return detections, round(elapsed_ms, 2)
