"""ONNX Runtime inference predictor — preferred production path for CPU."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.postprocess import detections_from_onnx
from src.inference.schemas import Detection
from src.settings import CONF_THRESHOLD, IOU_THRESHOLD, IMGSZ, ONNX_MODEL_PATH
from src.utils.logging import get_logger

log = get_logger(__name__)


class ONNXPredictor:
    """ONNX Runtime inference on CPU."""

    def __init__(
        self,
        model_path: Path = ONNX_MODEL_PATH,
        conf: float = CONF_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        imgsz: int = IMGSZ,
    ) -> None:
        import onnxruntime as ort

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        log.info("Loading ONNX model from %s ...", model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path),
            opts,
            providers=["CPUExecutionProvider"],
        )
        self._inp_name = self._session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        log.info("ONNX predictor ready (input: %s).", self._inp_name)

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        orig_size = (image.height, image.width)
        img = image.convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
        return arr, orig_size

    def predict(self, image: Image.Image) -> tuple[list[Detection], float]:
        """Run ONNX inference on a PIL Image. Returns (detections, inference_ms)."""
        inp, orig_shape = self._preprocess(image)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._inp_name: inp})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = detections_from_onnx(
            raw_output=outputs,
            orig_shape=orig_shape,
            input_shape=(self.imgsz, self.imgsz),
            conf_threshold=self.conf,
            iou_threshold=self.iou,
        )
        return detections, round(elapsed_ms, 2)
