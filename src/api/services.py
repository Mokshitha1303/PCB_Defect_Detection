"""Model loading and inference orchestration for the API."""

from __future__ import annotations

from pathlib import Path

from PIL.Image import Image

from src.inference.decision_logic import decide
from src.inference.schemas import Detection, PredictResponse
from src.settings import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    IMGSZ,
    ONNX_MODEL_PATH,
    PT_MODEL_PATH,
)
from src.utils.image_ops import annotate_image, image_to_base64
from src.utils.logging import get_logger

log = get_logger(__name__)

_predictor = None
_model_backend: str = "none"


def load_model() -> None:
    """Load the best available model: ONNX first, PyTorch fallback."""
    global _predictor, _model_backend

    if ONNX_MODEL_PATH.exists():
        try:
            from src.inference.onnx_predictor import ONNXPredictor
            _predictor = ONNXPredictor(ONNX_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, IMGSZ)
            _model_backend = "onnx"
            log.info("Model backend: ONNX Runtime")
            return
        except Exception as exc:
            log.warning("ONNX load failed (%s) — falling back to PyTorch.", exc)

    if PT_MODEL_PATH.exists():
        from src.inference.predictor import Predictor
        _predictor = Predictor(PT_MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, IMGSZ)
        _model_backend = "pytorch"
        log.info("Model backend: PyTorch")
        return

    log.warning(
        "No model found at %s or %s. "
        "Run `make train && make export` first.",
        ONNX_MODEL_PATH, PT_MODEL_PATH,
    )
    _model_backend = "none"


def is_model_loaded() -> bool:
    return _predictor is not None


def get_model_backend() -> str:
    return _model_backend


def run_inference(image: Image, filename: str) -> PredictResponse:
    if _predictor is None:
        raise RuntimeError("Model is not loaded. Run `make train && make export` first.")

    detections: list[Detection]
    detections, inf_ms = _predictor.predict(image)

    status, summary, counts, max_conf = decide(detections, CONF_THRESHOLD)
    annotated = annotate_image(image, detections)
    ann_b64 = image_to_base64(annotated)

    return PredictResponse(
        filename=filename,
        image_width=image.width,
        image_height=image.height,
        board_status=status,
        natural_language_summary=summary,
        detections=detections,
        counts_by_class=counts,
        max_conf_by_class=max_conf,
        total_defects=len([d for d in detections if d.confidence >= CONF_THRESHOLD]),
        inference_time_ms=inf_ms,
        annotated_image_b64=ann_b64,
    )
