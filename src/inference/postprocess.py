"""Convert raw YOLO/ONNX outputs to Detection objects."""

from __future__ import annotations

import numpy as np

from src.inference.schemas import BoundingBox, Detection
from src.settings import CLASS_FULL_NAMES, CLASS_NAMES


def detections_from_ultralytics(results) -> list[Detection]:
    """Convert an Ultralytics Results object to a list of Detection."""
    detections: list[Detection] = []
    if results is None:
        return detections

    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    class_full_name=CLASS_FULL_NAMES.get(cls_name, cls_name),
                    confidence=round(conf, 4),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
    return detections


def detections_from_onnx(
    raw_output: np.ndarray,
    orig_shape: tuple[int, int],
    input_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """
    Parse raw ONNX output from a YOLO model.

    YOLO ONNX output shape: [1, num_classes+4, num_anchors]
    """
    # raw_output: (1, 4+nc, num_preds) or (1, num_preds, 4+nc)
    out = raw_output[0]  # remove batch dim

    # Transpose if needed: expect shape (num_preds, 4+nc)
    if out.shape[0] < out.shape[-1]:
        out = out.T  # (4+nc, N) -> (N, 4+nc)

    num_classes = out.shape[1] - 4
    boxes_raw = out[:, :4]   # cx, cy, w, h (normalised)
    scores = out[:, 4:]      # class scores

    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    keep = confs >= conf_threshold
    boxes_raw = boxes_raw[keep]
    class_ids = class_ids[keep]
    confs = confs[keep]

    if len(confs) == 0:
        return []

    # Convert cx,cy,w,h -> x1,y1,x2,y2 (normalised)
    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Scale to original image pixels
    ih, iw = orig_shape
    inp_h, inp_w = input_shape
    scale_x = iw / inp_w
    scale_y = ih / inp_h
    x1 = (x1 * inp_w * scale_x).clip(0, iw)
    y1 = (y1 * inp_h * scale_y).clip(0, ih)
    x2 = (x2 * inp_w * scale_x).clip(0, iw)
    y2 = (y2 * inp_h * scale_y).clip(0, ih)

    # NMS
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    indices = _nms(boxes_xyxy, confs, iou_threshold)

    detections: list[Detection] = []
    for i in indices:
        cls_id = int(class_ids[i])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        detections.append(
            Detection(
                class_id=cls_id,
                class_name=cls_name,
                class_full_name=CLASS_FULL_NAMES.get(cls_name, cls_name),
                confidence=round(float(confs[i]), 4),
                bbox=BoundingBox(
                    x1=float(x1[i]), y1=float(y1[i]),
                    x2=float(x2[i]), y2=float(y2[i]),
                ),
            )
        )
    return detections


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Vectorised NMS. Returns indices to keep."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]

    return keep
