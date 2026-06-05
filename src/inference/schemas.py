"""Pydantic schemas for inference request/response."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BoardStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left pixel coordinate")
    y1: float = Field(..., description="Top pixel coordinate")
    x2: float = Field(..., description="Right pixel coordinate")
    y2: float = Field(..., description="Bottom pixel coordinate")


class Detection(BaseModel):
    class_id: int = Field(..., description="Integer class index")
    class_name: str = Field(..., description="Short defect code, e.g. 'OP'")
    class_full_name: str = Field(..., description="Human-readable name, e.g. 'Open'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class PredictResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    board_status: BoardStatus
    natural_language_summary: str
    detections: list[Detection]
    counts_by_class: dict[str, int]
    max_conf_by_class: dict[str, float]
    total_defects: int
    inference_time_ms: float
    annotated_image_b64: Optional[str] = Field(
        None, description="Base64-encoded PNG of the annotated image"
    )


class BatchPredictItem(BaseModel):
    filename: str
    board_status: BoardStatus
    total_defects: int
    counts_by_class: dict[str, int]
    natural_language_summary: str
    inference_time_ms: float


class BatchPredictResponse(BaseModel):
    total_images: int
    results: list[BatchPredictItem]
    csv_download_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    version: str = "1.0.0"
