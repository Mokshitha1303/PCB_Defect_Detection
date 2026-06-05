"""FastAPI route definitions."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.api.services import get_model_backend, is_model_loaded, run_inference
from src.inference.schemas import (
    BatchPredictItem,
    BatchPredictResponse,
    HealthResponse,
    PredictResponse,
)
from src.utils.image_ops import load_image
from src.utils.logging import get_logger
from src.utils.metrics import export_batch_csv

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(ROOT / "src" / "ui" / "templates"))

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
MAX_UPLOAD_MB = 20


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        model_backend=get_model_backend(),
    )


# ── UI ────────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse, tags=["ui"])
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "model_loaded": is_model_loaded(),
            "model_backend": get_model_backend(),
        },
    )


# ── Single image inference ────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    _validate_upload(file)
    raw = await file.read()
    image = load_image(raw)
    try:
        return run_inference(image, filename=file.filename or "upload.jpg")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Batch inference ───────────────────────────────────────────────────────────

@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results: list[BatchPredictItem] = []

    for upload in files:
        _validate_upload(upload)
        raw = await upload.read()
        try:
            image = load_image(raw)
            resp = run_inference(image, filename=upload.filename or "upload.jpg")
            results.append(
                BatchPredictItem(
                    filename=resp.filename,
                    board_status=resp.board_status,
                    total_defects=resp.total_defects,
                    counts_by_class=resp.counts_by_class,
                    natural_language_summary=resp.natural_language_summary,
                    inference_time_ms=resp.inference_time_ms,
                )
            )
        except Exception as exc:
            log.warning("Failed to process %s: %s", upload.filename, exc)

    # Export CSV
    csv_path = ROOT / "runs" / "batch_results.csv"
    export_batch_csv(results, csv_path)

    return BatchPredictResponse(
        total_images=len(results),
        results=results,
        csv_download_path=str(csv_path),
    )


def _validate_upload(file: UploadFile) -> None:
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use JPEG, PNG, BMP, or TIFF.",
        )
