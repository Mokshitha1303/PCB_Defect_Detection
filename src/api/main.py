"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.api.services import load_model
from src.utils.logging import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting PCB Defect Inspector ...")
    load_model()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="PCB Defect Inspector",
    description=(
        "Real-Time PCB Defect Inspection at the Edge using YOLO26. "
        "Detects and localises 9 categories of PCB surface defects "
        "and returns a PASS / REVIEW / REJECT board decision."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(ROOT / "src" / "ui" / "static")),
    name="static",
)

app.include_router(router)
