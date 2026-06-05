"""Central settings loaded from environment variables / .env file."""

from __future__ import annotations

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── App ───────────────────────────────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "development")
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Model ─────────────────────────────────────────────────────────────────────
ONNX_MODEL_PATH: Path = ROOT / os.getenv(
    "ONNX_MODEL_PATH", "runs/detect/runs/train/yolo26n_pcb/weights/best.onnx"
)
PT_MODEL_PATH: Path = ROOT / os.getenv(
    "PT_MODEL_PATH", "runs/detect/runs/train/yolo26n_pcb/weights/best.pt"
)
CONF_THRESHOLD: float = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
IMGSZ: int = int(os.getenv("IMGSZ", "640"))

# ── Class names ───────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "SH",   # Short
    "SP",   # Spur
    "SC",   # Spurious copper
    "OP",   # Open
    "MB",   # Mouse bite
    "HB",   # Hole breakout
    "CS",   # Conductor scratch
    "CFO",  # Conductor foreign object
    "BMFO", # Base material foreign object
]

CLASS_FULL_NAMES: dict[str, str] = {
    "SH": "Short",
    "SP": "Spur",
    "SC": "Spurious Copper",
    "OP": "Open",
    "MB": "Mouse Bite",
    "HB": "Hole Breakout",
    "CS": "Conductor Scratch",
    "CFO": "Conductor Foreign Object",
    "BMFO": "Base Material Foreign Object",
}

# ── Decision logic ────────────────────────────────────────────────────────────
_critical_raw: str = os.getenv("CRITICAL_CLASSES", "OP,SH")
CRITICAL_CLASSES: set[str] = {c.strip() for c in _critical_raw.split(",") if c.strip()}
REVIEW_MIN_DEFECTS: int = int(os.getenv("REVIEW_MIN_DEFECTS", "1"))

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_URL: str = os.getenv(
    "DATASET_URL", "https://figshare.com/ndownloader/files/44069552"
)
DATA_RAW_DIR: Path = ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR: Path = ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATASET_YAML: Path = ROOT / "configs" / "dataset.yaml"
