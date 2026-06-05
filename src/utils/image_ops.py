"""Image annotation utilities — draw bounding boxes on PIL images."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from src.inference.schemas import Detection
from src.utils.logging import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
APP_YAML = ROOT / "configs" / "app.yaml"

_DEFAULT_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "SH":   (220,  50,  50),
    "SP":   (255, 165,   0),
    "SC":   (255, 200,   0),
    "OP":   (180,   0,   0),
    "MB":   ( 50, 180,  50),
    "HB":   ( 50, 200, 200),
    "CS":   (150,  50, 255),
    "CFO":  ( 30, 100, 255),
    "BMFO": (200, 100,  50),
}

_DEFAULT_COLOR = (128, 128, 128)


def _load_color_map() -> dict[str, tuple[int, int, int]]:
    if APP_YAML.exists():
        with open(APP_YAML, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        raw = cfg.get("annotator", {}).get("color_map", {})
        return {k: tuple(v) for k, v in raw.items()} or _DEFAULT_COLOR_MAP
    return _DEFAULT_COLOR_MAP


def annotate_image(
    image: Image.Image,
    detections: list[Detection],
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> Image.Image:
    """Return a copy of the image with bounding boxes and labels drawn."""
    color_map = _load_color_map()
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, int(img.width * font_scale * 0.02)))
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        color = color_map.get(det.class_name, _DEFAULT_COLOR)
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 40), outline=(*color, 255), width=line_thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        # Label background
        bbox_text = draw.textbbox((x1, y1 - 14), label, font=font)
        draw.rectangle(bbox_text, fill=(*color, 200))
        draw.text((x1, y1 - 14), label, fill=(255, 255, 255), font=font)

    return img


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_image(source) -> Image.Image:
    """Load a PIL Image from a file path, bytes, or BytesIO."""
    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    return Image.open(source).convert("RGB")
