"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# We patch model loading so tests work without a trained model on disk
with patch("src.api.services.load_model"):
    from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "model_backend" in data


class TestUI:
    def test_index_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "PCB Defect Inspector" in resp.text


class TestPredict:
    def _make_upload(self, width: int = 64, height: int = 64) -> bytes:
        img = Image.new("RGB", (width, height), color=(200, 180, 160))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.read()

    def test_predict_no_model_returns_503(self):
        """Without a model loaded, /predict should return 503."""
        import src.api.services as svc
        original = svc._predictor
        svc._predictor = None

        img_bytes = self._make_upload()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        svc._predictor = original
        assert resp.status_code == 503

    def test_predict_with_mock_model(self):
        """Mock the predictor and verify response schema."""
        from src.inference.schemas import BoardStatus, BoundingBox, Detection
        import src.api.services as svc

        mock_det = Detection(
            class_id=3,
            class_name="OP",
            class_full_name="Open",
            confidence=0.85,
            bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        )
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = ([mock_det], 42.0)

        original = svc._predictor
        svc._predictor = mock_predictor

        img_bytes = self._make_upload()
        resp = client.post(
            "/predict",
            files={"file": ("pcb.jpg", img_bytes, "image/jpeg")},
        )
        svc._predictor = original

        assert resp.status_code == 200
        data = resp.json()
        assert data["board_status"] == BoardStatus.REJECT.value
        assert data["total_defects"] == 1
        assert data["counts_by_class"]["OP"] == 1
        assert "REJECT" in data["natural_language_summary"]
        assert data["inference_time_ms"] == 42.0
        assert "annotated_image_b64" in data

    def test_predict_unsupported_media_type(self):
        resp = client.post(
            "/predict",
            files={"file": ("test.pdf", b"%PDF", "application/pdf")},
        )
        assert resp.status_code == 415
