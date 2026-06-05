# ── PCB Defect Inspector — Dockerfile ────────────────────────────────────────
# Multi-arch: linux/arm64 (Oracle A1) + linux/amd64
# CPU-only inference; no GPU required.

FROM python:3.11-slim

LABEL maintainer="PCB Defect Inspector"
LABEL description="Real-Time PCB Defect Inspection using YOLO26"

# ── System dependencies for OpenCV headless ───────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY src/       ./src/
COPY configs/   ./configs/

# ── Create runtime directories ────────────────────────────────────────────────
RUN mkdir -p runs data/samples

# ── Environment defaults (override with --env-file or -e) ────────────────────
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    CONF_THRESHOLD=0.25 \
    IOU_THRESHOLD=0.45 \
    IMGSZ=640

EXPOSE 8000

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
