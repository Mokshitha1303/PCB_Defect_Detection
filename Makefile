.PHONY: install download prepare train eval export benchmark serve test docker-build docker-run clean

PYTHON := python
SRC    := src

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Data ──────────────────────────────────────────────────────
download:
	$(PYTHON) $(SRC)/data/download_dataset.py

verify:
	$(PYTHON) $(SRC)/data/verify_dataset.py

prepare:
	$(PYTHON) $(SRC)/data/prepare_dataset.py

# ── Training ──────────────────────────────────────────────────
train:
	$(PYTHON) $(SRC)/training/train.py

eval:
	$(PYTHON) $(SRC)/training/evaluate.py

export:
	$(PYTHON) $(SRC)/training/export_onnx.py

benchmark:
	$(PYTHON) $(SRC)/training/benchmark.py

# ── App ───────────────────────────────────────────────────────
serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

serve-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Tests ─────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── Docker ────────────────────────────────────────────────────
docker-build:
	docker build -t pcb-defect-yolo26:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env pcb-defect-yolo26:latest

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
