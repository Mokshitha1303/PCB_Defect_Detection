#!/usr/bin/env bash
# setup.sh — Full project setup: install deps, download dataset, prepare
set -euo pipefail
pip install -r requirements.txt
python src/data/download_dataset.py
python src/data/prepare_dataset.py
python src/data/verify_dataset.py
echo "Setup complete. Run: make train"
