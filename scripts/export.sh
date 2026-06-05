#!/usr/bin/env bash
set -euo pipefail
python src/training/export_onnx.py "$@"
