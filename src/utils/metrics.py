"""Helper utilities for metrics aggregation and CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.inference.schemas import BatchPredictItem


def export_batch_csv(results: list[BatchPredictItem], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "board_status", "total_defects", "inference_time_ms", "summary"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                "filename": item.filename,
                "board_status": item.board_status.value,
                "total_defects": item.total_defects,
                "inference_time_ms": item.inference_time_ms,
                "summary": item.natural_language_summary,
            })
    return out_path


def export_batch_json(results: list[BatchPredictItem], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([r.model_dump() for r in results], fh, indent=2, default=str)
    return out_path
