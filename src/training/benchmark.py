"""Benchmark PyTorch vs ONNX Runtime inference latency on CPU.

Usage:
    python src/training/benchmark.py [--pt PATH] [--onnx PATH] [--imgsz N] [--runs N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.settings import IMGSZ, ONNX_MODEL_PATH, PT_MODEL_PATH
from src.utils.logging import get_logger

log = get_logger(__name__)

WARMUP = 10


def _bench_pytorch(weights: Path, imgsz: int, runs: int) -> dict:
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(str(weights))
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(WARMUP):
        model(dummy, verbose=False)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(dummy, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    return _stats(latencies, "PyTorch")


def _bench_onnx(onnx_path: Path, imgsz: int, runs: int) -> dict:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    inp_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(WARMUP):
        session.run(None, {inp_name: dummy})

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {inp_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    return _stats(latencies, "ONNX Runtime")


def _stats(latencies: list[float], label: str) -> dict:
    arr = np.array(latencies)
    result = {
        "backend": label,
        "runs": len(latencies),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
    log.info(
        "%s — mean %.1f ms | p50 %.1f ms | p95 %.1f ms",
        label, result["mean_ms"], result["p50_ms"], result["p95_ms"],
    )
    return result


def benchmark(pt_path: Path, onnx_path: Path, imgsz: int, runs: int) -> list[dict]:
    results = []

    if pt_path.exists():
        log.info("Benchmarking PyTorch (%d runs, imgsz=%d) ...", runs, imgsz)
        results.append(_bench_pytorch(pt_path, imgsz, runs))
    else:
        log.warning("PyTorch weights not found: %s", pt_path)

    if onnx_path.exists():
        log.info("Benchmarking ONNX Runtime (%d runs, imgsz=%d) ...", runs, imgsz)
        results.append(_bench_onnx(onnx_path, imgsz, runs))
    else:
        log.warning("ONNX model not found: %s", onnx_path)

    if len(results) == 2:
        speedup = results[0]["mean_ms"] / results[1]["mean_ms"]
        log.info("ONNX Runtime is %.2fx %s than PyTorch on CPU", abs(speedup), "faster" if speedup > 1 else "slower")

    out_path = ROOT / "runs" / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Benchmark results saved -> %s", out_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime")
    parser.add_argument("--pt", type=Path, default=PT_MODEL_PATH)
    parser.add_argument("--onnx", type=Path, default=ONNX_MODEL_PATH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    benchmark(args.pt, args.onnx, args.imgsz, args.runs)


if __name__ == "__main__":
    main()
