"""Board-level PASS / REVIEW / REJECT decision logic.

Rules (all configurable via settings):
  REJECT  — any detection whose class is in CRITICAL_CLASSES and confidence >= threshold
  REVIEW  — non-critical detections exist (count >= REVIEW_MIN_DEFECTS)
  PASS    — no detections above threshold
"""

from __future__ import annotations

import collections

from src.inference.schemas import BoardStatus, Detection
from src.settings import CLASS_FULL_NAMES, CLASS_NAMES, CONF_THRESHOLD, CRITICAL_CLASSES, REVIEW_MIN_DEFECTS


def decide(
    detections: list[Detection],
    conf_threshold: float = CONF_THRESHOLD,
    critical_classes: set[str] | None = None,
    review_min_defects: int = REVIEW_MIN_DEFECTS,
) -> tuple[BoardStatus, str, dict[str, int], dict[str, float]]:
    """Return (board_status, summary_text, counts_by_class, max_conf_by_class)."""
    if critical_classes is None:
        critical_classes = CRITICAL_CLASSES

    above_threshold = [d for d in detections if d.confidence >= conf_threshold]

    counts: dict[str, int] = collections.Counter(d.class_name for d in above_threshold)
    max_conf: dict[str, float] = {}
    for d in above_threshold:
        max_conf[d.class_name] = max(max_conf.get(d.class_name, 0.0), d.confidence)

    total = len(above_threshold)

    # Determine status
    has_critical = any(d.class_name in critical_classes for d in above_threshold)

    if has_critical:
        status = BoardStatus.REJECT
    elif total >= review_min_defects:
        status = BoardStatus.REVIEW
    else:
        status = BoardStatus.PASS

    # Natural language summary
    summary = _build_summary(status, counts)

    return status, summary, dict(counts), max_conf


def _build_summary(status: BoardStatus, counts: dict[str, int]) -> str:
    if not counts:
        return "No defects detected. Board status: PASS."

    parts = []
    for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        full = CLASS_FULL_NAMES.get(cls_name, cls_name)
        parts.append(f"{cnt} {full}" if cnt == 1 else f"{cnt} {full}s")

    total = sum(counts.values())
    defect_list = ", ".join(parts)
    return f"Detected {total} defect{'s' if total > 1 else ''}: {defect_list}. Board status: {status.value}."
