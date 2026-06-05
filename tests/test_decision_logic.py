"""Unit tests for board decision logic."""

import pytest

from src.inference.decision_logic import decide
from src.inference.schemas import BoardStatus, BoundingBox, Detection


def _det(cls_name: str, conf: float) -> Detection:
    return Detection(
        class_id=0,
        class_name=cls_name,
        class_full_name=cls_name,
        confidence=conf,
        bbox=BoundingBox(x1=10, y1=10, x2=100, y2=100),
    )


THRESH = 0.25
CRITICAL = {"OP", "SH"}


class TestPass:
    def test_no_detections(self):
        status, summary, counts, _ = decide([], THRESH, CRITICAL)
        assert status == BoardStatus.PASS
        assert counts == {}
        assert "PASS" in summary

    def test_all_below_threshold(self):
        dets = [_det("SP", 0.10), _det("SC", 0.15)]
        status, _, counts, _ = decide(dets, THRESH, CRITICAL)
        assert status == BoardStatus.PASS
        assert counts == {}


class TestReview:
    def test_non_critical_defect(self):
        dets = [_det("SP", 0.50)]
        status, summary, counts, _ = decide(dets, THRESH, CRITICAL)
        assert status == BoardStatus.REVIEW
        assert counts["SP"] == 1
        assert "REVIEW" in summary

    def test_multiple_non_critical(self):
        dets = [_det("CS", 0.30), _det("MB", 0.60), _det("HB", 0.40)]
        status, _, counts, _ = decide(dets, THRESH, CRITICAL)
        assert status == BoardStatus.REVIEW
        assert sum(counts.values()) == 3


class TestReject:
    def test_open_defect_triggers_reject(self):
        dets = [_det("OP", 0.80)]
        status, summary, _, _ = decide(dets, THRESH, CRITICAL)
        assert status == BoardStatus.REJECT
        assert "REJECT" in summary

    def test_short_defect_triggers_reject(self):
        dets = [_det("SH", 0.50)]
        status, _, _, _ = decide(dets, THRESH, CRITICAL)
        assert status == BoardStatus.REJECT

    def test_critical_mixed_with_non_critical(self):
        dets = [_det("OP", 0.90), _det("SP", 0.70)]
        status, _, counts, _ = decide(dets, THRESH, CRITICAL)
        assert status == BoardStatus.REJECT
        assert counts["OP"] == 1
        assert counts["SP"] == 1


class TestCountsAndMaxConf:
    def test_counts_by_class(self):
        dets = [_det("SP", 0.5), _det("SP", 0.7), _det("MB", 0.4)]
        _, _, counts, max_conf = decide(dets, THRESH, CRITICAL)
        assert counts["SP"] == 2
        assert counts["MB"] == 1
        assert abs(max_conf["SP"] - 0.7) < 1e-6

    def test_max_conf_per_class(self):
        dets = [_det("CS", 0.3), _det("CS", 0.9)]
        _, _, _, max_conf = decide(dets, THRESH, CRITICAL)
        assert abs(max_conf["CS"] - 0.9) < 1e-6
