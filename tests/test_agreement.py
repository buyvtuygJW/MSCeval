import pytest

from veridic_eval.agreement import (
    agreement_report,
    detector_agreement,
    raw_agreement,
)


def test_perfect_agreement():
    a = [1, 0, 1, 1, 0]
    assert raw_agreement(a, a) == pytest.approx(1.0)


def test_partial_agreement():
    a = [1, 1, 0, 0]
    b = [1, 0, 0, 1]
    assert raw_agreement(a, b) == pytest.approx(0.5)


def test_report_shape_and_none_skipping():
    a = [1, 0, None, 1]
    b = [1, 0, 1, None]
    rep = agreement_report(a, b, "auto", "human")
    assert rep["n"] == 2            # two None-containing pairs dropped
    assert rep["raw_agreement"] == pytest.approx(1.0)
    assert rep["label_a"] == "auto" and rep["label_b"] == "human"


def test_report_carries_no_kappa():
    # Single-annotator gold, so no inter-annotator statistic is computed anywhere.
    assert "cohen_kappa" not in agreement_report([1, 0], [1, 0])


def test_detector_agreement_over_per_query():
    a = {"q1": {"faithful": 1}, "q2": {"faithful": 0}, "q3": {"faithful": 1}}
    b = {"q1": {"faithful": 1}, "q2": {"faithful": 1}, "q3": {"faithful": 1}}
    rep = detector_agreement(a, b)
    assert rep["n"] == 3
    assert rep["raw_agreement"] == pytest.approx(2 / 3)


def test_empty_returns_none():
    assert raw_agreement([], []) is None
