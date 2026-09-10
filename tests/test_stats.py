import pytest

from veridic_eval.stats import (
    bonferroni,
    bootstrap_ci,
    paired_delta_ci,
    paired_permutation_p,
)


def test_bootstrap_ci_basic():
    r = bootstrap_ci([1, 1, 1, 0, 0], n_resamples=999, seed=42)
    assert r["n"] == 5
    assert r["mean"] == pytest.approx(0.6)
    assert 0.0 <= r["ci_low"] <= r["mean"] <= r["ci_high"] <= 1.0
    assert r["method"] == "BCa"


def test_bootstrap_ci_degenerate_constant():
    r = bootstrap_ci([1, 1, 1])
    assert r["method"] == "degenerate"
    assert r["ci_low"] == r["ci_high"] == 1.0


def test_bootstrap_ci_empty():
    r = bootstrap_ci([])
    assert r["n"] == 0 and r["mean"] is None


def test_paired_delta_ci_positive():
    a = {"q1": 1, "q2": 1, "q3": 1, "q4": 0}
    b = {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
    r = paired_delta_ci(a, b, n_resamples=999, seed=42)
    assert r["n"] == 4
    assert r["delta"] == pytest.approx(0.75)
    assert r["ci_low"] <= r["delta"] <= r["ci_high"]


def test_paired_delta_ci_only_common_keys():
    a = {"q1": 1, "q2": 1, "zzz": 1}
    b = {"q1": 0, "q2": 0}
    r = paired_delta_ci(a, b, n_resamples=499, seed=1)
    assert r["n"] == 2  # only q1,q2 shared


def test_paired_permutation_p_detects_difference():
    a = {f"q{i}": 1 for i in range(8)}
    b = {f"q{i}": 0 for i in range(8)}
    p = paired_permutation_p(a, b, n_resamples=999, seed=42)
    assert p is not None and 0.0 <= p <= 1.0


def test_paired_permutation_p_constant_diff_is_none():
    a = {"q1": 1, "q2": 1}
    b = {"q1": 1, "q2": 1}
    assert paired_permutation_p(a, b) is None


def test_bonferroni():
    assert bonferroni(0.02, 3) == pytest.approx(0.06)
    assert bonferroni(0.5, 4) == 1.0     # capped at 1.0
    assert bonferroni(None, 3) is None
