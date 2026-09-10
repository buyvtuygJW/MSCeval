"""CFCA delta = a ratio of two per-question means, so it gets a paired interval."""
import pytest

from veridic_eval.report import cfca_delta_block
from veridic_eval.stats import (
    paired_ratio_delta_ci,
    paired_ratio_permutation_p,
)

Q8 = [f"q{i}" for i in range(8)]


def _col(vals):
    return {q: v for q, v in zip(Q8, vals)}


# cell: one expensive question, 6 of 8 answers pass both gates
COST_A = _col([0.002, 0.002, 0.020, 0.002, 0.002, 0.002, 0.002, 0.002])
JOINT_A = _col([1, 1, 1, 1, 0, 1, 1, 0])
# reference: flat cost, 4 of 8 pass
COST_B = _col([0.004] * 8)
JOINT_B = _col([1, 0, 1, 0, 1, 0, 1, 0])


def test_identical_cells_give_a_zero_delta_that_is_not_significant():
    res = paired_ratio_delta_ci(COST_A, JOINT_A, dict(COST_A), dict(JOINT_A))
    assert res["delta"] == 0.0
    assert res["significant"] is False
    assert res["ci_low"] <= 0.0 <= res["ci_high"]


def test_delta_is_the_difference_of_two_ratios_not_a_mean_of_differences():
    res = paired_ratio_delta_ci(COST_A, JOINT_A, COST_B, JOINT_B)
    want = (sum(COST_A.values()) / sum(JOINT_A.values())
            - sum(COST_B.values()) / sum(JOINT_B.values()))
    assert res["delta"] == pytest.approx(want, abs=1e-8)
    assert res["n"] == 8
    assert res["method"] == "BCa"
    assert res["ci_low"] <= res["delta"] <= res["ci_high"]


def test_one_expensive_question_widens_the_interval():
    flat = _col([0.00425] * 8)          # same mean cost, no spread
    spread = paired_ratio_delta_ci(COST_A, JOINT_A, COST_B, JOINT_B)
    even = paired_ratio_delta_ci(flat, JOINT_A, COST_B, JOINT_B)
    assert spread["delta"] == pytest.approx(even["delta"], abs=1e-6)
    assert (spread["ci_high"] - spread["ci_low"]) > (even["ci_high"] - even["ci_low"])


def test_a_cell_that_never_passes_both_gates_reports_no_delta():
    res = paired_ratio_delta_ci(COST_A, _col([0] * 8), COST_B, JOINT_B)
    assert res["delta"] is None and res["method"] is None
    assert "undefined" in res["note"]


def test_flat_columns_collapse_to_a_degenerate_point_interval():
    cost = _col([0.001] * 8)
    ones = _col([1] * 8)
    res = paired_ratio_delta_ci(cost, ones, _col([0.003] * 8), dict(ones))
    assert res["method"] == "degenerate"
    assert res["ci_low"] == res["delta"] == res["ci_high"] == pytest.approx(-0.002)
    assert res["significant"] is True


def test_zero_denominator_policy_is_explicit():
    with pytest.raises(ValueError):
        paired_ratio_delta_ci(COST_A, JOINT_A, COST_B, JOINT_B,
                              on_zero_denominator="silence-it")


def test_one_paired_question_has_nothing_to_resample():
    res = paired_ratio_delta_ci({"q0": 0.1}, {"q0": 1}, {"q0": 0.2}, {"q0": 1})
    assert res["n"] == 1 and res["method"] == "degenerate"
    assert res["ci_low"] == res["ci_high"] == res["delta"]


def test_no_shared_question_reports_nothing():
    res = paired_ratio_delta_ci({"q0": 0.1}, {"q0": 1}, {"q9": 0.2}, {"q9": 1})
    assert res["n"] == 0 and res["delta"] is None


def test_permutation_p_is_exact_for_a_small_pairing():
    p = paired_ratio_permutation_p(COST_A, JOINT_A, COST_B, JOINT_B)
    assert 0.0 < p <= 1.0
    assert p * 2 ** 8 == pytest.approx(round(p * 2 ** 8))   # k / 2**n exactly


def test_identical_cells_cannot_be_told_apart():
    p = paired_ratio_permutation_p(COST_A, JOINT_A, dict(COST_A), dict(JOINT_A))
    assert p == 1.0


def test_permutation_p_needs_two_paired_questions():
    assert paired_ratio_permutation_p({"q0": 1.0}, {"q0": 1}, {"q0": 2.0}, {"q0": 1}) is None


def test_block_keeps_the_reported_scalars_and_adds_the_interval():
    cols = {"cfca_cost": COST_A, "cfca_joint_P": JOINT_A}
    ref = {"cfca_cost": COST_B, "cfca_joint_P": JOINT_B}
    block = cfca_delta_block(cols, ref, cfca=0.0057, ref_cfca=0.008, n_contrasts=3)
    assert block["cell"] == 0.0057 and block["reference"] == 0.008
    assert block["delta"] == pytest.approx(0.0057 - 0.008, abs=1e-8)
    assert block["ci_low"] <= block["ci_high"]
    assert block["method"] == "BCa"
    assert 0.0 < block["perm_p"] <= 1.0
    assert block["perm_p_bonferroni"] == pytest.approx(
        min(1.0, block["perm_p"] * 3), abs=1e-6)


def test_block_without_cost_columns_stays_a_plain_subtraction():
    block = cfca_delta_block({"cfca_joint_P": JOINT_A}, {"cfca_joint_P": JOINT_B},
                             cfca=0.01, ref_cfca=0.004)
    assert block["delta"] == pytest.approx(0.006, abs=1e-8)
    assert "ci_low" not in block
    assert "plain subtraction" in block["note"]
