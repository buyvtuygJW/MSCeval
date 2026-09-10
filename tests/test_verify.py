"""
Math checks on the checker: every failure mode `verify_report` claims to catch
is built here by hand and asserted by check name, so a silent regression in the
Bonferroni divisor, the CI ordering, the significant flag, the CFCA
subtraction or the off-grid isolation fails this file instead of shipping.

Imports stay on cells/verify/stats so nothing pulls the model or DB chain in.
"""
import copy
import json

from veridic_eval.cells import (
    CHUNK_OPT,
    COMBINED,
    CONTROL,
    DELTAS_KEY,
    GRID_ORDER,
    OFFGRID_DELTAS_KEY,
    PREBASELINE,
    QDORA,
    contrast_key,
)
from veridic_eval.verify import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    render_verify_text,
    verify_report,
    verify_report_file,
)

ALPHA = 0.05


# ---------------------------------------------------------------- fixtures
def _delta(**over):
    """A clean paired-delta block: p=0.01 corrected by the grid family of 3."""
    block = {
        "n": 10,
        "delta": 0.2,
        "ci_low": 0.05,
        "ci_high": 0.35,
        "significant": True,
        "method": "BCa",
        "perm_p": 0.01,
        "perm_p_bonferroni": 0.03,
    }
    block.update(over)
    return block


def _offgrid_delta(**over):
    """The before/after block: same p, no correction applied."""
    block = _delta(
        delta=-0.3, ci_low=-0.45, ci_high=-0.15, perm_p_bonferroni=0.01
    )
    block.update(over)
    return block


def _cell(**over):
    cell = {
        "n_queries": 10,
        "confidence_intervals": {
            "faithful": {
                "n": 10,
                "mean": 0.6,
                "ci_low": 0.4,
                "ci_high": 0.8,
                "method": "BCa",
            }
        },
    }
    cell.update(over)
    return cell


def _report():
    """A report that every check must pass, with nothing to warn about."""
    return {
        "reference": CONTROL,
        "grid": list(GRID_ORDER),
        "offgrid": [PREBASELINE],
        "n_contrasts": 3,
        "offgrid_n_contrasts": 1,
        "cells": {name: _cell() for name in list(GRID_ORDER) + [PREBASELINE]},
        DELTAS_KEY: {
            contrast_key(CHUNK_OPT, CONTROL): {
                "faithful": _delta(),
                "cfca": {"cell": 1.0, "reference": 1.5, "delta": -0.5},
            },
            contrast_key(QDORA, CONTROL): {"faithful": _delta()},
            contrast_key(COMBINED, CONTROL): {"faithful": _delta()},
        },
        OFFGRID_DELTAS_KEY: {
            contrast_key(PREBASELINE, CONTROL): {"faithful": _offgrid_delta()}
        },
    }


def _check(report, **kwargs):
    kwargs.setdefault("alpha", ALPHA)
    return verify_report(report, **kwargs)


def _statuses(result, check):
    return [r["status"] for r in result["checks"] if r["check"] == check]


def _failed(result):
    return [r["check"] for r in result["failed"]]


def _warned(result):
    return [r["check"] for r in result["warned"]]


def _detail(result, check):
    return " ".join(
        r["detail"] for r in result["checks"] if r["check"] == check and r["detail"]
    )


# ---------------------------------------------------------------- clean path
def test_clean_report_has_no_failure_and_no_warning():
    result = _check(_report())
    assert result["ok"] is True
    assert _failed(result) == []
    assert _warned(result) == []
    assert result["counts"][FAIL] == 0
    assert result["counts"][PASS] > 0
    assert result["alpha"] == ALPHA
    assert result["n_contrasts"] == 3
    assert result["offgrid_n_contrasts"] == 1


def test_clean_report_actually_ran_the_checks_that_matter():
    result = _check(_report())
    ran = {r["check"] for r in result["checks"]}
    for check in (
        "family.grid_size",
        "family.offgrid_size",
        "coverage.contrast_present",
        "coverage.no_extra_contrast",
        "bonferroni.identity",
        "bonferroni.monotone",
        "bonferroni.offgrid_uncorrected",
        "ci.order",
        "ci.low_le_high",
        "significance.flag_identity",
        "decision.ci_vs_p",
        "cfca.delta_identity",
        "pairing.n",
        "bounds.delta",
        "cell.ci_order",
        "cell.bounds",
        "isolation.not_in_grid_deltas",
        "isolation.family_unchanged_by_offgrid",
    ):
        assert check in ran, check
    assert set(_statuses(result, "bonferroni.identity")) == {PASS}
    assert set(_statuses(result, "cfca.delta_identity")) == {PASS}


def test_clean_report_renders_as_passing_text():
    text = render_verify_text(_check(_report()))
    assert "report math check:" in text
    assert "every check passed" in text
    assert "[FAIL]" not in text


# ---------------------------------------------------------------- Bonferroni
def test_wrong_bonferroni_divisor_is_caught_with_the_implied_divisor():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"][
        "perm_p_bonferroni"
    ] = 0.06
    result = _check(report)
    assert result["ok"] is False
    assert "bonferroni.identity" in _failed(result)
    assert "divisor of 6.0" in _detail(result, "bonferroni.identity")


def test_correction_that_shrinks_the_p_value_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].update(
        {"perm_p": 0.04, "perm_p_bonferroni": 0.02}
    )
    result = _check(report)
    assert "bonferroni.monotone" in _failed(result)
    assert "bonferroni.identity" in _failed(result)


def test_p_value_outside_the_unit_interval_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].update(
        {"perm_p": 1.4, "perm_p_bonferroni": 1.0}
    )
    result = _check(report)
    assert "bonferroni.p_range" in _failed(result)


def test_corrected_p_outside_the_unit_interval_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].update(
        {"perm_p": 0.5, "perm_p_bonferroni": 1.4}
    )
    result = _check(report)
    assert "bonferroni.corrected_range" in _failed(result)
    assert "bonferroni.p_range" not in _failed(result)


def test_missing_corrected_p_is_a_failure_not_a_skip():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].pop(
        "perm_p_bonferroni"
    )
    result = _check(report)
    assert "bonferroni.identity" in _failed(result)


def test_contrast_without_any_permutation_p_is_skipped():
    report = _report()
    block = report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"]
    block.pop("perm_p")
    block.pop("perm_p_bonferroni")
    result = _check(report)
    assert SKIP in _statuses(result, "bonferroni.identity")
    assert "bonferroni.identity" not in _failed(result)


def test_bonferroni_family_can_be_switched_off():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"][
        "perm_p_bonferroni"
    ] = 0.06
    result = _check(report, check_bonferroni=False, check_decision_agreement=False)
    assert result["ok"] is True
    assert not any(r["check"].startswith("bonferroni.") for r in result["checks"])


# ---------------------------------------------------------------- family size
def test_declared_family_must_match_the_grid():
    report = _report()
    report["n_contrasts"] = 6
    result = _check(report)
    assert result["ok"] is False
    assert "family.grid_size" in _failed(result)
    assert "coverage.family_matches_contrasts" in _failed(result)
    assert "isolation.family_unchanged_by_offgrid" in _failed(result)


def test_declared_family_is_accepted_when_the_run_declares_it():
    report = _report()
    report["n_contrasts"] = 6
    for block in report[DELTAS_KEY].values():
        block["faithful"]["perm_p_bonferroni"] = 0.06
    result = _check(report, expect_grid_contrasts=6, check_contrast_coverage=False)
    assert "family.grid_size" not in _failed(result)
    assert "bonferroni.identity" not in _failed(result)
    assert "isolation.family_unchanged_by_offgrid" not in _failed(result)


def test_grid_with_five_cells_needs_a_family_of_four():
    report = _report()
    report["grid"] = list(GRID_ORDER) + ["extra_cell"]
    report["cells"]["extra_cell"] = _cell()
    report["n_contrasts"] = 4
    report[DELTAS_KEY][contrast_key("extra_cell", CONTROL)] = {"faithful": _delta()}
    for block in report[DELTAS_KEY].values():
        block["faithful"]["perm_p_bonferroni"] = 0.04
    result = _check(report)
    assert "family.grid_size" not in _failed(result)
    assert "bonferroni.identity" not in _failed(result)
    assert _warned(result) == ["structure.grid_vocabulary"]
    assert result["ok"] is True


# ---------------------------------------------------------------- intervals
def test_inverted_ci_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(CHUNK_OPT, CONTROL)]["faithful"].update(
        {"ci_low": 0.35, "ci_high": 0.05}
    )
    result = _check(report)
    assert "ci.low_le_high" in _failed(result)
    assert "ci.order" in _failed(result)


def test_point_estimate_outside_its_own_ci_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(CHUNK_OPT, CONTROL)]["faithful"]["delta"] = 0.9
    result = _check(report)
    assert "ci.order" in _failed(result)
    assert "ci.low_le_high" not in _failed(result)


def test_significant_flag_must_match_its_interval():
    report = _report()
    report[DELTAS_KEY][contrast_key(COMBINED, CONTROL)]["faithful"].update(
        {"delta": 0.1, "ci_low": -0.1, "ci_high": 0.3}
    )
    result = _check(report)
    assert "significance.flag_identity" in _failed(result)


def test_cell_mean_outside_its_own_ci_is_caught():
    report = _report()
    report["cells"][QDORA]["confidence_intervals"]["faithful"]["mean"] = 0.95
    result = _check(report)
    assert "cell.ci_order" in _failed(result)


def test_cell_ci_leaving_the_unit_interval_is_caught():
    report = _report()
    report["cells"][QDORA]["confidence_intervals"]["faithful"]["ci_high"] = 1.4
    result = _check(report)
    assert "cell.bounds" in _failed(result)


def test_scored_column_longer_than_the_cell_is_caught():
    report = _report()
    report["cells"][QDORA]["confidence_intervals"]["faithful"]["n"] = 99
    result = _check(report)
    assert "cell.n" in _failed(result)


def test_unexpected_ci_method_is_caught_and_can_be_allowed():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"][
        "method"
    ] = "percentile"
    assert "method.name" in _failed(_check(report))
    allowed = _check(report, allowed_methods=("BCa", "degenerate", "percentile"))
    assert "method.name" not in _failed(allowed)


# ---------------------------------------------------------------- decisions
def test_ci_rejects_but_the_corrected_p_does_not():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].update(
        {"perm_p": 0.02, "perm_p_bonferroni": 0.06}
    )
    result = _check(report)
    assert result["ok"] is True
    assert _failed(result) == []
    assert "decision.ci_vs_p" in _warned(result)
    assert "does not reject the null" in _detail(result, "decision.ci_vs_p")
    strict = _check(report, fail_on=(FAIL, WARN))
    assert strict["ok"] is False


def test_corrected_p_rejects_but_the_ci_spans_zero():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].update(
        {"delta": 0.1, "ci_low": -0.1, "ci_high": 0.3, "significant": False,
         "perm_p": 0.005, "perm_p_bonferroni": 0.015}
    )
    result = _check(report)
    assert _failed(result) == []
    assert "decision.ci_vs_p" in _warned(result)
    assert "CI spans 0" in _detail(result, "decision.ci_vs_p")


def test_alpha_moves_the_cross_read():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].update(
        {"perm_p": 0.02, "perm_p_bonferroni": 0.06}
    )
    assert "decision.ci_vs_p" in _warned(_check(report, alpha=0.05))
    assert "decision.ci_vs_p" not in _warned(_check(report, alpha=0.10))


# ---------------------------------------------------------------- off-grid
def test_offgrid_block_carrying_a_correction_is_caught():
    report = _report()
    report["offgrid_n_contrasts"] = 3
    report[OFFGRID_DELTAS_KEY][contrast_key(PREBASELINE, CONTROL)]["faithful"][
        "perm_p_bonferroni"
    ] = 0.03
    result = _check(report)
    assert result["ok"] is False
    assert "family.offgrid_size" in _failed(result)
    assert "bonferroni.offgrid_uncorrected" in _failed(result)


def test_offgrid_correction_is_accepted_when_declared():
    report = _report()
    report["offgrid_n_contrasts"] = 3
    report[OFFGRID_DELTAS_KEY][contrast_key(PREBASELINE, CONTROL)]["faithful"][
        "perm_p_bonferroni"
    ] = 0.03
    result = _check(report, expect_offgrid_contrasts=None)
    assert result["ok"] is True
    assert SKIP in _statuses(result, "family.offgrid_size")
    assert "bonferroni.offgrid_uncorrected" not in {
        r["check"] for r in result["checks"]
    }


def test_offgrid_cell_leaking_into_the_corrected_block_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(PREBASELINE, CONTROL)] = {"faithful": _delta()}
    result = _check(report)
    assert result["ok"] is False
    assert "isolation.not_in_grid_deltas" in _failed(result)
    assert "coverage.no_extra_contrast" in _failed(result)
    assert "coverage.family_matches_contrasts" in _failed(result)


def test_offgrid_name_scored_inside_the_grid_is_caught():
    report = _report()
    report["grid"] = list(GRID_ORDER) + [PREBASELINE]
    report["offgrid"] = []
    report.pop(OFFGRID_DELTAS_KEY)
    report.pop("offgrid_n_contrasts")
    result = _check(report)
    assert "structure.offgrid_name_in_grid" in _failed(result)


def test_offgrid_block_holding_a_grid_contrast_is_caught():
    report = _report()
    report[OFFGRID_DELTAS_KEY] = {
        contrast_key(QDORA, CONTROL): {"faithful": _offgrid_delta()}
    }
    result = _check(report)
    assert "isolation.offgrid_contrast_names_offgrid_cell" in _failed(result)


def test_offgrid_reference_must_be_scored():
    report = _report()
    report["offgrid_reference"] = "ghost_cell"
    result = _check(report)
    assert "isolation.offgrid_reference_scored" in _failed(result)


def test_run_without_an_offgrid_block_skips_or_fails_on_demand():
    report = _report()
    report["offgrid"] = []
    report.pop(OFFGRID_DELTAS_KEY)
    report.pop("offgrid_n_contrasts")
    relaxed = _check(report)
    assert relaxed["ok"] is True
    assert SKIP in _statuses(relaxed, "isolation.offgrid")
    assert SKIP in _statuses(relaxed, "family.offgrid_size")
    strict = _check(report, require_offgrid=True)
    assert strict["ok"] is False
    assert "structure.offgrid_present" in _failed(strict)


# ---------------------------------------------------------------- CFCA, bounds
def test_cfca_delta_must_be_cell_minus_reference():
    report = _report()
    report[DELTAS_KEY][contrast_key(CHUNK_OPT, CONTROL)]["cfca"]["delta"] = 0.5
    result = _check(report)
    assert "cfca.delta_identity" in _failed(result)
    assert "cfca.delta_identity" not in _failed(_check(report, cfca_tol=2.0))


def test_cfca_row_with_a_missing_value_is_skipped():
    report = _report()
    report[DELTAS_KEY][contrast_key(CHUNK_OPT, CONTROL)]["cfca"]["cell"] = None
    result = _check(report)
    assert SKIP in _statuses(result, "cfca.delta_identity")
    assert "cfca.delta_identity" not in _failed(result)


def test_delta_on_a_bounded_column_must_stay_in_range():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].update(
        {"delta": 1.4, "ci_low": 1.3, "ci_high": 1.5}
    )
    result = _check(report)
    assert "bounds.delta" in _failed(result)


def test_unbounded_column_is_not_range_checked():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["latency_ms"] = _delta(
        delta=120.0, ci_low=80.0, ci_high=160.0
    )
    result = _check(report)
    assert "bounds.delta" not in _failed(result)


def test_paired_n_cannot_exceed_the_smaller_cell():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"]["n"] = 99
    result = _check(report)
    assert "pairing.n" in _failed(result)


# ---------------------------------------------------------------- structure
def test_missing_contrast_is_caught():
    report = _report()
    report[DELTAS_KEY].pop(contrast_key(COMBINED, CONTROL))
    result = _check(report)
    assert "coverage.contrast_present" in _failed(result)


def test_expected_metric_columns_are_enforced_only_when_asked():
    report = _report()
    assert "coverage.metric_present" not in _failed(_check(report))
    result = _check(report, expect_metrics=("recall@5",))
    assert "coverage.metric_present" in _failed(result)


def test_unscored_cell_and_missing_reference_are_caught():
    report = _report()
    report["cells"].pop(CONTROL)
    result = _check(report)
    assert "structure.reference_scored" in _failed(result)
    assert "structure.cell_scored" in _failed(result)


def test_reference_outside_the_grid_is_caught():
    report = _report()
    report["reference"] = "ghost_cell"
    result = _check(report)
    assert "structure.reference_in_grid" in _failed(result)
    assert result["reference"] == "ghost_cell"


def test_cell_in_both_grid_and_offgrid_is_caught():
    report = _report()
    report["offgrid"] = [PREBASELINE, QDORA]
    result = _check(report)
    assert "structure.grid_offgrid_disjoint" in _failed(result)


def test_delta_block_missing_keys_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"].pop("method")
    result = _check(report)
    assert "block.keys" in _failed(result)


def test_entry_that_is_not_a_delta_block_warns():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["notes"] = {"free": "text"}
    result = _check(report)
    assert "block.shape" in _warned(result)
    assert _failed(result) == []


def test_empty_report_fails_loudly():
    result = _check({})
    failed = _failed(result)
    assert result["ok"] is False
    assert "structure.cells_present" in failed
    assert "structure.grid_present" in failed
    assert "structure.grid_deltas_present" in failed


# ---------------------------------------------------------------- ASO score
def test_aso_divisor_disagreement_warns_without_failing():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"]["aso"] = 0.02
    result = _check(report, aso_num_comparisons=6)
    assert result["ok"] is True
    assert "aso.divisor_matches_family" in _warned(result)
    assert "6 comparisons" in _detail(result, "aso.divisor_matches_family")
    agreed = _check(report)
    assert _warned(agreed) == []


def test_aso_value_outside_the_unit_interval_is_caught():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"]["aso"] = 1.5
    result = _check(report)
    assert "aso.range" in _failed(result)


def test_aso_check_is_skipped_when_the_score_produced_nothing():
    result = _check(_report())
    assert SKIP in _statuses(result, "aso.divisor_matches_family")


# ---------------------------------------------------------------- file + render
def test_verify_report_file_reads_a_written_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    result = verify_report_file(str(path), alpha=ALPHA)
    assert result["ok"] is True
    assert result["path"] == str(path)


def test_verify_report_file_catches_a_broken_written_report(tmp_path):
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"][
        "perm_p_bonferroni"
    ] = 0.06
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = verify_report_file(str(path), alpha=ALPHA)
    assert result["ok"] is False
    assert "bonferroni.identity" in _failed(result)


def test_render_verify_text_lists_the_failure_and_can_cap_rows():
    report = _report()
    report[DELTAS_KEY][contrast_key(QDORA, CONTROL)]["faithful"][
        "perm_p_bonferroni"
    ] = 0.06
    result = _check(report)
    text = render_verify_text(result)
    assert "[FAIL] bonferroni.identity" in text
    assert "at qdora_vs_control.faithful" in text
    capped = render_verify_text(result, max_rows=1)
    assert "more" in capped.splitlines()[-1]
    full = render_verify_text(result, show=(PASS, WARN, FAIL, SKIP), max_rows=None)
    assert len(full.splitlines()) > len(text.splitlines())


def test_report_dict_is_not_mutated_by_the_check():
    report = _report()
    before = copy.deepcopy(report)
    _check(report)
    assert report == before
