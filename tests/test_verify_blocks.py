"""
The two report blocks that used to be checked by nobody: the GBP money table
and the served pool the IR metrics were capped by. Every failure mode is built
by hand and asserted by check name, so a regression fails here.

Imports stay on verify/cells so nothing pulls the model or DB chain in.
"""
import pytest

from veridic_eval.cells import CHUNK_OPT, CONTROL
from veridic_eval.verify import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    check_cfca_gbp_block,
    check_ir_pool_block,
)


def _statuses(result, check):
    return [r["status"] for r in result["checks"] if r["check"] == check]


# ------------------------------------------------------------------ GBP table
def _gbp(**over):
    """cost / p_hat, priced and differenced exactly as cfca_for_cells writes it."""
    block = {
        "currency": "GBP",
        "reference": CONTROL,
        "cells": {
            CONTROL: {"cost_per_answer": 0.004, "currency": "GBP",
                      "p_hat": 0.8, "cfca": 0.005},
            CHUNK_OPT: {"cost_per_answer": 0.006, "currency": "GBP",
                        "p_hat": 0.6, "cfca": 0.01},
        },
        "deltas": {CHUNK_OPT: {"cost_per_answer": 0.002, "cfca": 0.005}},
    }
    block.update(over)
    return block


def _priced_report(block=None, *, control_p=0.8, cell_p=0.6):
    return {
        "cells": {
            CONTROL: {"cfca": {"p_faithful_cited_version": control_p}},
            CHUNK_OPT: {"cfca": {"p_faithful_cited_version": cell_p}},
        },
        "cfca_gbp": _gbp() if block is None else block,
    }


def test_a_clean_gbp_table_passes_every_family():
    res = check_cfca_gbp_block(_priced_report())
    assert res["ok"] is True
    assert res["counts"][FAIL] == 0 and res["counts"][WARN] == 0
    assert _statuses(res, "cfca_gbp.ratio_identity") == [PASS, PASS]
    assert _statuses(res, "cfca_gbp.delta_identity") == [PASS, PASS]
    assert _statuses(res, "cfca_gbp.p_hat_matches_report") == [PASS, PASS]


def test_cfca_that_is_not_cost_over_the_rate_fails():
    block = _gbp()
    block["cells"][CHUNK_OPT]["cfca"] = 0.02          # 0.006 / 0.6 is 0.01
    res = check_cfca_gbp_block(_priced_report(block))
    assert FAIL in _statuses(res, "cfca_gbp.ratio_identity")
    assert res["ok"] is False


def test_a_small_rate_does_not_fail_on_its_own_rounding():
    block = _gbp()
    block["cells"][CHUNK_OPT].update({"cost_per_answer": 1.0, "p_hat": 0.02,
                                      "cfca": 50.0000001})
    block["deltas"][CHUNK_OPT] = {"cost_per_answer": 0.996, "cfca": 49.9950001}
    res = check_cfca_gbp_block(_priced_report(block, cell_p=0.02))
    assert _statuses(res, "cfca_gbp.ratio_identity") == [PASS, PASS]


def test_a_delta_that_is_not_cell_minus_reference_fails():
    block = _gbp()
    block["deltas"][CHUNK_OPT]["cfca"] = 0.009        # 0.01 - 0.005 is 0.005
    res = check_cfca_gbp_block(_priced_report(block))
    assert FAIL in _statuses(res, "cfca_gbp.delta_identity")


def test_a_priced_rate_that_left_the_scored_report_warns_then_fails_on_demand():
    warned = check_cfca_gbp_block(_priced_report(cell_p=0.9))
    assert WARN in _statuses(warned, "cfca_gbp.p_hat_matches_report")
    assert warned["ok"] is True                       # an override explains it
    strict = check_cfca_gbp_block(_priced_report(cell_p=0.9), p_hat_source="require")
    assert strict["ok"] is False


def test_an_undefined_rate_may_not_carry_a_price():
    block = _gbp()
    block["cells"][CHUNK_OPT].update({"p_hat": 0.0, "cfca": 0.01})
    block["deltas"] = {}
    res = check_cfca_gbp_block(_priced_report(block, cell_p=0.0),
                               require_deltas=False)
    assert FAIL in _statuses(res, "cfca_gbp.undefined_is_empty")


def test_an_unmeasured_cell_keeps_its_cost_and_no_price():
    block = _gbp()
    block["cells"][CHUNK_OPT].update({"p_hat": None, "cfca": None})
    block["deltas"] = {}
    res = check_cfca_gbp_block(_priced_report(block, cell_p=None),
                               require_deltas=False)
    assert _statuses(res, "cfca_gbp.undefined_is_empty") == [PASS]
    assert res["ok"] is True


def test_a_run_with_no_cost_block_is_skipped_unless_it_is_required():
    assert _statuses(check_cfca_gbp_block({"cells": {}}), "cfca_gbp.present") == [SKIP]
    strict = check_cfca_gbp_block({"cells": {}}, require_block=True)
    assert strict["ok"] is False


def test_a_reference_with_no_price_is_named():
    block = _gbp(reference="nowhere")
    res = check_cfca_gbp_block(_priced_report(block))
    assert FAIL in _statuses(res, "cfca_gbp.reference_in_cells")


def test_a_negative_cost_is_caught_unless_it_is_allowed():
    block = _gbp()
    block["cells"][CHUNK_OPT]["cost_per_answer"] = -0.006
    block["cells"][CHUNK_OPT]["cfca"] = -0.01
    block["deltas"][CHUNK_OPT] = {"cost_per_answer": -0.01, "cfca": -0.015}
    res = check_cfca_gbp_block(_priced_report(block))
    assert FAIL in _statuses(res, "cfca_gbp.cost_bounds")
    ok = check_cfca_gbp_block(_priced_report(block), allow_negative_cost=True)
    assert _statuses(ok, "cfca_gbp.cost_bounds") == [PASS, PASS]


# ------------------------------------------------------------------ served pool
def _pool(**over):
    pool = {
        "k": 5,
        "n_records": 10,
        "n_with_top_n": 10,
        "n_missing_top_n": 0,
        "top_n_min": 5,
        "top_n_max": 5,
        "top_n_values": {"5": 10},
        "top_k_min": 20,
        "top_k_max": 20,
        "served_min": 4,
        "served_max": 5,
    }
    pool.update(over)
    return pool


def _pooled_report(control=None, cell=None):
    return {
        "cells": {
            CONTROL: {"ir_pool": _pool() if control is None else control},
            CHUNK_OPT: {"ir_pool": _pool() if cell is None else cell},
        }
    }


def test_a_pool_as_deep_as_k_passes():
    res = check_ir_pool_block(_pooled_report())
    assert res["ok"] is True
    assert res["counts"][WARN] == 0
    assert _statuses(res, "ir_pool.pool_reaches_k") == [PASS, PASS]
    assert _statuses(res, "ir_pool.top_n_agrees") == [PASS]


def test_a_cell_served_shallower_than_k_fails_and_can_be_downgraded():
    shallow = _pool(top_n_min=3, top_n_max=3, top_n_values={"3": 10}, served_max=3)
    res = check_ir_pool_block(_pooled_report(cell=shallow))
    assert FAIL in _statuses(res, "ir_pool.pool_reaches_k")
    assert res["ok"] is False
    soft = check_ir_pool_block(_pooled_report(cell=shallow), allow_top_n_below_k=True,
                               require_equal_top_n=False)
    assert WARN in _statuses(soft, "ir_pool.pool_reaches_k")
    assert soft["ok"] is True


def test_cells_that_served_different_widths_are_not_comparable():
    other = _pool(top_n_min=8, top_n_max=8, top_n_values={"8": 10}, served_max=8)
    res = check_ir_pool_block(_pooled_report(cell=other))
    assert FAIL in _statuses(res, "ir_pool.top_n_agrees")
    soft = check_ir_pool_block(_pooled_report(cell=other), require_equal_top_n=False)
    assert WARN in _statuses(soft, "ir_pool.top_n_agrees")


def test_a_missing_rag_log_falls_back_to_the_deepest_served_list():
    blind = _pool(top_n_min=None, top_n_max=None, top_n_values={},
                  n_with_top_n=0, n_missing_top_n=10, served_max=3)
    res = check_ir_pool_block(_pooled_report(cell=blind))
    assert WARN in _statuses(res, "ir_pool.pool_reaches_k")
    assert WARN in _statuses(res, "ir_pool.top_n_present")
    assert res["ok"] is True
    off = check_ir_pool_block(_pooled_report(cell=blind), fallback_to_served=False)
    assert SKIP in _statuses(off, "ir_pool.pool_reaches_k")


def test_cells_scored_at_different_k_are_not_one_measurement():
    res = check_ir_pool_block(_pooled_report(cell=_pool(k=10)))
    assert FAIL in _statuses(res, "ir_pool.k_agrees")
    assert _statuses(res, "ir_pool.pool_reaches_k") == [SKIP, SKIP]


def test_an_expected_k_is_checked_cell_by_cell():
    res = check_ir_pool_block(_pooled_report(), expect_k=10)
    assert _statuses(res, "ir_pool.k_matches") == [FAIL, FAIL]


def test_a_report_written_before_the_pool_block_is_skipped():
    older = {"cells": {CONTROL: {}, CHUNK_OPT: {}}}
    assert _statuses(check_ir_pool_block(older), "ir_pool.present") == [SKIP]
    strict = check_ir_pool_block(older, require_pool=True)
    assert strict["ok"] is False


def test_only_restricts_the_cells_checked():
    other = _pool(top_n_min=8, top_n_max=8, top_n_values={"8": 10}, served_max=8)
    res = check_ir_pool_block(_pooled_report(cell=other), only=[CONTROL])
    assert _statuses(res, "ir_pool.top_n_agrees") == [PASS]


def test_the_two_checks_share_one_ordered_sink():
    from veridic_eval.verify import verify_report

    report = _priced_report()
    report["cells"][CONTROL]["ir_pool"] = _pool()
    report["cells"][CHUNK_OPT]["ir_pool"] = _pool()
    result = verify_report(report, require_grid_deltas=False)
    names = [r["check"] for r in result["checks"]]
    assert "cfca_gbp.ratio_identity" in names
    assert "ir_pool.pool_reaches_k" in names
    assert names.index("cfca_gbp.present") < names.index("ir_pool.present")


def test_an_unknown_p_hat_policy_is_refused():
    with pytest.raises(ValueError):
        check_cfca_gbp_block(_priced_report(), p_hat_source="whatever")
