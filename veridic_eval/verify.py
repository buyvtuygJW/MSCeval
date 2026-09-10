"""
Post-run math check on a report that was already written.

Reads `out/report.json` (or a report dict still in memory) and re-derives every
statistical claim in it from the numbers stored beside the claim, so a wrong
Bonferroni divisor, an inverted CI, a `significant` flag that does not match its
own interval, a GBP CFCA that is not its own cost over its own all-pass rate, a
k scored deeper than the pool the app served, or an off-grid before/after that
quietly joined the corrected family all surface as a named failure with the
contrast and metric that carry it. Nothing is re-scored and no model, DB or
network is touched: this reads the report only, which is why it can run straight
after `veridic-eval run`.

    veridic-eval verify --report out/report.json

`verify_report` is the whole check and takes every threshold, expected family
size and column list as an explicit argument, so a run with a different grid,
a different correction or a folded-in off-grid block is checked on its own terms
instead of against these defaults.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cells import (
    DEFAULT_REFERENCE,
    GRID_ORDER,
    OFFGRID_ORDER,
    contrast_key,
    deltas_of,
    offgrid_deltas_of,
)
from .config import settings
from .stats import bonferroni

PASS: str = "pass"
FAIL: str = "fail"
WARN: str = "warn"
SKIP: str = "skip"

STATUS_ORDER: Tuple[str, ...] = (FAIL, WARN, PASS, SKIP)

# Per-query columns that are 0/1 or a normalised rank score, so the cell mean
# sits in [0, 1] and any paired delta sits in [-1, 1].
BOUNDED_COLUMNS: Tuple[str, ...] = (
    f"recall@{settings.ir_k}",
    f"mrr@{settings.ir_k}",
    f"ndcg@{settings.ir_k}",
    f"hit_rate@{settings.ir_k}",
    "faithful",
    "abstention_recall",
    "over_abstention_rate",
    "cfca_joint_P",
)

# Keys of a paired-delta block, as stats.paired_delta_ci writes them.
DELTA_KEYS: Tuple[str, ...] = ("n", "delta", "ci_low", "ci_high", "significant", "method")

# Keys of the aggregate CFCA row: two cell values and their difference. A row
# that also carries DELTA_KEYS is checked as a paired delta on top of these.
CFCA_KEYS: Tuple[str, ...] = ("cell", "reference", "delta")

# The three stored numbers of one cell in the GBP CFCA block, as
# conditions.cfca_for_cells writes them: the money, the rate, and the ratio.
CFCA_GBP_KEYS: Tuple[str, ...] = ("cost_per_answer", "p_hat", "cfca")


def _num(value: Any) -> Optional[float]:
    """`value` as a float, or None when it is missing or not a number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_delta_block(block: Any) -> bool:
    return isinstance(block, dict) and "delta" in block and "ci_low" in block


def _is_cfca_block(block: Any) -> bool:
    return isinstance(block, dict) and "cell" in block and "reference" in block


class _Log:
    """Ordered check sink: one row per assertion, with where it was made."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, check: str, status: str, where: str = "", detail: str = "",
            expected: Any = None, got: Any = None) -> None:
        self.rows.append({
            "check": check, "status": status, "where": where, "detail": detail,
            "expected": expected, "got": got,
        })

    def require(self, check: str, ok: bool, where: str = "", detail: str = "",
                expected: Any = None, got: Any = None, on_fail: str = FAIL) -> bool:
        self.add(check, PASS if ok else on_fail, where, "" if ok else detail,
                 expected, got)
        return ok


def check_cfca_gbp_block(
    report: Dict[str, Any],
    *,
    log: Optional[_Log] = None,
    block_key: str = "cfca_gbp",
    cells_key: str = "cells",
    deltas_key: str = "deltas",
    reference_key: str = "reference",
    currency_key: str = "currency",
    cost_key: str = "cost_per_answer",
    p_hat_key: str = "p_hat",
    cfca_key: str = "cfca",
    report_cells_key: str = "cells",
    report_cfca_key: str = "cfca",
    report_p_hat_key: str = "p_faithful_cited_version",
    require_block: bool = False,
    require_deltas: bool = True,
    check_structure: bool = True,
    check_ratio: bool = True,
    check_deltas: bool = True,
    check_bounds: bool = True,
    check_currency: bool = True,
    skip_unscored_reference: bool = True,
    p_hat_source: str = "warn",
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
    assume_round_to: Optional[int] = 8,
    p_hat_tol: float = 5e-5,
    p_hat_bounds: Tuple[float, float] = (0.0, 1.0),
    allow_negative_cost: bool = False,
    fail_on: Sequence[str] = (FAIL,),
) -> Dict[str, Any]:
    """
    Re-derive the GBP CFCA table from the numbers stored beside it.

    The pound figures written by `conditions.cfca_for_cells` are the money
    headline of report.md and they were the one block no check touched: their
    deltas are a plain subtraction and their ratio is asserted nowhere, so a
    mistyped electricity rate, or a cost block that quietly kept a declared
    ``p_gpu_hour`` over the measured watts, printed a wrong pound figure with a
    `pass` beside it. Nothing is recomputed from a model, a DB or a price sheet;
    this reads the block's own cost, rate and difference.

    Args:
        log: shared check sink; None makes a private one and returns its rows.
        block_key: report field holding the block; `pipeline` writes
            ``cfca_gbp`` and takes ``cfca_field`` to move it.
        cells_key / deltas_key / reference_key / currency_key: layout of the
            block, as `cfca_for_cells` writes it.
        cost_key / p_hat_key / cfca_key: the three numbers of one cell's entry.
        report_cells_key / report_cfca_key / report_p_hat_key: where the
            all-pass rate lives in the scored report, for the cross-read that
            the block's ``p_hat`` is the same rate the 2x2 reported.
        require_block: True fails a report with no GBP block; False skips it,
            which is right for a run whose conditions declared no ``cost:``.
        require_deltas: every non-reference cell must carry a delta row.
        check_structure / check_ratio / check_deltas / check_bounds /
            check_currency: switch off any single family of assertions.
        skip_unscored_reference: a partial report holds no GBP entry for a
            reference cell nobody has answered yet, so that row is skipped
            rather than failed. False fails it. A reference that IS scored and
            still has no entry fails either way.
        p_hat_source: ``warn`` | ``require`` | ``skip`` on a ``p_hat`` that
            differs from the scored report's rate. A hand-set ``p_hat_override``
            is a legitimate reason for the gap, so the default only warns.
        abs_tol / rel_tol: absolute and relative slack on the ratio and the
            deltas.
        assume_round_to: decimals the block was rounded to; the ratio inherits
            the numerator's rounding amplified by ``1 / p_hat``, so this widens
            the ratio tolerance by ``0.5e-round_to / p_hat`` instead of letting
            a small all-pass rate fail on arithmetic that is correct. None
            assumes the stored numbers are exact.
        p_hat_tol: slack on the cross-read, which compares a 4-decimal rate.
        p_hat_bounds: the rate is a probability; widen only with a reason.
        allow_negative_cost: True permits a negative cost per answer.
        fail_on: statuses that make the returned ``ok`` False.

    Returns:
        ``{ok, counts, checks}`` for the rows this call added.
    """
    if p_hat_source not in ("warn", "require", "skip"):
        raise ValueError("p_hat_source must be 'warn', 'require' or 'skip'")
    own = log if log is not None else _Log()
    start = len(own.rows)

    def _done() -> Dict[str, Any]:
        rows = own.rows[start:]
        counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_ORDER}
        return {
            "ok": not any(counts.get(s, 0) for s in tuple(fail_on)),
            "counts": counts,
            "checks": rows,
        }

    block = report.get(block_key)
    if not isinstance(block, dict) or not block:
        if require_block:
            own.require("cfca_gbp.present", False, where=block_key,
                        detail="report carries no GBP CFCA block",
                        expected=block_key, got=None)
        else:
            own.add("cfca_gbp.present", SKIP, where=block_key,
                    detail="no GBP CFCA block (no cell declared a cost: block)")
        return _done()

    cells = block.get(cells_key)
    if not isinstance(cells, dict) or not cells:
        own.require("cfca_gbp.cells_present", False, where=block_key,
                    detail="GBP block holds no cells", expected=cells_key, got=cells)
        return _done()

    own.add("cfca_gbp.present", PASS, where=block_key)
    reference = block.get(reference_key)
    deltas = block.get(deltas_key) if isinstance(block.get(deltas_key), dict) else {}
    scored = report.get(report_cells_key) if isinstance(report.get(report_cells_key), dict) else {}
    currency = block.get(currency_key)

    # ------------------------------------------------------------- structure
    if check_structure:
        for name, entry in cells.items():
            where = f"{block_key}.{name}"
            if not isinstance(entry, dict):
                own.require("cfca_gbp.entry_shape", False, where=where,
                            detail="cell entry is not a mapping",
                            expected="mapping", got=type(entry).__name__)
                continue
            missing = [k for k in CFCA_GBP_KEYS if k not in entry]
            own.require("cfca_gbp.keys", not missing, where=where,
                        detail="cell entry is missing a stored number",
                        expected=list(CFCA_GBP_KEYS), got=missing)
        if reference is None:
            own.add("cfca_gbp.reference_in_cells", SKIP, where=block_key,
                    detail="block declares no reference, so it carries no deltas")
        elif (skip_unscored_reference and reference not in cells
                and reference not in scored
                and isinstance(report.get("partial"), dict)):
            own.add("cfca_gbp.reference_in_cells", SKIP, where=block_key,
                    detail=f"partial report: reference cell {reference!r} is not "
                           "scored yet, so it has no GBP entry to compare",
                    expected=sorted(cells), got=reference)
        else:
            own.require("cfca_gbp.reference_in_cells", reference in cells,
                        where=block_key, detail="reference cell has no GBP entry",
                        expected=sorted(cells), got=reference)
        if deltas:
            own.require("cfca_gbp.deltas_exclude_reference", reference not in deltas,
                        where=f"{block_key}.{deltas_key}",
                        detail="the reference cell carries a delta against itself",
                        expected=[], got=[reference] if reference in deltas else [])
            stray = sorted(k for k in deltas if k not in cells)
            own.require("cfca_gbp.deltas_in_cells", not stray,
                        where=f"{block_key}.{deltas_key}",
                        detail="delta named for a cell with no GBP entry",
                        expected=[], got=stray)

    # ----------------------------------------------------------------- ratio
    for name, entry in cells.items():
        if not isinstance(entry, dict):
            continue
        where = f"{block_key}.{name}"
        cost = _num(entry.get(cost_key))
        rate = _num(entry.get(p_hat_key))
        value = _num(entry.get(cfca_key))

        if check_currency:
            own.require("cfca_gbp.currency", entry.get(currency_key, currency) == currency,
                        where=where, detail="cell is priced in another currency",
                        expected=currency, got=entry.get(currency_key))

        if check_bounds:
            if cost is None:
                own.add("cfca_gbp.cost_bounds", SKIP, where=where,
                        detail="no cost per answer stored")
            else:
                own.require("cfca_gbp.cost_bounds", allow_negative_cost or cost >= 0.0,
                            where=where, detail="cost per answer is negative",
                            expected=">= 0", got=cost)
            if rate is None:
                own.add("cfca_gbp.p_hat_bounds", SKIP, where=where,
                        detail="no all-pass rate stored")
            else:
                lo, hi = p_hat_bounds
                own.require("cfca_gbp.p_hat_bounds", lo <= rate <= hi, where=where,
                            detail="all-pass rate is not a probability",
                            expected=f"{lo} <= p_hat <= {hi}", got=rate)

        if check_ratio:
            if rate is None or rate <= 0.0:
                own.require("cfca_gbp.undefined_is_empty", value is None, where=where,
                            detail="CFCA is stored while the all-pass rate is 0 "
                                   "or unmeasured, so the ratio has no value",
                            expected=None, got=value)
            elif cost is None or value is None:
                own.add("cfca_gbp.ratio_identity", SKIP, where=where,
                        detail="cost or CFCA missing, nothing to divide")
            else:
                want = cost / rate
                allow = 0.0
                if assume_round_to is not None:
                    allow = 0.5 * (10.0 ** -assume_round_to) / rate
                tol = max(abs_tol, rel_tol * abs(want), allow)
                own.require("cfca_gbp.ratio_identity", abs(value - want) <= tol,
                            where=where,
                            detail="CFCA is not cost per answer over the all-pass rate",
                            expected=round(want, 8), got=value)

        if p_hat_source != "skip":
            cell_block = scored.get(name) if isinstance(scored.get(name), dict) else {}
            cfca_row = cell_block.get(report_cfca_key)
            scored_rate = _num(cfca_row.get(report_p_hat_key)) if isinstance(cfca_row, dict) else None
            if rate is None or scored_rate is None:
                own.add("cfca_gbp.p_hat_matches_report", SKIP, where=where,
                        detail="the scored report carries no rate for this cell")
            else:
                own.require(
                    "cfca_gbp.p_hat_matches_report", abs(rate - scored_rate) <= p_hat_tol,
                    where=where,
                    detail=("the priced rate is not the rate the 2x2 reported; a "
                            "p_hat_override explains it, a stale cost run does not"),
                    expected=scored_rate, got=rate,
                    on_fail=FAIL if p_hat_source == "require" else WARN,
                )

    # ---------------------------------------------------------------- deltas
    if check_deltas:
        if reference is None or reference not in cells:
            own.add("cfca_gbp.delta_identity", SKIP, where=block_key,
                    detail="no reference cell, so no delta to re-derive")
        else:
            ref_entry = cells[reference] if isinstance(cells[reference], dict) else {}
            for name, entry in cells.items():
                if name == reference or not isinstance(entry, dict):
                    continue
                where = f"{block_key}.{deltas_key}.{name}"
                row = deltas.get(name)
                if not isinstance(row, dict):
                    if require_deltas:
                        own.require("cfca_gbp.delta_present", False, where=where,
                                    detail="cell has no GBP delta against the reference",
                                    expected=name, got=None)
                    else:
                        own.add("cfca_gbp.delta_present", SKIP, where=where,
                                detail="no GBP delta for this cell")
                    continue
                own.add("cfca_gbp.delta_present", PASS, where=where)
                for field in (cost_key, cfca_key):
                    a, b = _num(entry.get(field)), _num(ref_entry.get(field))
                    d = _num(row.get(field))
                    if a is None or b is None:
                        own.require("cfca_gbp.delta_identity", d is None, where=where,
                                    detail=f"{field}: a delta is stored while one "
                                           f"side of the subtraction is missing",
                                    expected=None, got=d)
                        continue
                    if d is None:
                        own.require("cfca_gbp.delta_identity", False, where=where,
                                    detail=f"{field}: both cells are priced but no "
                                           f"difference is stored",
                                    expected=round(a - b, 8), got=None)
                        continue
                    tol = max(abs_tol, rel_tol * abs(a - b))
                    own.require("cfca_gbp.delta_identity", abs((a - b) - d) <= tol,
                                where=where,
                                detail=f"{field}: delta is not cell minus reference",
                                expected=round(a - b, 8), got=d)
    return _done()


def check_ir_pool_block(
    report: Dict[str, Any],
    *,
    log: Optional[_Log] = None,
    cells_key: str = "cells",
    pool_key: str = "ir_pool",
    k_key: str = "k",
    top_n_min_key: str = "top_n_min",
    top_n_values_key: str = "top_n_values",
    missing_key: str = "n_missing_top_n",
    served_max_key: str = "served_max",
    expect_k: Optional[int] = None,
    only: Optional[Sequence[str]] = None,
    compare_names: Optional[Sequence[str]] = None,
    require_pool: bool = False,
    require_equal_top_n: bool = True,
    allow_top_n_below_k: bool = False,
    fallback_to_served: bool = True,
    served_status: str = WARN,
    missing_status: str = WARN,
    fail_on: Sequence[str] = (FAIL,),
) -> Dict[str, Any]:
    """
    Hold the logs-only cap to its claim: every scored k must fit the served pool.

    `retrieval_eval`, `config`, `report` and the package docstring all state that
    metrics are capped at ``k <= top_n``, because the pre-rerank candidates are
    never logged. Nothing enforced it, so a cell whose app served three rows was
    free to report recall@5 over a pool of three, and its delta against a cell
    served at five would be an artefact of the app's configuration rather than a
    retrieval result. This reads the per-cell served-pool summary that
    `retrieval_eval.served_pool_summary` writes into the report.

    Args:
        log: shared check sink; None makes a private one and returns its rows.
        cells_key / pool_key: where each cell's pool summary sits.
        k_key / top_n_min_key / top_n_values_key / missing_key /
            served_max_key: fields of that summary.
        expect_k: the k this run is supposed to have used; None takes the one
            the cells agree on, so a run scored at another k checks itself.
        only: cells to check; None takes every cell in the report.
        compare_names: cells whose widths must agree, the ones whose deltas are
            read against each other; None compares every checked cell. The
            off-grid before/after cell ran another configuration on purpose, so
            `verify_report` narrows this to the grid.
        require_pool: True fails a report written before the summary existed;
            False skips those cells.
        require_equal_top_n: every checked cell must have served the same
            width, else the grid is comparing different pool depths.
        allow_top_n_below_k: True downgrades a pool shallower than k to a
            warning, for a run that reports the shortfall as a limitation.
        fallback_to_served: when the rag log is missing, judge the pool by the
            largest served evidence list instead, at ``served_status``, because
            that is indirect: a query can serve fewer rows than the app asked
            for when the corpus holds fewer chunks.
        served_status / missing_status: status used for the served fallback and
            for records that carry no logged width.
        fail_on: statuses that make the returned ``ok`` False.

    Returns:
        ``{ok, counts, checks}`` for the rows this call added.
    """
    own = log if log is not None else _Log()
    start = len(own.rows)

    def _done() -> Dict[str, Any]:
        rows = own.rows[start:]
        counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_ORDER}
        return {
            "ok": not any(counts.get(s, 0) for s in tuple(fail_on)),
            "counts": counts,
            "checks": rows,
        }

    cells = report.get(cells_key) if isinstance(report.get(cells_key), dict) else {}
    names = [n for n in (only if only is not None else list(cells))]
    pools: Dict[str, Dict[str, Any]] = {}
    absent: List[str] = []
    for name in names:
        cell = cells.get(name)
        pool = cell.get(pool_key) if isinstance(cell, dict) else None
        if isinstance(pool, dict):
            pools[name] = pool
        else:
            absent.append(name)

    if absent:
        if require_pool:
            own.require("ir_pool.present", False, where=cells_key,
                        detail="cells carry no served-pool summary",
                        expected=[], got=sorted(absent))
        else:
            own.add("ir_pool.present", SKIP, where=cells_key,
                    detail=f"no served-pool summary for {sorted(absent)} "
                           f"(report written before the block existed)")
    if not pools:
        return _done()
    if not absent:
        own.add("ir_pool.present", PASS, where=cells_key)

    ks = {n: _num(p.get(k_key)) for n, p in pools.items()}
    distinct_k = sorted({v for v in ks.values() if v is not None})
    if expect_k is None:
        own.require("ir_pool.k_agrees", len(distinct_k) <= 1, where=cells_key,
                    detail="cells were scored at different k, so their IR "
                           "columns are not the same measurement",
                    expected="one k", got={n: v for n, v in ks.items()})
        want_k = distinct_k[0] if len(distinct_k) == 1 else None
    else:
        want_k = float(expect_k)
        for name, v in ks.items():
            own.require("ir_pool.k_matches", v == want_k, where=f"{cells_key}.{name}",
                        detail="cell was not scored at the k this check was given",
                        expected=expect_k, got=v)

    widths: Dict[str, Any] = {}
    missing_total = 0
    for name, pool in pools.items():
        where = f"{cells_key}.{name}"
        values = pool.get(top_n_values_key)
        widths[name] = sorted(values) if isinstance(values, dict) else None
        missing_total += int(_num(pool.get(missing_key)) or 0)

        low = _num(pool.get(top_n_min_key))
        if low is None:
            served = _num(pool.get(served_max_key)) if fallback_to_served else None
            if served is None or want_k is None:
                own.add("ir_pool.pool_reaches_k", SKIP, where=where,
                        detail="no logged top_n and no served count to fall back on")
            else:
                own.require("ir_pool.pool_reaches_k", served >= want_k, where=where,
                            detail="no rag log; the deepest served evidence list is "
                                   "shorter than k, so the pool never held k rows",
                            expected=f">= {want_k}", got=served,
                            on_fail=served_status)
        elif want_k is None:
            own.add("ir_pool.pool_reaches_k", SKIP, where=where,
                    detail="no agreed k to compare the served width against")
        else:
            own.require("ir_pool.pool_reaches_k", low >= want_k, where=where,
                        detail="the app served fewer rows than the k this cell is "
                               "scored at, so recall@k divides by a pool that "
                               "never held k rows",
                        expected=f"top_n >= {want_k}", got=low,
                        on_fail=WARN if allow_top_n_below_k else FAIL)

    compared = ({n: w for n, w in widths.items() if n in set(compare_names)}
                if compare_names is not None else widths)
    seen = {tuple(v) for v in compared.values() if v}
    if not seen:
        own.add("ir_pool.top_n_agrees", SKIP, where=cells_key,
                detail="no compared cell logged a top_n")
    else:
        own.require("ir_pool.top_n_agrees", len(seen) == 1, where=cells_key,
                    detail="cells served different pool widths, so a delta between "
                           "them mixes a retrieval effect with a configuration one",
                    expected="one width", got=compared,
                    on_fail=FAIL if require_equal_top_n else WARN)

    own.require("ir_pool.top_n_present", missing_total == 0, where=cells_key,
                detail="answers with no rag log, so their served width is unknown",
                expected=0, got=missing_total, on_fail=missing_status)
    return _done()


def verify_report(
    report: Dict[str, Any],
    *,
    reference: Optional[str] = None,
    grid_names: Sequence[str] = GRID_ORDER,
    offgrid_names: Sequence[str] = OFFGRID_ORDER,
    expect_grid_contrasts: Optional[int] = None,
    expect_offgrid_contrasts: Optional[int] = 1,
    expect_metrics: Sequence[str] = (),
    bounded_columns: Sequence[str] = BOUNDED_COLUMNS,
    cfca_key: str = "cfca",
    alpha: Optional[float] = None,
    p_tol: float = 1e-5,
    ci_tol: float = 1e-9,
    bound_tol: float = 1e-9,
    cfca_tol: float = 1e-6,
    allowed_methods: Sequence[str] = ("BCa", "degenerate"),
    aso_num_comparisons: int = 3,
    require_offgrid: bool = False,
    require_grid_deltas: bool = True,
    check_structure: bool = True,
    check_family_size: bool = True,
    check_contrast_coverage: bool = True,
    check_bonferroni: bool = True,
    check_ci_order: bool = True,
    check_significance_flag: bool = True,
    check_method_names: bool = True,
    check_pairing_n: bool = True,
    check_bounds: bool = True,
    check_cfca_delta: bool = True,
    check_offgrid_isolation: bool = True,
    check_decision_agreement: bool = True,
    check_aso_divisor: bool = True,
    check_cfca_gbp: bool = True,
    cfca_gbp_kwargs: Optional[Dict[str, Any]] = None,
    check_warm_only: bool = True,
    warm_only_key: str = "cfca_warm_only",
    warm_only_kwargs: Optional[Dict[str, Any]] = None,
    check_ir_pool: bool = True,
    ir_pool_kwargs: Optional[Dict[str, Any]] = None,
    fail_on: Sequence[str] = (FAIL,),
) -> Dict[str, Any]:
    """
    Re-derive every statistic in `report` from the report's own numbers.

    Args:
        reference: cell the grid deltas must be measured against; None reads
            ``report["reference"]`` and falls back to the control cell.
        grid_names / offgrid_names: the vocabulary this run is expected to use.
            A name in neither list is accepted wherever the report filed it and
            only reported, since `split_grid` lets a run score extra cells.
        expect_grid_contrasts: Bonferroni divisor the grid must use; None derives
            it as len(grid) - 1, which is the 3 the four-cell grid gives.
        expect_offgrid_contrasts: divisor the off-grid block must use; 1 is
            uncorrected, None accepts whatever the report declares and only
            checks the arithmetic against that declaration.
        expect_metrics: metric columns every contrast must carry; () accepts
            whatever the report has and checks only what is present.
        bounded_columns: columns whose cell mean must sit in [0, 1] and whose
            delta must sit in [-1, 1].
        cfca_key: delta-block key holding the aggregate CFCA subtraction.
        alpha: significance level the corrected p is read against; None uses
            1 - settings.confidence_level.
        p_tol: slack on the recomputed Bonferroni p, which absorbs the 6-decimal
            rounding the report stores.
        ci_tol / bound_tol / cfca_tol: slack on interval ordering, on the
            [0, 1] bounds, and on the CFCA subtraction.
        allowed_methods: CI method strings this run may contain.
        aso_num_comparisons: divisor the optional ASO score was called with, so
            a disagreement with the permutation divisor is reported rather than
            assumed harmless. Deployed code hands stats.aso_significance the
            same n_contrasts the permutation used, so 3 is the four-cell grid's
            family; pass the run's own number for any other grid.
        require_offgrid: a run with no off-grid block fails instead of skipping,
            for a report that is supposed to carry the before/after check.
        require_grid_deltas: a run with no grid delta block fails.
        check_*: switch off any single family of assertions.
        cfca_gbp_kwargs / ir_pool_kwargs: forwarded to
            `check_cfca_gbp_block` / `check_ir_pool_block`, which hold every
            tunable of the GBP money table and the served-pool cap. A report
            written without either block skips those checks by default.
        warm_only_key / warm_only_kwargs: the `warmup` sensitivity block, checked
            with the same arithmetic as the money table it is derived from;
            ``warm_only_kwargs`` overrides ``cfca_gbp_kwargs`` key by key for
            that pass only. Absent, it skips: a report is complete without it.
        fail_on: statuses that make ``ok`` False; ("fail", "warn") treats a
            warning as blocking.

    Returns a dict with ``ok``, ``counts`` per status, ``failed``/``warned``
    shortlists, and ``checks``, the full ordered row list.
    """
    log = _Log()
    alpha = (1.0 - settings.confidence_level) if alpha is None else alpha

    ref = reference or report.get("reference") or DEFAULT_REFERENCE
    cells: Dict[str, Any] = report.get("cells") or {}
    grid: List[str] = list(report.get("grid") or [])
    offgrid: List[str] = list(report.get("offgrid") or [])
    deltas: Dict[str, Any] = deltas_of(report)
    off_deltas: Dict[str, Any] = offgrid_deltas_of(report)
    off_ref = report.get("offgrid_reference") or ref
    declared_family = _num(report.get("n_contrasts"))
    declared_off_family = _num(report.get("offgrid_n_contrasts"))
    partial = isinstance(report.get("partial"), dict)
    partial_why = ((report.get("partial") or {}).get("reason")
                   or "partial report") if partial else ""

    # ---------------------------------------------------------------- structure
    if check_structure:
        log.require("structure.cells_present", bool(cells),
                    where="cells", detail="report has no cells block")
        if partial and not grid:
            log.add("structure.grid_present", SKIP, where="grid",
                    detail=f"{partial_why}; the grid block is written as its "
                           "cells are scored")
        else:
            log.require("structure.grid_present", bool(grid),
                        where="grid", detail="report declares no grid")
        if partial:
            log.add("structure.reference_scored", WARN, where=f"reference={ref}",
                    detail=f"{partial_why}; per-cell blocks are checked, every "
                           "delta check waits on that cell")
        else:
            log.require("structure.reference_in_grid", ref in grid or not grid,
                        where=f"reference={ref}",
                        detail="reference cell is not in the grid", expected=grid, got=ref)
            log.require("structure.reference_scored", ref in cells,
                        where=f"reference={ref}", detail="reference cell has no scored block")
        log.require("structure.grid_offgrid_disjoint", not (set(grid) & set(offgrid)),
                    where="grid/offgrid",
                    detail="a cell is listed in both the grid and the off-grid block",
                    got=sorted(set(grid) & set(offgrid)))
        for name in grid + offgrid:
            log.require("structure.cell_scored", name in cells, where=name,
                        detail="listed condition has no scored block")
        unexpected = [n for n in grid if n not in tuple(grid_names)]
        log.add("structure.grid_vocabulary", PASS if not unexpected else WARN,
                where="grid", detail="" if not unexpected else
                "grid holds names outside the declared 2x2 vocabulary, each one "
                "enlarging the corrected family",
                expected=list(grid_names), got=unexpected)
        misfiled = [n for n in grid if n in tuple(offgrid_names)]
        log.require("structure.offgrid_name_in_grid", not misfiled, where="grid",
                    detail="an off-grid cell name was scored inside the grid",
                    expected=[], got=misfiled)
        log.require("structure.grid_deltas_present",
                    bool(deltas) or not require_grid_deltas or partial,
                    where="deltas", detail="no grid delta block in the report")
        if require_offgrid:
            log.require("structure.offgrid_present", bool(off_deltas), where="offgrid_deltas",
                        detail="run carries no off-grid before/after block")

    # ---------------------------------------------------------------- family size
    if check_family_size:
        want = max(1, len(grid) - 1) if expect_grid_contrasts is None else expect_grid_contrasts
        log.require("family.grid_size", declared_family == float(want),
                    where="n_contrasts",
                    detail="declared Bonferroni family size does not match the grid",
                    expected=want, got=report.get("n_contrasts"))
        if offgrid or off_deltas:
            if expect_offgrid_contrasts is None:
                log.add("family.offgrid_size", SKIP, where="offgrid_n_contrasts",
                        detail="accepted as declared", got=report.get("offgrid_n_contrasts"))
            elif partial and not off_deltas and declared_off_family is None:
                log.add("family.offgrid_size", SKIP, where="offgrid_n_contrasts",
                        detail=f"{partial_why}; the off-grid block declares its "
                               "divisor when it is written",
                        expected=expect_offgrid_contrasts, got=None)
            else:
                log.require("family.offgrid_size",
                            declared_off_family == float(expect_offgrid_contrasts),
                            where="offgrid_n_contrasts",
                            detail="off-grid block is not corrected the way this run expects",
                            expected=expect_offgrid_contrasts,
                            got=report.get("offgrid_n_contrasts"))
        else:
            log.add("family.offgrid_size", SKIP, where="offgrid_n_contrasts",
                    detail="no off-grid block")

    # ---------------------------------------------------------------- coverage
    if check_contrast_coverage:
        want_keys = ([] if partial and not deltas
                     else [contrast_key(n, ref) for n in grid if n != ref])
        for key in want_keys:
            log.require("coverage.contrast_present", key in deltas, where=key,
                        detail="grid cell has no delta block against the reference")
        extra = [k for k in deltas if k not in want_keys]
        log.require("coverage.no_extra_contrast", not extra, where="deltas",
                    detail="delta block holds a contrast the grid does not license",
                    expected=want_keys, got=extra)
        log.require("coverage.family_matches_contrasts",
                    declared_family == float(max(1, len(deltas))) or not deltas,
                    where="n_contrasts",
                    detail="declared family size does not match the number of contrasts",
                    expected=len(deltas), got=report.get("n_contrasts"))
        for key, block in deltas.items():
            for metric in expect_metrics:
                log.require("coverage.metric_present",
                            isinstance(block, dict) and metric in block,
                            where=f"{key}.{metric}", detail="metric missing from contrast")

    # ---------------------------------------------------------------- per block
    blocks: List[Tuple[str, Dict[str, Any], Optional[float], str]] = []
    for key, block in deltas.items():
        blocks.append((key, block if isinstance(block, dict) else {}, declared_family, "grid"))
    for key, block in off_deltas.items():
        blocks.append((key, block if isinstance(block, dict) else {},
                       declared_off_family, "offgrid"))

    aso_seen = False
    for key, block, divisor, kind in blocks:
        for metric, entry in block.items():
            where = f"{key}.{metric}"

            if _is_cfca_block(entry) and metric == cfca_key:
                if check_cfca_delta:
                    a, b = _num(entry.get("cell")), _num(entry.get("reference"))
                    d = _num(entry.get("delta"))
                    if a is None or b is None or d is None:
                        log.add("cfca.delta_identity", SKIP, where=where,
                                detail="CFCA value missing, nothing to subtract")
                    else:
                        log.require("cfca.delta_identity", abs((a - b) - d) <= cfca_tol,
                                    where=where,
                                    detail="CFCA delta is not cell minus reference",
                                    expected=a - b, got=d)
                if not _is_delta_block(entry):
                    continue
                # a CFCA row that carries an interval is checked as a delta too

            if not _is_delta_block(entry):
                log.add("block.shape", WARN, where=where,
                        detail="entry is neither a paired-delta block nor the CFCA row",
                        got=sorted(entry) if isinstance(entry, dict) else type(entry).__name__)
                continue

            delta = _num(entry.get("delta"))
            low, high = _num(entry.get("ci_low")), _num(entry.get("ci_high"))
            n = _num(entry.get("n"))
            method = entry.get("method")
            perm_p = _num(entry.get("perm_p"))
            corrected = _num(entry.get("perm_p_bonferroni"))
            flag = entry.get("significant")

            if check_structure:
                missing = [k for k in DELTA_KEYS if k not in entry]
                log.require("block.keys", not missing, where=where,
                            detail="paired-delta block is missing keys",
                            expected=list(DELTA_KEYS), got=missing)

            if check_ci_order:
                if delta is None or low is None or high is None:
                    log.add("ci.order", SKIP, where=where,
                            detail="empty contrast, no interval to order")
                else:
                    log.require("ci.order", low - ci_tol <= delta <= high + ci_tol,
                                where=where,
                                detail="point estimate is outside its own CI",
                                expected=f"{low} <= delta <= {high}", got=delta)
                    log.require("ci.low_le_high", low <= high + ci_tol, where=where,
                                detail="CI bounds are inverted", expected=f"<= {high}", got=low)

            if check_significance_flag:
                if low is None or high is None or flag is None:
                    log.add("significance.flag_identity", SKIP, where=where,
                            detail="empty contrast, no flag to check")
                else:
                    want_flag = bool(low > 0 or high < 0)
                    log.require("significance.flag_identity", bool(flag) == want_flag,
                                where=where,
                                detail="significant flag disagrees with its interval",
                                expected=want_flag, got=flag)

            if check_method_names:
                if method is None:
                    log.add("method.name", SKIP, where=where, detail="empty contrast")
                else:
                    log.require("method.name", method in tuple(allowed_methods), where=where,
                                detail="unexpected CI method string",
                                expected=list(allowed_methods), got=method)

            if check_bonferroni:
                if perm_p is None and corrected is None:
                    log.add("bonferroni.identity", SKIP, where=where,
                            detail="no permutation p on this contrast")
                elif perm_p is None or corrected is None or divisor is None:
                    log.require("bonferroni.identity", False, where=where,
                                detail="one of p, corrected p, or the family size is missing",
                                expected="all three present",
                                got={"perm_p": perm_p, "perm_p_bonferroni": corrected,
                                     "n_contrasts": divisor})
                else:
                    log.require("bonferroni.p_range", 0.0 <= perm_p <= 1.0, where=where,
                                detail="permutation p is outside [0, 1]", got=perm_p)
                    log.require("bonferroni.corrected_range", 0.0 <= corrected <= 1.0,
                                where=where, detail="corrected p is outside [0, 1]",
                                got=corrected)
                    want_p = bonferroni(perm_p, int(divisor))
                    ok = want_p is not None and abs(want_p - corrected) <= p_tol
                    implied = None
                    if not ok and perm_p > 0:
                        implied = round(corrected / perm_p, 4)
                    log.require("bonferroni.identity", ok, where=where,
                                detail=(f"corrected p is not min(1, p * {int(divisor)})"
                                        + (f"; the stored pair implies a divisor of {implied}"
                                           if implied is not None else "")),
                                expected=want_p, got=corrected)
                    log.require("bonferroni.monotone", corrected + p_tol >= perm_p,
                                where=where,
                                detail="correction made the p-value smaller",
                                expected=f">= {perm_p}", got=corrected)
                    if kind == "offgrid" and expect_offgrid_contrasts == 1:
                        log.require("bonferroni.offgrid_uncorrected",
                                    abs(corrected - min(1.0, perm_p)) <= p_tol, where=where,
                                    detail="off-grid before/after carries a correction, so it "
                                           "is being read as one of the scored contrasts",
                                    expected=perm_p, got=corrected)

            if check_pairing_n:
                cell_name = key.split("_vs_")[0]
                pair_ref = off_ref if kind == "offgrid" else ref
                sizes = [
                    _num((cells.get(c) or {}).get("n_queries"))
                    for c in (cell_name, pair_ref)
                ]
                sizes = [s for s in sizes if s is not None]
                if n is None or not sizes:
                    log.add("pairing.n", SKIP, where=where, detail="cell sizes unavailable")
                else:
                    log.require("pairing.n", 0 <= n <= min(sizes), where=where,
                                detail="paired n exceeds the smaller cell",
                                expected=f"<= {min(sizes)}", got=n)

            if check_bounds and metric in tuple(bounded_columns):
                if delta is None:
                    log.add("bounds.delta", SKIP, where=where, detail="empty contrast")
                else:
                    log.require("bounds.delta", -1.0 - bound_tol <= delta <= 1.0 + bound_tol,
                                where=where,
                                detail="delta on a 0/1 column is outside [-1, 1]", got=delta)

            if check_decision_agreement:
                if flag is None or corrected is None:
                    log.add("decision.ci_vs_p", SKIP, where=where,
                            detail="nothing to cross-read")
                elif bool(flag) and corrected > alpha:
                    log.add("decision.ci_vs_p", WARN, where=where,
                            detail=(f"CI excludes 0 but the corrected p is {corrected} "
                                    f"> alpha {alpha}, so the family-corrected test does "
                                    f"not reject the null; report both or neither"),
                            expected=f"p <= {alpha}", got=corrected)
                elif not bool(flag) and corrected <= alpha:
                    log.add("decision.ci_vs_p", WARN, where=where,
                            detail=(f"corrected p is {corrected} <= alpha {alpha} but the "
                                    f"CI spans 0, so the interval does not support the "
                                    f"rejection"),
                            expected=f"p > {alpha}", got=corrected)
                else:
                    log.add("decision.ci_vs_p", PASS, where=where)

            if entry.get("aso") is not None:
                aso_seen = True
                aso = _num(entry.get("aso"))
                log.require("aso.range", aso is not None and 0.0 <= aso <= 1.0, where=where,
                            detail="ASO violation ratio is outside [0, 1]", got=entry.get("aso"))

    # ---------------------------------------------------------------- per cell
    if check_bounds:
        for name, cell in cells.items():
            cis = (cell or {}).get("confidence_intervals") or {}
            n_queries = _num((cell or {}).get("n_queries"))
            for metric, ci in cis.items():
                where = f"cells.{name}.{metric}"
                mean = _num((ci or {}).get("mean"))
                low, high = _num((ci or {}).get("ci_low")), _num((ci or {}).get("ci_high"))
                n = _num((ci or {}).get("n"))
                if mean is None or low is None or high is None:
                    log.add("cell.ci_order", SKIP, where=where, detail="empty column")
                else:
                    log.require("cell.ci_order", low - ci_tol <= mean <= high + ci_tol,
                                where=where, detail="cell mean is outside its own CI",
                                expected=f"{low} <= mean <= {high}", got=mean)
                    if metric in tuple(bounded_columns):
                        log.require("cell.bounds",
                                    -bound_tol <= low and high <= 1.0 + bound_tol,
                                    where=where,
                                    detail="CI on a 0/1 column leaves [0, 1]",
                                    got=[low, high])
                if n is not None and n_queries is not None:
                    log.require("cell.n", n <= n_queries, where=where,
                                detail="scored column has more rows than the cell has queries",
                                expected=f"<= {n_queries}", got=n)

    # ---------------------------------------------------------------- isolation
    if check_offgrid_isolation:
        if not offgrid and not off_deltas:
            log.add("isolation.offgrid", SKIP, where="offgrid", detail="no off-grid cell")
        elif partial and not off_deltas:
            log.add("isolation.offgrid", SKIP, where="offgrid",
                    detail="no off-grid delta block in a partial report")
        else:
            leaked = [k for k in deltas if any(n in k.split("_vs_") for n in offgrid)]
            log.require("isolation.not_in_grid_deltas", not leaked, where="deltas",
                        detail="an off-grid cell has a delta inside the corrected block",
                        expected=[], got=leaked)
            log.require("isolation.offgrid_reference_scored", off_ref in cells,
                        where=f"offgrid_reference={off_ref}",
                        detail="off-grid deltas name a reference that was never scored")
            log.require("isolation.family_unchanged_by_offgrid",
                        declared_family == float(max(1, len(grid) - 1))
                        if expect_grid_contrasts is None else True,
                        where="n_contrasts",
                        detail="off-grid cells appear to have entered the family count",
                        expected=max(1, len(grid) - 1), got=report.get("n_contrasts"))
            for key in off_deltas:
                cell_name = key.split("_vs_")[0]
                log.require("isolation.offgrid_contrast_names_offgrid_cell",
                            cell_name in offgrid, where=key,
                            detail="off-grid block holds a contrast between two grid cells",
                            expected=offgrid, got=cell_name)

    # ------------------------------------------------------------- GBP + pool
    if check_cfca_gbp:
        sub = dict(cfca_gbp_kwargs or {})
        sub.pop("log", None)
        check_cfca_gbp_block(report, log=log, **sub)
    if check_warm_only:
        # the warm-only sensitivity is the same table on other seconds, so it
        # gets the same arithmetic check; absent, it is a SKIP and not a FAIL,
        # because a report is complete without it
        sub = dict(cfca_gbp_kwargs or {})
        sub.update(warm_only_kwargs or {})
        sub.pop("log", None)
        sub["block_key"] = warm_only_key
        sub.setdefault("require_block", False)
        check_cfca_gbp_block(report, log=log, **sub)
    if check_ir_pool:
        sub = dict(ir_pool_kwargs or {})
        sub.pop("log", None)
        # widths must agree across the cells whose deltas are read against each
        # other; the off-grid cell ran another configuration by design
        sub.setdefault("compare_names", grid if partial else (grid or None))
        check_ir_pool_block(report, log=log, **sub)

    # ---------------------------------------------------------------- ASO divisor
    if check_aso_divisor:
        if not aso_seen:
            log.add("aso.divisor_matches_family", SKIP, where="aso",
                    detail="ASO score produced nothing (deepsig off or not installed)")
        elif declared_family is not None and float(aso_num_comparisons) != declared_family:
            log.add("aso.divisor_matches_family", WARN, where="aso",
                    detail=(f"the ASO score corrects for {aso_num_comparisons} comparisons "
                            f"while the permutation corrects for "
                            f"{int(declared_family)}; the two are answering "
                            f"different questions until one family is chosen"),
                    expected=int(declared_family), got=aso_num_comparisons)
        else:
            log.add("aso.divisor_matches_family", PASS, where="aso")

    counts = {s: sum(1 for r in log.rows if r["status"] == s) for s in STATUS_ORDER}
    blocking = tuple(fail_on)
    return {
        "ok": not any(counts.get(s, 0) for s in blocking),
        "counts": counts,
        "alpha": alpha,
        "reference": ref,
        "grid": grid,
        "offgrid": offgrid,
        "n_contrasts": report.get("n_contrasts"),
        "offgrid_n_contrasts": report.get("offgrid_n_contrasts"),
        "failed": [r for r in log.rows if r["status"] == FAIL],
        "warned": [r for r in log.rows if r["status"] == WARN],
        "checks": log.rows,
    }


def verify_report_file(
    path: str,
    *,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> Dict[str, Any]:
    """`verify_report` on a written report.json; kwargs pass straight through.

    Given the ``.md`` twin, reads the ``report.json`` written beside it: the
    markdown is a rendering, the json holds the numbers being re-derived.
    """
    if path.lower().endswith(".md"):
        twin = os.path.splitext(path)[0] + ".json"
        if os.path.exists(twin):
            path = twin
    with open(path, "r", encoding=encoding) as fh:
        report = json.load(fh)
    result = verify_report(report, **kwargs)
    result["path"] = path
    return result


def render_verify_text(
    result: Dict[str, Any],
    *,
    show: Sequence[str] = (FAIL, WARN),
    max_rows: Optional[int] = 40,
    show_counts: bool = True,
    show_header: bool = True,
    indent: str = "  ",
) -> str:
    """
    Plain-text check report.

    Args:
        show: statuses listed row by row; add "pass" for the full trace.
        max_rows: cap on listed rows; None prints all of them.
        show_counts: print the per-status tally line.
        show_header: print the reference cell and both family sizes.
        indent: prefix for listed rows.
    """
    lines: List[str] = []
    if show_header:
        lines.append(
            f"report math check: reference={result.get('reference')} "
            f"grid={result.get('grid')} n_contrasts={result.get('n_contrasts')} "
            f"offgrid={result.get('offgrid')} "
            f"offgrid_n_contrasts={result.get('offgrid_n_contrasts')} "
            f"alpha={result.get('alpha')}"
        )
    if show_counts:
        counts = result.get("counts") or {}
        lines.append("  ".join(f"{s}={counts.get(s, 0)}" for s in STATUS_ORDER))

    rows = [r for r in result.get("checks", []) if r["status"] in tuple(show)]
    shown = rows if max_rows is None else rows[:max_rows]
    for r in shown:
        bits = [f"{indent}[{r['status'].upper()}] {r['check']}"]
        if r.get("where"):
            bits.append(f"at {r['where']}")
        if r.get("detail"):
            bits.append(f"- {r['detail']}")
        if r.get("expected") is not None:
            bits.append(f"(expected {r['expected']!r}, got {r.get('got')!r})")
        lines.append(" ".join(bits))
    if max_rows is not None and len(rows) > len(shown):
        lines.append(f"{indent}... {len(rows) - len(shown)} more")
    if not rows:
        lines.append(f"{indent}every check passed")
    return "\n".join(lines)
