"""
Report assembly: per-cell metrics + cross-cell paired deltas with CIs.

Consumes the per-query 0/1 columns emitted by every metric module and runs the
scipy helpers in stats.py to produce, for the 2x2:
  * per-cell BCa CIs on every metric column
  * paired deltas (each other cell vs the control cell) with BCa CIs, the main
    uncertainty figures for RQ1/RQ2 (significant when both CI ends one side of 0)
  * CFCA per cell and its delta vs the control cell

`evaluate_experiment_split` additionally measures the declared off-grid windows
(`prebaseline`) against the reference and reports that before/after delta in its
own uncorrected block, so the one-time baseline construction is sized without
entering the 2x2 table or its Bonferroni family.

Nothing is re-run; every number comes from the logs the 2x2 already produced.
"""
from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence

from .abstention import score_abstention
from .cells import (
    DEFAULT_REFERENCE,
    GRID_ORDER,
    OFFGRID_ORDER,
    contrast_key,
    deltas_key,
    deltas_of,
    offgrid_deltas_key,
    offgrid_deltas_of,
    split_grid,
)
from .cfca_metric import compute_cfca, cost_overview
from .config import DEFAULT_RUN_ID, settings
from .extract import QueryRecord
from .faithfulness import RAGA_FIELDS, score_faithfulness
from .retrieval_eval import evaluate_retrieval, served_pool_summary
from .stats import (
    aso_significance,
    bonferroni,
    bootstrap_ci,
    paired_delta_ci,
    paired_permutation_p,
    paired_ratio_delta_ci,
    paired_ratio_permutation_p,
)


def _latency(records: List[QueryRecord]) -> Dict:
    def pct(vals, p):
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        k = max(0, min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1)))))
        return vals[k]

    e2e = [r.latency_ms for r in records if r.latency_ms is not None]
    retr = [r.retrieval_latency_ms for r in records if r.retrieval_latency_ms is not None]
    rerank = [r.rerank_latency_ms for r in records if r.rerank_latency_ms is not None]
    return {
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p95_ms": pct(e2e, 95),
        "retrieval_p50_ms": pct(retr, 50),
        "rerank_p50_ms": pct(rerank, 50),
        "e2e_mean_ms": round(statistics.mean(e2e), 1) if e2e else None,
    }


def evaluate_cell(
    records: List[QueryRecord],
    per_answer_cost: Optional[float] = None,
    *,
    per_answer_cost_by_query: Optional[Dict[str, float]] = None,
    ir_pool_fn: Optional[Callable[..., Dict]] = served_pool_summary,
    ir_pool_key: str = "ir_pool",
    ir_pool_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict:
    """All per-cell metrics for one condition.

    per_answer_cost / per_answer_cost_by_query: the cost numerator in GBP, flat
    or per question for a run where one question costs more to answer than
    another; see `compute_cfca`.

    ir_pool_fn / ir_pool_key / ir_pool_kwargs: the served-pool summary that
    records the logged `top_n` beside the k this cell was scored at, so
    `verify.check_ir_pool_block` can hold the logs-only cap to its claim. None
    omits the block and the check then skips.
    """
    retrieval = evaluate_retrieval(records)
    faith = score_faithfulness(records)
    abst = score_abstention(records)
    cfca = compute_cfca(
        records, faith.get("per_query", {}),
        per_answer_cost=per_answer_cost,
        per_answer_cost_by_query=per_answer_cost_by_query,
    )
    cost = cost_overview(records)

    linked = [r for r in records if r.linked]

    # unified per-query columns for stats.py
    columns: Dict[str, Dict[str, float]] = {}
    for m, colmap in retrieval["per_query"].items():
        columns[m] = colmap
    columns["faithful"] = {q: v["faithful"] for q, v in faith.get("per_query", {}).items()}
    columns["abstention_recall"] = abst["per_query"]["abstention_recall"]
    columns["over_abstention_rate"] = abst["per_query"]["over_abstention_rate"]
    columns["cfca_joint_P"] = cfca["per_query"]["joint"]
    columns["cfca_cost"] = cfca["per_query"]["cost"]

    per_cell_ci = {m: bootstrap_ci(list(col.values())) for m, col in columns.items()}

    out = {
        "n_queries": len(records),
        "n_linked": len(linked),
        "n_unlinked": len(records) - len(linked),
        "retrieval": {k: v for k, v in retrieval.items() if k != "per_query"},
        # per_query stays for faithfulness alone: the strict faithful flag is
        # what P_hat and CFCA divide by, so the per-answer spans and the window
        # read behind that flag have to be on disk to be argued with.
        "faithfulness": faith,
        "abstention": {k: v for k, v in abst.items() if k != "per_query"},
        "cfca": {k: v for k, v in cfca.items() if k != "per_query"},
        "cost": cost,
        "latency": _latency(linked),
        "confidence_intervals": per_cell_ci,
        "_columns": columns,   # kept for cross-cell deltas; stripped from final report
    }
    if ir_pool_fn is not None:
        pool_kwargs = dict(ir_pool_kwargs or {})
        pool_kwargs.setdefault("k", retrieval.get("k"))
        out[ir_pool_key] = ir_pool_fn(records, **pool_kwargs)
    return out


def group_runs(
    records: Sequence[QueryRecord],
    *,
    run_field: str = "run_id",
    default_run: str = DEFAULT_RUN_ID,
    run_order: Optional[Sequence[str]] = None,
    only: Optional[Sequence[str]] = None,
    skip: Sequence[str] = (),
) -> Dict[str, List[QueryRecord]]:
    """Split one cell's records into ``{run_id: records}``, in first-seen order.

    Args:
        run_field: attribute carrying the run id; a record without one lands in
            ``default_run``.
        run_order: force the key order.
        only / skip: restrict which runs come back, by id.
    """
    out: Dict[str, List[QueryRecord]] = {}
    for rec in records:
        rid = str(getattr(rec, run_field, None) or default_run)
        out.setdefault(rid, []).append(rec)
    if only is not None:
        wanted = {str(r) for r in only}
        out = {k: v for k, v in out.items() if k in wanted}
    if skip:
        dropped = {str(r) for r in skip}
        out = {k: v for k, v in out.items() if k not in dropped}
    if run_order:
        ordered = {str(r): out[str(r)] for r in run_order if str(r) in out}
        for k, v in out.items():
            ordered.setdefault(k, v)
        return ordered
    return out


def pool_run_columns(
    run_columns: Dict[str, Dict[str, Dict[str, float]]],
    *,
    metrics: Optional[Sequence[str]] = None,
    stat: str = "mean",
    min_runs_per_query: int = 1,
    require_same_queries: bool = False,
    round_to: Optional[int] = 6,
    exact_columns: Sequence[str] = ("cfca_cost",),
) -> Dict[str, Dict[str, float]]:
    """Average each per-question column across runs, so a cell keeps one row
    per question however often it was repeated.

    Args:
        run_columns: ``{run_id: {metric: {query_id: value}}}``.
        metrics: restrict to these columns; None pools every column present.
        stat: ``mean`` | ``median`` | ``min`` | ``max`` over the runs.
        min_runs_per_query: drop a question answered in fewer runs than this.
        require_same_queries: raise when the runs cover different questions.
        round_to: decimals on the pooled value; None keeps full precision.
        exact_columns: columns that keep full precision whatever `round_to`
            says, because a money column loses its cheap answers at 6 decimals.
    """
    pick = {
        "mean": lambda vals: sum(vals) / len(vals),
        "median": statistics.median,
        "min": min,
        "max": max,
    }
    if stat not in pick:
        raise ValueError(f"stat must be one of {sorted(pick)}, got {stat!r}")
    names = list(metrics) if metrics is not None else list(
        dict.fromkeys(m for cols in run_columns.values() for m in cols)
    )
    if require_same_queries:
        seen = {rid: {q for col in cols.values() for q in col} for rid, cols in run_columns.items()}
        first = next(iter(seen.values()), set())
        for rid, qs in seen.items():
            if qs != first:
                raise ValueError(
                    f"run {rid!r} covers {len(qs)} query ids, the first run covers {len(first)}"
                )
    exact = tuple(exact_columns)
    pooled: Dict[str, Dict[str, float]] = {}
    for m in names:
        gathered: Dict[str, List[float]] = {}
        for cols in run_columns.values():
            for qid, val in (cols.get(m) or {}).items():
                if val is None:
                    continue
                gathered.setdefault(qid, []).append(float(val))
        col: Dict[str, float] = {}
        for qid, vals in gathered.items():
            if len(vals) < min_runs_per_query:
                continue
            v = pick[stat](vals)
            keep_exact = round_to is None or m in exact
            col[qid] = v if keep_exact else round(v, round_to)
        pooled[m] = col
    return pooled


def _pool_numbers(
    blocks: Sequence[Dict],
    *,
    skip: Sequence[str] = (),
    sum_keys: Sequence[str] = (),
    depth: int = 2,
) -> Dict:
    """Mean of the numeric leaves across per-run blocks, first value otherwise."""
    if not blocks:
        return {}
    out: Dict = {}
    for key, first in blocks[0].items():
        if key in skip:
            continue
        vals = [b.get(key) for b in blocks if key in b]
        if all(isinstance(v, bool) for v in vals):
            out[key] = any(vals)
        elif all(isinstance(v, (int, float)) and v is not None for v in vals):
            total = sum(float(v) for v in vals)
            if key in sum_keys:
                out[key] = int(total) if all(isinstance(v, int) for v in vals) else total
            else:
                out[key] = round(total / len(vals), 6)
        elif depth > 1 and all(isinstance(v, dict) for v in vals):
            out[key] = _pool_numbers(vals, skip=skip, sum_keys=sum_keys, depth=depth - 1)
        else:
            out[key] = first
    return out


def evaluate_cell_runs(
    records: Sequence[QueryRecord],
    per_answer_cost: Optional[float] = None,
    *,
    cell_fn: Callable[..., Dict] = evaluate_cell,
    run_field: str = "run_id",
    default_run: str = DEFAULT_RUN_ID,
    run_order: Optional[Sequence[str]] = None,
    only_runs: Optional[Sequence[str]] = None,
    skip_runs: Sequence[str] = (),
    stat: str = "mean",
    min_runs_per_query: int = 1,
    require_same_queries: bool = False,
    single_run_verbatim: bool = True,
    keep_run_blocks: bool = True,
    keep_run_columns: bool = False,
    run_summary_keys: Sequence[str] = ("n_queries", "n_linked", "n_unlinked"),
    round_to: Optional[int] = 6,
    exact_columns: Sequence[str] = ("cfca_cost",),
    per_answer_cost_by_query: Optional[Dict[str, float]] = None,
    ir_pool_fn: Optional[Callable[..., Dict]] = served_pool_summary,
    ir_pool_key: str = "ir_pool",
    ir_pool_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict:
    """One cell's metrics with its repeats averaged question by question.

    N runs are scored N times, then collapsed: each per-question column is the
    mean over the runs that answered it, aggregates and CIs come off those
    pooled columns, cost and latency stay pooled over all N x Q answers.

    Args:
        cell_fn: the single-run scorer, injectable.
        run_field / default_run: where the run id lives, and the fallback id.
        run_order / only_runs / skip_runs: which repeats to score, in what order.
        stat: how the runs collapse per question.
        min_runs_per_query / require_same_queries: see `pool_run_columns`.
        single_run_verbatim: a one-run cell bypasses pooling, so its numbers
            cannot move by a rounding step.
        keep_run_blocks / keep_run_columns / run_summary_keys: what each run
            leaves under ``runs``.
        round_to: decimals on pooled per-question values.
        exact_columns: columns pooled at full precision; see `pool_run_columns`.
        per_answer_cost_by_query: per-question serving cost, passed to the cell
            scorer only when given, so an injected `cell_fn` keeps its signature.
        ir_pool_fn / ir_pool_key / ir_pool_kwargs: the served-pool summary,
            recomputed over every repeat at once because the logged `top_n`
            belongs to the answers and not to the pooling. None keeps whatever
            the first run's block held, and nothing when no run held one.
    """
    recs = list(records)
    extra = ({"per_answer_cost_by_query": per_answer_cost_by_query}
             if per_answer_cost_by_query else {})
    groups = group_runs(
        recs, run_field=run_field, default_run=default_run,
        run_order=run_order, only=only_runs, skip=skip_runs,
    )
    if not groups:
        groups = {default_run: []}
    run_ids = list(groups)

    if len(run_ids) == 1 and single_run_verbatim:
        out = cell_fn(groups[run_ids[0]], per_answer_cost, **extra)
        out["n_runs"] = 1
        out["run_ids"] = run_ids
        out["n_answers"] = out.get("n_queries")
        out["n_questions"] = len({r.query_id for r in groups[run_ids[0]]})
        return out

    blocks = {rid: cell_fn(rows, per_answer_cost, **extra) for rid, rows in groups.items()}
    columns = pool_run_columns(
        {rid: b.get("_columns", {}) for rid, b in blocks.items()},
        stat=stat,
        min_runs_per_query=min_runs_per_query,
        require_same_queries=require_same_queries,
        round_to=round_to,
        exact_columns=exact_columns,
    )

    def col_mean(name: str) -> Optional[float]:
        vals = list(columns.get(name, {}).values())
        return round(sum(vals) / len(vals), 6) if vals else None

    per_run = list(blocks.values())
    linked = [r for r in recs if r.linked]

    retrieval = _pool_numbers([b["retrieval"] for b in per_run], skip=("n_judged",))
    retrieval["n_judged"] = len(columns.get(f"recall@{settings.ir_k}", {}))
    for m in list(retrieval.get("aggregate") or {}):
        if m in columns:
            retrieval["aggregate"][m] = col_mean(m)

    # depth 3 reaches langmet_summary -> scores / evaluation_counts leaves, so a
    # repeated cell averages its RAGA scores instead of keeping the first run's.
    faith = _pool_numbers(
        [b["faithfulness"] for b in per_run],
        skip=("n_scored",),
        depth=3,
    )
    faith["n_scored"] = len(columns.get("faithful", {}))
    if isinstance(faith.get("aggregate"), dict) and "faithful_rate" in faith["aggregate"]:
        faith["aggregate"]["faithful_rate"] = col_mean("faithful")

    abst = _pool_numbers([b["abstention"] for b in per_run])
    for m in ("abstention_recall", "over_abstention_rate"):
        if isinstance(abst.get("aggregate"), dict) and m in abst["aggregate"]:
            abst["aggregate"][m] = col_mean(m)

    cfca = _pool_numbers([b["cfca"] for b in per_run], skip=("n_answerable", "cfca"))
    cfca["n_answerable"] = len(columns.get("cfca_joint_P", {}))
    p_joint = col_mean("cfca_joint_P")
    cfca["p_faithful_cited_version"] = round(p_joint, 4) if p_joint is not None else None
    cost_num = cfca.get("cost_per_answer_gbp")
    if p_joint in (None, 0.0) or cost_num is None:
        cfca["cfca"] = None
        cfca["note"] = "P(faithful·cited·right-version) = 0 or undefined; CFCA undefined."
    else:
        cfca["cfca"] = round(float(cost_num) / p_joint, 8)
        cfca["note"] = None

    out = {
        "n_queries": len(recs),
        "n_answers": len(recs),
        "n_questions": len({r.query_id for r in recs}),
        "n_linked": len(linked),
        "n_unlinked": len(recs) - len(linked),
        "n_runs": len(run_ids),
        "run_ids": run_ids,
        "pooled_by": stat,
        "retrieval": retrieval,
        "faithfulness": faith,
        "abstention": abst,
        "cfca": cfca,
        "cost": cost_overview(recs),
        "latency": _latency(linked),
        "confidence_intervals": {m: bootstrap_ci(list(col.values())) for m, col in columns.items()},
        "_columns": columns,
    }
    # The served pool is a property of the answers, not of the pooling, so it is
    # recomputed over every repeat at once; the min/max then cover all N runs.
    if ir_pool_fn is not None:
        pool_kwargs = dict(ir_pool_kwargs or {})
        pool_kwargs.setdefault("k", retrieval.get("k"))
        out[ir_pool_key] = ir_pool_fn(recs, **pool_kwargs)
    elif per_run and all(isinstance(b.get(ir_pool_key), dict) for b in per_run):
        out[ir_pool_key] = per_run[0][ir_pool_key]
    if keep_run_blocks:
        runs_out: Dict[str, Dict] = {}
        for rid, b in blocks.items():
            entry = {k: b.get(k) for k in run_summary_keys}
            entry["retrieval"] = (b["retrieval"].get("aggregate") or {})
            entry["faithful_rate"] = (b["faithfulness"].get("aggregate") or {}).get("faithful_rate")
            entry["cfca"] = b["cfca"].get("cfca")
            if keep_run_columns:
                entry["_columns"] = b.get("_columns", {})
            runs_out[rid] = entry
        out["runs"] = runs_out
    return out


# metric columns that get paired deltas vs the reference cell
_DELTA_METRICS = [
    f"recall@{settings.ir_k}",
    f"mrr@{settings.ir_k}",
    f"ndcg@{settings.ir_k}",
    f"hit_rate@{settings.ir_k}",
    "faithful",
    "abstention_recall",
    "over_abstention_rate",
    "cfca_joint_P",
]


def paired_deltas(
    columns: Dict[str, Dict[str, float]],
    ref_columns: Dict[str, Dict[str, float]],
    *,
    metrics: Sequence[str] = tuple(_DELTA_METRICS),
    n_contrasts: int = 1,
    cfca: Optional[float] = None,
    ref_cfca: Optional[float] = None,
    with_perm: bool = True,
    with_aso: bool = True,
    cfca_key: str = "cfca",
) -> Dict:
    """
    One cell's paired deltas against a reference cell's per-query columns.

    Args:
        metrics: columns that get a BCa paired delta.
        n_contrasts: Bonferroni family size for ``perm_p_bonferroni`` and for the
            ASO score, so one family covers both; 1 leaves the permutation p
            uncorrected, which is what an off-grid validity check reports.
        cfca / ref_cfca: aggregate CFCA values. The row written here is the
            plain subtraction; `cfca_delta_block` replaces it with the same
            numbers plus a paired interval when both cost columns are present.
        with_perm / with_aso: drop the permutation or the ASO score.
        cfca_key: field the CFCA delta is written under; "" omits it.
    """
    out: Dict[str, Dict] = {}
    for m in metrics:
        a = columns.get(m, {})
        b = ref_columns.get(m, {})
        d = paired_delta_ci(a, b)
        if with_perm:
            perm_p = paired_permutation_p(a, b)
            d["perm_p"] = round(perm_p, 6) if perm_p is not None else None
            d["perm_p_bonferroni"] = (
                round(bonferroni(perm_p, n_contrasts), 6) if perm_p is not None else None
            )
        if with_aso:
            d["aso"] = aso_significance(a, b, num_comparisons=n_contrasts)
        out[m] = d
    if cfca_key:
        out[cfca_key] = {
            "cell": cfca,
            "reference": ref_cfca,
            "delta": _safe_sub(cfca, ref_cfca),
        }
    return out


def cfca_delta_block(
    columns: Dict[str, Dict[str, float]],
    ref_columns: Dict[str, Dict[str, float]],
    *,
    cfca: Optional[float] = None,
    ref_cfca: Optional[float] = None,
    n_contrasts: int = 1,
    cost_column: str = "cfca_cost",
    joint_column: str = "cfca_joint_P",
    with_perm: bool = True,
    on_zero_denominator: str = "warn",
    warn: Optional[Callable[[str], None]] = None,
    delta_tol: float = 1e-6,
    round_to: int = 8,
    n_resamples: Optional[int] = None,
    confidence_level: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    The CFCA row with a paired interval instead of a bare subtraction.

    CFCA is a ratio of two per-question means, cost over the all-pass rate, so
    resampling questions once and re-deriving both ratios gives it the same kind
    of interval every other metric row carries. RQ2 and
    H3 are read off this interval, and the headline number stops being the only
    quantity in the report with no uncertainty on it.

    Args:
        cfca / ref_cfca: the cell-level CFCA scalars. They stay the reported
            cell, reference and delta, so this row still reconciles with the
            per-cell table; the bootstrap adds the interval and the p-value.
        cost_column / joint_column: per-question numerator and denominator.
        n_contrasts: Bonferroni family size for the permutation p; 1 leaves it
            uncorrected, which is what the off-grid validity block reports.
        with_perm: drop the permutation.
        on_zero_denominator: "warn" | "raise" | "ignore" for resample draws
            where no answer passes both gates; see `paired_ratio_delta_ci`.
        warn: sink for that warning, usually the run printer.
        delta_tol: gap between the reported delta and the resampled point
            estimate that earns a note. The two separate only when a cell's
            repeats answered different questions, which reweights the pooling.
        round_to: decimals on the emitted numbers.
        n_resamples / confidence_level / seed: bootstrap knobs; None takes the
            pinned settings.

    ASO is absent by design: it orders two samples of per-item scores, and a
    ratio of means has no per-item score to order.
    """
    plain: Dict = {"cell": cfca, "reference": ref_cfca,
                   "delta": _safe_sub(cfca, ref_cfca)}
    a_cost, a_joint = columns.get(cost_column) or {}, columns.get(joint_column) or {}
    b_cost = ref_columns.get(cost_column) or {}
    b_joint = ref_columns.get(joint_column) or {}
    if not (a_cost and a_joint and b_cost and b_joint):
        plain["note"] = (f"no {cost_column} / {joint_column} pair on both cells, "
                         "so the delta stays a plain subtraction")
        return plain

    res = paired_ratio_delta_ci(
        a_cost, a_joint, b_cost, b_joint,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        on_zero_denominator=on_zero_denominator,
        warn=warn,
        round_to=round_to,
    )
    notes = [res["note"]] if res.get("note") else []
    out: Dict = {**plain, **{k: v for k, v in res.items()
                             if k not in ("delta", "note")}}
    out["delta_resampled"] = res["delta"]

    reported = plain["delta"]
    if res["method"] == "degenerate":
        # no spread: the interval is the point, and the point is the one reported
        out["ci_low"] = out["ci_high"] = reported
        out["significant"] = None if reported is None else bool(reported != 0.0)
    if reported is not None and res["delta"] is not None:
        gap = abs(reported - res["delta"])
        if gap > delta_tol:
            notes.append(
                f"the cell scalars and the pooled columns differ by "
                f"{round(gap, round_to)}, which happens when a cell's repeats "
                f"answered different questions"
            )
    if with_perm:
        perm_p = paired_ratio_permutation_p(
            a_cost, a_joint, b_cost, b_joint, seed=seed
        )
        out["perm_p"] = round(perm_p, 6) if perm_p is not None else None
        out["perm_p_bonferroni"] = (
            round(bonferroni(perm_p, n_contrasts), 6) if perm_p is not None else None
        )
    out["note"] = "; ".join(notes) or None
    return out


def evaluate_experiment_split(
    extracted: Dict[str, List[QueryRecord]],
    *,
    reference: str = DEFAULT_REFERENCE,
    per_answer_cost: Optional[Dict[str, float]] = None,
    grid_names: Sequence[str] = GRID_ORDER,
    offgrid_names: Sequence[str] = OFFGRID_ORDER,
    unknown: str = "grid",
    metrics: Sequence[str] = tuple(_DELTA_METRICS),
    n_contrasts: Optional[int] = None,
    offgrid_reference: Optional[str] = None,
    offgrid_n_contrasts: int = 1,
    min_offgrid_linked: int = 1,
    include_offgrid_cells: bool = True,
    require_offgrid: bool = False,
    deltas_field: Optional[str] = None,
    offgrid_field: Optional[str] = None,
    cell_fn: Callable[..., Dict] = evaluate_cell_runs,
    warn_unequal_runs: bool = True,
    printer: Optional[Callable[[str], None]] = print,
) -> Dict:
    """
    Full report with the scored 2x2 kept apart from the off-grid cells.

    The grid is scored exactly as `evaluate_experiment` scores it, with the
    Bonferroni family taken over the grid alone, so declaring a ``prebaseline``
    window cannot widen a single grid CI or add a row to the result table. The
    off-grid cells get their own block against the same reference, uncorrected
    by default: that is the before/after check on the one-time baseline
    construction, a validity prerequisite and not a
    measured result.

    Args:
        grid_names / offgrid_names / unknown: passed to `cells.split_grid`;
            ``unknown`` keeps any other vocabulary scored as before.
        metrics: per-query columns that get paired deltas.
        n_contrasts: override the grid family size; None uses len(grid) - 1.
        offgrid_reference: cell the off-grid deltas are measured against; None
            uses the grid reference, which is the before/after pairing.
        offgrid_n_contrasts: family size for the off-grid permutation p; 1
            leaves it uncorrected. Raise it only to fold the check into the
            corrected family, which changes what the grid CIs mean.
        min_offgrid_linked: an off-grid cell with fewer linked answers than this
            is still listed under ``cells`` but gets no delta block, so a window
            nobody ran prints nothing instead of a delta of nothing.
        include_offgrid_cells: keep off-grid cells in ``cells``; False drops them
            after their deltas are computed.
        require_offgrid: raise when no off-grid cell is present, for a run that
            must carry the before/after check.
        deltas_field / offgrid_field: report keys for the two blocks; None uses
            `cells.deltas_key` / `cells.offgrid_deltas_key`.
        cell_fn: per-cell scorer. The default averages repeats per question;
            `evaluate_cell` scores the records as one flat set instead.
        warn_unequal_runs: print a line when grid cells carry different run
            counts. Deltas stay paired by question either way.
        printer: where that warning goes; None silences it.
    """
    per_answer_cost = per_answer_cost or {}
    grid, offgrid = split_grid(
        extracted.keys(),
        grid_names=grid_names,
        offgrid_names=offgrid_names,
        unknown=unknown,
        reference=reference,
    )
    if reference not in grid:
        raise ValueError(f"reference cell '{reference}' not among grid conditions {grid}")
    if require_offgrid and not offgrid:
        raise ValueError(
            f"no off-grid cell among {list(extracted)}; expected one of {list(offgrid_names)}"
        )

    cells = {
        name: cell_fn(extracted[name], per_answer_cost.get(name))
        for name in list(grid) + list(offgrid)
    }
    run_counts = {n: int(cells[n].get("n_runs") or 1) for n in cells}
    if warn_unequal_runs and printer and len(set(run_counts[n] for n in grid)) > 1:
        spread = ", ".join(f"{n}={run_counts[n]}" for n in grid)
        printer(
            f"  warn: unequal run counts across the grid ({spread}); every delta is "
            "still paired per question, but the cells are averaged over different "
            "numbers of sittings"
        )
    family = max(1, len(grid) - 1) if n_contrasts is None else n_contrasts
    ref_cols = cells[reference]["_columns"]
    ref_cfca = cells[reference]["cfca"].get("cfca")

    deltas: Dict[str, Dict] = {}
    for name in grid:
        if name == reference:
            continue
        cell_cols = cells[name]["_columns"]
        cell_cfca = cells[name]["cfca"].get("cfca")
        row = paired_deltas(
            cell_cols, ref_cols,
            metrics=metrics, n_contrasts=family,
            cfca=cell_cfca, ref_cfca=ref_cfca,
        )
        row["cfca"] = cfca_delta_block(
            cell_cols, ref_cols,
            cfca=cell_cfca, ref_cfca=ref_cfca,
            n_contrasts=family, warn=printer,
        )
        deltas[contrast_key(name, reference)] = row

    off_ref = offgrid_reference or reference
    if offgrid and off_ref not in cells:
        raise ValueError(f"off-grid reference '{off_ref}' not among conditions {list(cells)}")
    off_deltas: Dict[str, Dict] = {}
    for name in offgrid:
        if cells[name]["n_linked"] < min_offgrid_linked:
            continue
        off_row = paired_deltas(
            cells[name]["_columns"], cells[off_ref]["_columns"],
            metrics=metrics, n_contrasts=offgrid_n_contrasts,
            cfca=cells[name]["cfca"].get("cfca"),
            ref_cfca=cells[off_ref]["cfca"].get("cfca"),
        )
        off_row["cfca"] = cfca_delta_block(
            cells[name]["_columns"], cells[off_ref]["_columns"],
            cfca=cells[name]["cfca"].get("cfca"),
            ref_cfca=cells[off_ref]["cfca"].get("cfca"),
            n_contrasts=offgrid_n_contrasts, warn=printer,
        )
        off_deltas[contrast_key(name, off_ref)] = off_row

    for cell in cells.values():
        cell.pop("_columns", None)
    if not include_offgrid_cells:
        for name in offgrid:
            cells.pop(name, None)

    report = {
        "config": settings.as_pricing_basis(),
        "reference": reference,
        "conditions": [n for n in list(grid) + list(offgrid) if n in cells],
        "grid": list(grid),
        "offgrid": list(offgrid),
        "n_contrasts": family,
        "run_counts": {n: run_counts[n] for n in cells},
        "cells": cells,
        (deltas_field or deltas_key(reference)): deltas,
    }
    if offgrid:
        report["offgrid_reference"] = off_ref
        report["offgrid_n_contrasts"] = offgrid_n_contrasts
        report[offgrid_field or offgrid_deltas_key(off_ref)] = off_deltas
    return report


def evaluate_partial_split(
    extracted: Dict[str, List[QueryRecord]],
    *,
    reference: str = DEFAULT_REFERENCE,
    per_answer_cost: Optional[Dict[str, float]] = None,
    grid_names: Sequence[str] = GRID_ORDER,
    offgrid_names: Sequence[str] = OFFGRID_ORDER,
    unknown: str = "grid",
    deltas_field: Optional[str] = None,
    cell_fn: Callable[..., Dict] = evaluate_cell_runs,
    printer: Optional[Callable[[str], None]] = print,
) -> Dict:
    """
    Cells-only report for a run whose reference cell has no scored answers yet.

    Every scored cell gets its full per-cell block, so a sitting is readable
    the moment it lands. No delta exists without the reference, so the grid
    delta block is written empty, no off-grid delta block is written at all,
    and the report carries a ``partial`` marker naming the missing cell.
    `render_markdown` prints that marker, and `verify_report` reads it to
    report the absent delta blocks at its ``on_partial`` status instead of
    failing. Re-running `run` once the reference cell is scored replaces this
    report with the full `evaluate_experiment_split` one.

    Args: as `evaluate_experiment_split`, minus the delta tunables, which
        would have nothing to act on here.
    """
    per_answer_cost = per_answer_cost or {}
    grid, offgrid = split_grid(
        extracted.keys(),
        grid_names=grid_names,
        offgrid_names=offgrid_names,
        unknown=unknown,
        reference=reference,
    )
    cells = {
        name: cell_fn(extracted[name], per_answer_cost.get(name))
        for name in list(grid) + list(offgrid)
    }
    run_counts = {n: int(cells[n].get("n_runs") or 1) for n in cells}
    for cell in cells.values():
        cell.pop("_columns", None)
    if printer:
        printer(f"  partial report: reference cell {reference!r} has no scored "
                "answers, so no delta block is written yet")
    return {
        "config": settings.as_pricing_basis(),
        "reference": reference,
        "conditions": list(grid) + list(offgrid),
        "grid": list(grid),
        "offgrid": list(offgrid),
        "n_contrasts": max(1, len(grid) - 1),
        "run_counts": run_counts,
        "cells": cells,
        (deltas_field or deltas_key(reference)): {},
        "partial": {
            "reason": f"reference cell {reference!r} has no scored answers",
            "missing_reference": reference,
        },
    }


def evaluate_experiment(
    extracted: Dict[str, List[QueryRecord]],
    reference: str = DEFAULT_REFERENCE,
    per_answer_cost: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Full 2x2 report: cells + paired deltas vs the reference cell.

    Every condition passed in is scored and joins the Bonferroni family, so an
    off-grid window given here becomes a fifth result. `evaluate_experiment_split`
    is the entry point that keeps the before/after check out of the grid.
    """
    per_answer_cost = per_answer_cost or {}
    cells = {
        name: evaluate_cell_runs(recs, per_answer_cost.get(name))
        for name, recs in extracted.items()
    }
    if reference not in cells:
        raise ValueError(
            f"reference cell '{reference}' not among conditions {list(cells)}"
        )

    ref_cols = cells[reference]["_columns"]
    n_contrasts = max(1, len(cells) - 1)   # Bonferroni family size (vs-reference contrasts)
    deltas: Dict[str, Dict] = {}
    for name, cell in cells.items():
        if name == reference:
            continue
        # CFCA rides along as a ratio delta: cost over the all-pass rate
        row = paired_deltas(
            cell["_columns"], ref_cols,
            n_contrasts=n_contrasts,
            cfca=cell["cfca"].get("cfca"),
            ref_cfca=cells[reference]["cfca"].get("cfca"),
        )
        row["cfca"] = cfca_delta_block(
            cell["_columns"], ref_cols,
            cfca=cell["cfca"].get("cfca"),
            ref_cfca=cells[reference]["cfca"].get("cfca"),
            n_contrasts=n_contrasts,
        )
        deltas[contrast_key(name, reference)] = row

    # strip internal columns from the emitted cells
    for cell in cells.values():
        cell.pop("_columns", None)

    return {
        "config": settings.as_pricing_basis(),
        "reference": reference,
        "conditions": list(extracted.keys()),
        "cells": cells,
        deltas_key(reference): deltas,
    }


def _safe_sub(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 8)


# ------------------------------------------------------------------ markdown
OFFGRID_CELLS_NOTE = "\n(off-grid, measured but not scored in the 2x2: {names})"

REPEATS_NOTE = (
    "\n(repeats are averaged per question; the CI is over questions, not sittings)"
)

OFFGRID_TITLE = "Off-grid before/after (validity check, uncorrected, not a 2x2 result)"

OFFGRID_NOTE = (
    "The one-time baseline construction moved the pipeline from {cell} to "
    "{reference}: 256-tok truncation removed, embedder window >=512 tok, chunk "
    "size ~400 tok, hybrid BM25 and dense fused by RRF. This block sizes that "
    "move. It sits outside the Bonferroni family of {n_contrasts} grid "
    "contrasts and carries no research claim."
)


def render_offgrid_section(
    report: Dict,
    *,
    title: str = OFFGRID_TITLE,
    note: str = OFFGRID_NOTE,
    columns: str = "| metric | delta | 95% CI | perm p (uncorrected) | n |",
    cfca_label: str = "CFCA (GBP)",
    cfca_key: str = "cfca",
    empty: Sequence[str] = (),
) -> List[str]:
    """
    Markdown lines for the off-grid delta block, or ``empty`` when a report has
    none (every report written before the split, and any run without a
    before-window).

    Args:
        title: section heading.
        note: sentence under it; ``{cell}``, ``{reference}`` and
            ``{n_contrasts}`` are filled from the report. "" omits it.
        columns: table header; the separator row is derived from it.
        cfca_label / cfca_key: how the aggregate CFCA row is labelled and
            which delta field holds it.
    """
    block = offgrid_deltas_of(report)
    if not block:
        return list(empty)
    reference = report.get("offgrid_reference") or report.get("reference")
    lines = [f"\n## {title}\n"]
    if note:
        lines.append(note.format(
            cell=", ".join(report.get("offgrid") or ["prebaseline"]),
            reference=reference,
            n_contrasts=report.get("n_contrasts"),
        ) + "\n")
    sep = "|" + "---|" * max(1, columns.count("|") - 1)
    for contrast, dmap in block.items():
        lines.append(f"### {contrast}\n")
        lines.append(columns)
        lines.append(sep)
        for m, d in dmap.items():
            if m == cfca_key and d.get("method") is None:
                lines.append(f"| {cfca_label} | {_fmt(d.get('delta'))} | (aggregate) |  |  |")
                continue
            ci = f"[{_fmt(d.get('ci_low'))}, {_fmt(d.get('ci_high'))}]"
            lines.append(
                f"| {cfca_label if m == cfca_key else m} | {_fmt(d.get('delta'))} | {ci} "
                f"| {_fmt(d.get('perm_p'))} | {_fmt(d.get('n'))} |"
            )
        lines.append("")
    return lines


def served_width_note(
    report: Dict,
    *,
    cells_key: str = "cells",
    pool_key: str = "ir_pool",
    values_key: str = "top_n_values",
    prefix: str = ", served top_n ",
    unknown: str = "",
    sep: str = "/",
) -> str:
    """
    The widths the app actually served, for the line that claims the k cap.

    The cap is a property of the logs, so the line that states it carries the
    measured widths beside it; a report written without the pool block says
    nothing rather than guessing.
    """
    widths: List[str] = []
    for cell in (report.get(cells_key) or {}).values():
        pool = cell.get(pool_key) if isinstance(cell, dict) else None
        for value in ((pool or {}).get(values_key) or {}):
            if str(value) not in widths:
                widths.append(str(value))
    if not widths:
        return unknown
    return prefix + sep.join(sorted(widths, key=lambda v: (len(v), v)))


LANGMET_TITLE = "RAGA scores (LangMet `compute_raga_metrics`)"

LANGMET_NOTE = (
    "\n(one RagaEvaluationEvent per linked answer; `(n)` is the events that "
    "carried that metric, so `n/a (0)` means the source is absent, not zero: "
    "answer relevancy has no logged score and no judge runs by default)"
)


def render_langmet_section(
    report: Dict,
    *,
    title: str = LANGMET_TITLE,
    note: str = LANGMET_NOTE,
    cells_key: str = "cells",
    block_key: str = "faithfulness",
    summary_key: str = "langmet_summary",
    fields: Sequence[str] = RAGA_FIELDS,
    scores_key: str = "scores",
    counts_key: str = "evaluation_counts",
    overview_key: str = "overview",
    overall_key: str = "overall_score",
    total_key: str = "total_evaluations",
    overall_label: str = "**overall**",
    error_key: str = "error",
    show_counts: bool = True,
    count_fmt: Callable[[Any], str] = lambda n: _fmt_count(n),
    empty: Sequence[str] = (),
) -> List[str]:
    """
    Markdown for LangMet's RAGA block: one row per metric, one column per cell.

    Args:
        fields: metric rows, in order; the overall row is appended after them.
        scores_key / counts_key: where the per-metric average and its event
            count sit in LangMet's output.
        overview_key / overall_key / total_key: the overall row's two sources.
        overall_label: row label for that overall row.
        show_counts: False prints the score alone, without ``(n)``.
        count_fmt: renders that count; the default drops the decimals a
            repeated cell's averaged count carries.
        error_key: a summary carrying this key prints its message as a bullet
            under the table, so a LangMet failure is visible in the report.
        empty: lines returned when no cell holds a summary.
    """
    summaries: Dict[str, Dict] = {}
    for name, cell in (report.get(cells_key) or {}).items():
        block = cell.get(block_key) if isinstance(cell, dict) else None
        summary = (block or {}).get(summary_key) if isinstance(block, dict) else None
        if isinstance(summary, dict) and summary:
            summaries[name] = summary
    if not summaries:
        return list(empty)

    names = list(summaries)
    lines = [f"## {title}\n"]
    lines.append("| metric | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))

    rows = [(field, scores_key, counts_key, field) for field in fields]
    rows.append((overall_label, overview_key, overview_key, overall_key))
    for label, value_block, count_block, key in rows:
        cells: List[str] = []
        for name in names:
            summary = summaries[name]
            value = (summary.get(value_block) or {}).get(key)
            text = _fmt(value)
            if show_counts:
                count = (summary.get(count_block) or {}).get(
                    total_key if key == overall_key else key
                )
                text += f" ({count_fmt(count)})"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append(note)
    for name in names:
        message = summaries[name].get(error_key)
        if message:
            lines.append(f"- {name}: {message}")
    lines.append("")
    return lines


FAITH_ANSWERS_TITLE = "Per-answer faithfulness (what the strict flag is made of)"

FAITH_ANSWERS_NOTE = (
    "\n- `strict` counts an answer faithful only at 0 confident ungrounded spans, "
    "so one flagged clause zeroes it, and a cell of near-clean answers still "
    "reports 0.0000; `grounded` is 1 - flagged/answer chars over the same answer."
    "\n- `tok/win` is the largest prompt the detector built against its own "
    "window; `trunc` > 0 means that prompt was cut to fit, and since the cut "
    "takes the context and keeps the answer, those spans were scored against "
    "evidence the model never read (`faithfulness.lettuce_window_read`)."
)


def render_faithfulness_answers(
    report: Dict,
    *,
    title: str = FAITH_ANSWERS_TITLE,
    note: str = FAITH_ANSWERS_NOTE,
    cells_key: str = "cells",
    block_key: str = "faithfulness",
    per_query_key: str = "per_query",
    span_key: str = "spans",
    span_chars: int = 70,
    only_flagged: bool = False,
    empty: Sequence[str] = (),
) -> List[str]:
    """One row per scored answer: the strict flag, its spans, and the window read.

    Args:
        per_query_key: where the per-answer dicts sit in each cell's
            faithfulness block; `evaluate_experiment_split` keeps them on disk
            for this table alone.
        span_key / span_chars: the flagged span list, and how much of the first
            span's text to quote; 0 drops that column's content.
        only_flagged: True prints only answers carrying at least one span.
        empty: lines returned when no cell holds per-answer detail.
    """
    rows: List[str] = []
    for name, cell in (report.get(cells_key) or {}).items():
        block = cell.get(block_key) if isinstance(cell, dict) else None
        per_query = (block or {}).get(per_query_key) if isinstance(block, dict) else None
        if not isinstance(per_query, dict):
            continue
        for qid, v in per_query.items():
            if only_flagged and not v.get("ungrounded_spans"):
                continue
            spans = v.get(span_key) or []
            quote = ""
            if span_chars and spans:
                text = " ".join(str(spans[0].get("text") or "").split())
                clipped = text[:span_chars] + ("..." if len(text) > span_chars else "")
                quote = clipped.replace("|", "\\|")
            rows.append(
                f"| {name} | {qid} | {_fmt(v.get('faithful'))} "
                f"| {_fmt(v.get('faithfulness'))} | {_fmt(v.get('ungrounded_spans'))} "
                f"| {_fmt(v.get('halluc_char_frac'))} | {_fmt(v.get('answer_chars'))} "
                f"| {_fmt(v.get('context_chars'))} "
                f"| {_fmt(v.get('prompt_tokens'))}/{_fmt(v.get('window'))} "
                f"| {_fmt(v.get('truncated'))} | {quote} |"
            )
    if not rows:
        return list(empty)
    lines = [f"## {title}\n"]
    lines.append("| cell | query | strict | grounded | spans | flagged frac | "
                 "answer chars | ctx chars | tok/win | trunc | first flagged span |")
    lines.append("|" + "---|" * 11)
    lines.extend(rows)
    lines.append(note)
    lines.append("")
    return lines


def render_markdown(report: Dict) -> str:
    k = settings.ir_k
    lines: List[str] = []
    lines.append("# VERIDIC eval report\n")
    reference = report.get("reference")
    partial = report.get("partial") if isinstance(report.get("partial"), dict) else None
    lines.append(f"- reference cell: **{reference}**")
    if partial:
        lines.append(f"- **PARTIAL REPORT**: {partial.get('reason')}; the delta "
                     "tables return when that cell is scored and `run` is re-run")
    lines.append(f"- conditions: {', '.join(report['conditions'])}")
    lines.append(f"- IR cap: k = {k} (logs-only; served evidence only)"
                 f"{served_width_note(report)}\n")

    lines.append("## Per-cell summary\n")
    # Two faithfulness rates, because they disagree: `strict` is the all-or-
    # nothing per-answer flag P_hat and CFCA divide by, `grounded` is the mean
    # unflagged character fraction of the same answers.
    header = (f"| cell | runs | linked | recall@{k} | ndcg@{k} | faithful (strict) | "
              f"grounded chars | abst.recall | over-abst | CFCA (GBP) |")
    lines.append(header)
    lines.append("|" + "---|" * 10)
    for name, cell in report["cells"].items():
        r = cell["retrieval"]["aggregate"]
        fa = cell["faithfulness"].get("aggregate", {})
        ab = cell["abstention"]["aggregate"]
        cf = cell["cfca"]
        lines.append(
            f"| {name} | {_fmt_runs(cell)} | {cell['n_linked']}/{cell['n_queries']} "
            f"| {_fmt(r.get(f'recall@{k}'))} | {_fmt(r.get(f'ndcg@{k}'))} "
            f"| {_fmt(fa.get('faithful_rate'))} | {_fmt(fa.get('faithfulness'))} "
            f"| {_fmt(ab.get('abstention_recall'))} "
            f"| {_fmt(ab.get('over_abstention_rate'))} | {_fmt(cf.get('cfca'))} |"
        )
    if any(int(c.get("n_runs") or 1) > 1 for c in report["cells"].values()):
        lines.append(REPEATS_NOTE)

    offgrid = report.get("offgrid") or []
    if offgrid:
        lines.append(OFFGRID_CELLS_NOTE.format(names=", ".join(offgrid)))

    lines.append("")
    lines.extend(render_faithfulness_answers(report))
    lines.extend(render_langmet_section(report))

    deltas = deltas_of(report)
    if deltas or not partial:
        lines.append(
            f"\n## Paired deltas vs {reference} (BCa 95% CI; * = both CI ends one side of 0)\n"
        )
    for contrast, dmap in deltas.items():
        lines.append(f"### {contrast}\n")
        lines.append("| metric | delta | 95% CI | perm p (Bonf) | sig |")
        lines.append("|---|---|---|---|---|")
        for m, d in dmap.items():
            if m == "cfca" and d.get("method") is None:
                lines.append(f"| CFCA (GBP) | {_fmt(d.get('delta'))} | (aggregate) |  |  |")
                continue
            ci = f"[{_fmt(d.get('ci_low'))}, {_fmt(d.get('ci_high'))}]"
            sig = "*" if d.get("significant") else ""
            pfmt = _fmt(d.get("perm_p_bonferroni"))
            label = "CFCA (GBP)" if m == "cfca" else m
            lines.append(f"| {label} | {_fmt(d.get('delta'))} | {ci} | {pfmt} | {sig} |")
        lines.append("")

    lines.extend(render_offgrid_section(report))

    lines.append("## Limitations (logs-only; see methodnow Part B)\n")
    lines.append("- IR metrics capped at k <= top_n; pre-rerank pool not logged.")
    lines.append("- Served order reconstructed as rerank desc, retrieval desc.")
    lines.append("- Faithfulness context rebuilt from message_evidence -> chunks.text.")
    lines.append("- Shallow-pool qrels bias; settle qrels + judge blind to condition.")
    lines.append("- 'cited' = >=1 evidence link; 'right-version' from served doc order.")
    return "\n".join(lines)


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _fmt_count(v):
    """``10`` for an event count, ``9.5`` once repeats average it, ``0`` for None."""
    if v is None:
        return "0"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _fmt_runs(cell: Dict) -> str:
    """``1`` for a cell asked once, ``3 x 10 q`` once it carries repeats."""
    n_runs = int(cell.get("n_runs") or 1)
    if n_runs <= 1:
        return "1"
    return f"{n_runs} x {cell.get('n_questions')} q"
