"""
Retrieval (IR) evaluation with ranx.

Logs-only cap: only the served top_n evidence is persisted, so every metric is
capped at k <= top_n (default 5). We report recall@k, mrr@k, ndcg@k, hit_rate@k
over what the system actually served. Retrieval-stage vs rerank-stage
decomposition is out of scope (the pre-rerank pool is not logged) - see
methodnow-rescoped.md Part B.

qrels come from each record's judged_chunk_ids, which gold_evidence.py resolved
from the benchmark's gold_evidence against that cell's own ingest. Where the
researcher pools
judgments from served evidence across conditions, that pooling and its
shallow-pool bias must be handled at labeling time (settle the qrels before any
cell is compared); this module consumes whatever gold is supplied.

The run is built from served order using deterministic rank-based synthetic
scores (1/(rank+1)); this reproduces the served ranking exactly and sidesteps
score ties / [0,1] clamping in message_evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .config import settings
from .extract import QueryRecord


def _metric_names(k: int) -> List[str]:
    return [f"recall@{k}", f"mrr@{k}", f"ndcg@{k}", f"hit_rate@{k}"]


def build_qrels_run(records: List[QueryRecord]) -> tuple[dict, dict, List[str]]:
    """
    Return (qrels_dict, run_dict, judged_query_ids).

    Only queries with non-empty judged_chunk_ids are judged (you cannot score IR
    without relevance judgments). Linked queries that served nothing still
    appear in the run (as a single non-relevant placeholder) so they score 0
    rather than being silently dropped.
    """
    qrels: Dict[str, Dict[str, int]] = {}
    run: Dict[str, Dict[str, float]] = {}
    judged: List[str] = []

    for rec in records:
        if not rec.judged_chunk_ids:
            continue
        qid = rec.query_id
        judged.append(qid)
        qrels[qid] = {cid: 1 for cid in rec.judged_chunk_ids}

        served = rec.served_evidence
        if served:
            run[qid] = {e.chunk_id: 1.0 / (e.served_rank + 1) for e in served}
        else:
            # placeholder guarantees a scored (=0) query
            run[qid] = {"__none__": 0.0}

    return qrels, run, judged


def evaluate_retrieval(records: List[QueryRecord], k: int | None = None) -> Dict:
    """
    Compute aggregate + per-query IR metrics for one condition.

    Returns::
        {
          "k": 5,
          "n_judged": 12,
          "aggregate": {"recall@5": .., "mrr@5": .., "ndcg@5": .., "hit_rate@5": ..},
          "per_query": {"recall@5": {qid: val, ...}, ...},
        }
    """
    k = k or settings.ir_k
    metrics = _metric_names(k)
    qrels_d, run_d, judged = build_qrels_run(records)

    if not judged:
        return {"k": k, "n_judged": 0, "aggregate": {m: None for m in metrics},
                "per_query": {m: {} for m in metrics},
                "note": "no judged queries (no gold_evidence resolved for this cell)"}

    from ranx import Qrels, Run, evaluate as ranx_evaluate  # lazy import

    qrels = Qrels(qrels_d)
    run = Run(run_d)

    aggregate = ranx_evaluate(qrels, run, metrics, return_mean=True)
    per_query_raw = ranx_evaluate(qrels, run, metrics, return_mean=False)

    # ranx returns {metric: np.ndarray} aligned to qrels query order.
    ordered_qids = list(qrels_d.keys())
    per_query: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        arr = per_query_raw[m] if isinstance(per_query_raw, dict) else per_query_raw
        per_query[m] = {qid: float(arr[i]) for i, qid in enumerate(ordered_qids)}

    # normalise aggregate to plain floats
    if isinstance(aggregate, dict):
        aggregate = {m: float(v) for m, v in aggregate.items()}
    else:  # single metric edge case
        aggregate = {metrics[0]: float(aggregate)}

    return {
        "k": k,
        "n_judged": len(judged),
        "aggregate": aggregate,
        "per_query": per_query,
    }


# --------------------------------------------------------------- served pool
def served_pool_summary(
    records: Sequence[QueryRecord],
    *,
    k: Optional[int] = None,
    top_n_field: str = "top_n",
    top_k_field: str = "top_k",
    linked_only: bool = True,
    judged_only: bool = False,
    count_served: bool = True,
) -> Dict[str, Any]:
    """
    The retrieval pool each answer actually had, so ``k <= top_n`` is checkable.

    Every metric in this module is capped by the served pool: the pre-rerank
    candidates are never logged, so a cell served with ``top_n`` below ``k``
    scores recall@k against a pool that never held k rows, and its delta against
    a cell served at k is an artefact of the app's configuration. The claim sits
    in four docstrings and nothing enforced it, so this reads the logged widths
    and `verify.check_ir_pool_block` turns them into assertions.

    Args:
        k: the k the cell is scored at; None takes ``settings.ir_k``.
        top_n_field / top_k_field: record attributes holding the app's rerank
            and retrieval widths, as `extract` copies them off ``rag_logs``.
        linked_only: count only records with a matched answer, the only ones
            that can carry a rag log. False counts every record.
        judged_only: restrict further to records with gold, the exact set the IR
            metrics score.
        count_served: also summarise ``len(served_evidence)``, the pool floor
            that survives when the rag log is missing.

    Returns:
        ``{k, n_records, n_with_top_n, n_missing_top_n, top_n_min, top_n_max,
        top_n_values, top_k_min, top_k_max, served_min, served_max}``. Every
        min/max is None when nothing supplied it.
    """
    k = k or settings.ir_k
    picked = [r for r in records if (r.linked or not linked_only)]
    if judged_only:
        picked = [r for r in picked if r.judged_chunk_ids]

    top_ns: List[int] = []
    top_ks: List[int] = []
    served: List[int] = []
    values: Dict[str, int] = {}
    for rec in picked:
        n = getattr(rec, top_n_field, None)
        if isinstance(n, int) and not isinstance(n, bool):
            top_ns.append(n)
            key = str(n)
            values[key] = values.get(key, 0) + 1
        kk = getattr(rec, top_k_field, None)
        if isinstance(kk, int) and not isinstance(kk, bool):
            top_ks.append(kk)
        if count_served:
            served.append(len(rec.served_evidence))

    return {
        "k": k,
        "n_records": len(picked),
        "n_with_top_n": len(top_ns),
        "n_missing_top_n": len(picked) - len(top_ns),
        "top_n_min": min(top_ns) if top_ns else None,
        "top_n_max": max(top_ns) if top_ns else None,
        "top_n_values": values,
        "top_k_min": min(top_ks) if top_ks else None,
        "top_k_max": max(top_ks) if top_ks else None,
        "served_min": min(served) if served else None,
        "served_max": max(served) if served else None,
    }
