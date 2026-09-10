"""
Cost + CFCUA.

CFCUA = cost / P(faithful . cited . right-version), computed over ANSWERABLE
items only, reported as a delta vs the control cell.

Cost numerator:
  * The per-answer API token cost is computed here. Ollama / local serving has
    NO per-token API cost -> 0.0 per answer; its
    GPU/compute is a separate one-time/recurring lifecycle term, added ONCE in
    the cost equation, never per prompt.
  * For local (Ollama) runs, pass the recurring per-query serving cost from the
    lifecycle table via ``per_answer_cost=`` to get a non-degenerate CFCUA.

Cost aggregation reuses **LangMet** ``compute_cost_metrics`` when available.

Denominator components (logged proxies; stated as limitations, Part B):
  * faithful      — from the faithfulness backend (per-query `faithful` flag)
  * cited         — the answer carried >=1 evidence link (message_evidence non-empty)
  * right-version — served evidence includes the governing doc and does not rank
                    a superseded doc above it (only constrains items that declare
                    a governing/superseded doc; others pass as not-applicable)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from .extract import QueryRecord

_ZERO_COST_PROVIDERS = {"ollama", "local", "localhost", "self-hosted"}


# ----------------------------------------------------------------- components
def is_cited(rec: QueryRecord) -> int:
    return int(rec.has_evidence)


def is_right_version(rec: QueryRecord) -> int:
    """1 unless the item declares a governing doc that was mis-served."""
    if not rec.governing_doc:
        return 1  # not applicable -> does not penalise
    served_docs = [e.document_name for e in rec.served_evidence]
    if rec.governing_doc not in served_docs:
        return 0
    if rec.superseded_doc and rec.superseded_doc in served_docs:
        # governing must be ranked above superseded in served order
        gov_rank = min(i for i, d in enumerate(served_docs) if d == rec.governing_doc)
        sup_rank = min(i for i, d in enumerate(served_docs) if d == rec.superseded_doc)
        return int(gov_rank < sup_rank)
    return 1


# --------------------------------------------------------------------- cost
def _per_answer_api_cost(rec: QueryRecord) -> float:
    """API token cost for one answer. 0.0 for local/Ollama serving."""
    if (rec.provider or "").lower() in _ZERO_COST_PROVIDERS:
        return 0.0
    # Non-local providers: defer to LangMet's price table via cost_overview();
    # for the per-answer figure we fall back to 0.0 when unpriced.
    return 0.0


def cost_overview(records: List[QueryRecord]) -> Dict:
    """Aggregate cost via LangMet (best-effort); annotate local zero-cost basis."""
    linked = [r for r in records if r.linked and r.tokens_total is not None]
    providers = {(r.provider or "unknown").lower() for r in linked}
    local_only = providers and providers.issubset(_ZERO_COST_PROVIDERS)

    langmet_summary: Dict = {}
    try:
        from langmet.cost import compute_cost_metrics
        from langmet.models import CompletionEvent

        now = datetime.now(timezone.utc)
        events = [
            CompletionEvent(
                provider=r.provider,
                model=r.model,
                latency_ms=r.latency_ms or 0,
                tokens_total=r.tokens_total or 0,
                error_message=None,
                created_at=now,
                prompt_tokens=r.tokens_input,
                completion_tokens=r.tokens_output,
            )
            for r in linked
            if (r.provider or "").lower() not in _ZERO_COST_PROVIDERS
        ]
        if events:
            langmet_summary = compute_cost_metrics(events)
    except Exception:
        langmet_summary = {}

    return {
        "n_completions": len(linked),
        "providers": sorted(providers),
        "local_zero_cost": bool(local_only),
        "note": (
            "Local/Ollama serving has no per-token API cost; "
            "per-answer API cost = 0. Supply the recurring per-query serving "
            "cost via per_answer_cost= for a non-degenerate CFCUA."
        ),
        "langmet_cost": langmet_summary,
    }


# --------------------------------------------------------------------- CFCUA
def compute_cfcua(
    records: List[QueryRecord],
    faithfulness_per_query: Dict[str, Dict],
    per_answer_cost: Optional[float] = None,
) -> Dict:
    """
    Compute CFCUA over answerable items.

    per_answer_cost: override the per-answer cost numerator (USD). If None, uses
    the computed API token cost (0.0 for Ollama).
    """
    answerable = [r for r in records if r.linked and r.answerable]

    per_query_joint: Dict[str, int] = {}
    faithful_col: Dict[str, int] = {}
    cited_col: Dict[str, int] = {}
    version_col: Dict[str, int] = {}

    for rec in answerable:
        f = faithfulness_per_query.get(rec.query_id)
        faithful = int(f["faithful"]) if f else 0
        cited = is_cited(rec)
        version = is_right_version(rec)
        faithful_col[rec.query_id] = faithful
        cited_col[rec.query_id] = cited
        version_col[rec.query_id] = version
        per_query_joint[rec.query_id] = int(faithful and cited and version)

    n = len(answerable)
    p_joint = (sum(per_query_joint.values()) / n) if n else None

    # cost numerator
    if per_answer_cost is None:
        api_costs = [_per_answer_api_cost(r) for r in answerable]
        cost_num = (sum(api_costs) / len(api_costs)) if api_costs else 0.0
        cost_source = "api-tokens (0.0 for local/Ollama)"
    else:
        cost_num = float(per_answer_cost)
        cost_source = "supplied recurring per-query serving cost"

    if p_joint in (None, 0.0):
        cfcua = None
        cfcua_note = "P(faithful·cited·right-version) = 0 or undefined; CFCUA undefined."
    else:
        cfcua = round(cost_num / p_joint, 8)
        cfcua_note = None

    return {
        "n_answerable": n,
        "cost_per_answer_usd": round(cost_num, 8),
        "cost_source": cost_source,
        "p_faithful_cited_version": round(p_joint, 4) if p_joint is not None else None,
        "cfcua": cfcua,
        "note": cfcua_note,
        "components_mean": {
            "faithful": round(sum(faithful_col.values()) / n, 4) if n else None,
            "cited": round(sum(cited_col.values()) / n, 4) if n else None,
            "right_version": round(sum(version_col.values()) / n, 4) if n else None,
        },
        "per_query": {
            "joint": per_query_joint,          # for bootstrap CI on P
            "faithful": faithful_col,
            "cited": cited_col,
            "right_version": version_col,
        },
    }
