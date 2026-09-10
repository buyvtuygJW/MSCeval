"""
Cost + CFCA.

CFCA = cost / P(faithful . cited), computed over ANSWERABLE items only and
reported as a delta vs the control cell.

Cost numerator:
  * The per-answer API token cost is computed here. Ollama / local serving has
    NO per-token API cost -> 0.0 per answer; its
    GPU/compute is a separate one-time/recurring lifecycle term, added ONCE in
    the cost equation, never per prompt.
  * For local (Ollama) runs, pass the recurring per-query serving cost from the
    lifecycle table via ``per_answer_cost=`` to get a non-degenerate CFCA.

Cost aggregation reuses **LangMet** ``compute_cost_metrics`` when available.

Denominator gates (logged proxies; stated as limitations, Part B):
  * faithful      - from the faithfulness backend (per-query `faithful` flag)
  * cited         - the answer carried >=1 evidence link (message_evidence non-empty)
  * right-version - a third gate that is inert on this corpus and left in the
                    code as a guard: it passes any item declaring no governing
                    document, and a single-version corpus supersedes nothing, so
                    the product equals the two-gate CFCA denominator. Version
                    correctness is future work, not part of the metric.
                    It can only bite if a superseded document is added later.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .cfca_cost import GBP_USD, gbp_from_usd
from .extract import QueryRecord

_ZERO_COST_PROVIDERS = {"ollama", "local", "localhost", "self-hosted"}

#: The two CFCA per-query token quantities, named as `cfca_cost.cost_per_answer`
#: names them, so a measured pair drops into a cost block through
#: `ingest_cost.apply_measured_ingest` with no second applier.
SERVING_TOKEN_KEYS: Tuple[str, ...] = ("query_in_mtok", "query_out_mtok")


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


def serving_token_totals(
    records: Sequence[QueryRecord],
    *,
    linked_only: bool = True,
    answerable_only: bool = True,
    providers: Optional[Sequence[str]] = None,
    in_key: str = SERVING_TOKEN_KEYS[0],
    out_key: str = SERVING_TOKEN_KEYS[1],
    min_coverage: float = 1.0,
    on_incomplete: str = "report",
    round_mtok: int = 6,
    round_coverage: int = 4,
) -> Dict[str, Any]:
    """
    The serving tokens this cell actually spent, per answer, in millions.

    The counts come from the app's ``completion_logs`` rows, which the serving
    backend wrote from its own tokeniser, so this is a measurement and not a
    words-to-tokens conversion. The two figures are per answer because that is
    what the CFCA query term prices, and the sums are reported beside them.

    Args:
        linked_only: count answers the cell actually linked, which is the only
            place a token row exists.
        answerable_only: match the CFCA scope, which is answerable items. False
            adds the abstention items, whose prompts also cost tokens.
        providers: restrict to these provider names, casefolded; None takes all.
        in_key / out_key: names the two figures are returned under, matching
            `cfca_cost.cost_per_answer`.
        min_coverage: share of counted answers required before the figures are
            offered as measured.
        on_incomplete: ``report`` returns None figures with the reason,
            ``raise`` stops.
        round_mtok / round_coverage: decimals on the reported figures.

    Returns:
        ``{in_key, out_key, measured, source, ...}``, the shape
        `apply_measured_tokens` writes into a cost block. Both figures are None
        when the count is short of ``min_coverage``, so nothing is applied.
    """
    if on_incomplete not in ("report", "raise"):
        raise ValueError("on_incomplete must be 'report' or 'raise'")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage!r}")

    wanted = {str(p).lower() for p in providers} if providers is not None else None
    used = [
        r for r in records
        if (r.linked or not linked_only)
        and (r.answerable or not answerable_only)
        and (wanted is None or (r.provider or "").lower() in wanted)
    ]
    split = [r for r in used if r.tokens_input is not None and r.tokens_output is not None]
    counted = {id(r) for r in split}
    total_only = [r for r in used
                  if id(r) not in counted and r.tokens_total is not None]
    tokens_in = sum(int(r.tokens_input or 0) for r in split)
    tokens_out = sum(int(r.tokens_output or 0) for r in split)
    coverage = (len(split) / len(used)) if used else 0.0
    measured = bool(split) and coverage >= min_coverage
    missing = [r.query_id for r in used if id(r) not in counted]
    reason = (
        f"{len(split)}/{len(used)} answers carry an input/output token split"
        + (f", {len(total_only)} carry a total only" if total_only else "")
    )
    if not measured and on_incomplete == "raise":
        raise ValueError(
            f"serving token counts incomplete: {reason}; re-dump the cell, or "
            f"lower min_coverage"
        )
    per_in = (tokens_in / len(split)) if split else 0.0
    per_out = (tokens_out / len(split)) if split else 0.0
    return {
        in_key: round(per_in / 1_000_000.0, round_mtok) if measured else None,
        out_key: round(per_out / 1_000_000.0, round_mtok) if measured else None,
        "measured": measured,
        "upper_bound": False,
        "answers": len(used),
        "counted": len(split),
        "total_only": len(total_only),
        "missing": missing,
        "coverage": round(coverage, round_coverage),
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "per_answer_input": round(per_in, 1),
        "per_answer_output": round(per_out, 1),
        "providers": sorted({(r.provider or "unknown").lower() for r in used}),
        "models": sorted({r.model for r in used if r.model}),
        "source": (f"measured (completion_logs, {reason})" if measured
                   else f"unmeasured: {reason}"),
    }


def apply_measured_tokens(
    cost_inputs: Dict[str, Dict[str, float]],
    measured: Dict[str, Optional[Dict[str, Any]]],
    *,
    keys: Sequence[str] = SERVING_TOKEN_KEYS,
    prefer: str = "measured",
    require_measured: Sequence[str] = (),
    printer: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Put each cell's counted serving tokens into its cost block.

    The sibling of `ingest_cost.apply_measured_ingest`, same vocabulary, for the
    per-answer query terms instead of the one-time ones. A cell whose dump
    carries no count keeps the number the yaml declared, and says so.

    Args:
        measured: ``{cell: serving_token_totals(...) or None}``.
        keys: block keys to write; both query terms by default.
        prefer: ``measured`` overwrites, ``declared`` keeps the yaml number,
            ``missing`` fills only keys the yaml never gave.
        require_measured: cells that must carry a count; the others raise.
        printer: one line per changed cell; None stays silent.

    Returns:
        ``(cost_inputs, provenance)``, the second naming each cell's source.
    """
    if prefer not in ("measured", "declared", "missing"):
        raise ValueError("prefer must be 'measured', 'declared' or 'missing'")
    out: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in cost_inputs.items()}
    provenance: Dict[str, str] = {}
    missing: List[str] = []
    for name, block in out.items():
        got = measured.get(name) or {}
        usable = {k: got.get(k) for k in keys if got.get(k) is not None}
        if not usable:
            provenance[name] = f"declared ({got.get('source') or 'no token count'})"
            missing.append(name)
            continue
        if prefer == "declared":
            provenance[name] = f"declared (measured: {got.get('source')} not applied)"
            continue
        wrote: List[str] = []
        for key, value in sorted(usable.items()):
            if prefer == "missing" and key in block:
                continue
            block[key] = float(value)
            wrote.append(f"{key}={value}")
        if not wrote:
            provenance[name] = "declared (every key already declared)"
            continue
        provenance[name] = str(got.get("source") or "measured")
        if printer:
            printer(f"  {name}: {', '.join(wrote)} from {provenance[name]}")
    hard = [n for n in require_measured if n in missing]
    if hard:
        raise ValueError(
            f"cells {sorted(hard)} have no counted serving tokens: re-dump them "
            f"so completion_logs is read, or drop them from require_measured"
        )
    return out, provenance


def cost_overview(records: List[QueryRecord], *,
                  serving_kwargs: Optional[Dict[str, Any]] = None) -> Dict:
    """Aggregate cost via LangMet (best-effort); annotate local zero-cost basis.

    ``serving_kwargs`` passes through to `serving_token_totals`, whose measured
    per-answer Mtok pair rides along under ``serving_tokens`` so a priced run
    reads the app's own counts instead of a typed estimate.
    """
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
            "cost via per_answer_cost= for a non-degenerate CFCA."
        ),
        "langmet_cost": langmet_summary,
        "serving_tokens": serving_token_totals(records, **(serving_kwargs or {})),
    }


# --------------------------------------------------------------------- CFCA
def compute_cfca(
    records: List[QueryRecord],
    faithfulness_per_query: Dict[str, Dict],
    per_answer_cost: Optional[float] = None,
    per_answer_cost_by_query: Optional[Dict[str, float]] = None,
    gbp_usd: float = GBP_USD,
) -> Dict:
    """
    Compute CFCA over answerable items. Every cost here is GBP.

    per_answer_cost: override the per-answer cost numerator (GBP). If None, uses
    the computed API token cost (0.0 for Ollama), converted at `gbp_usd`.
    per_answer_cost_by_query: per-question serving cost (GBP), for a run where
    one question costs more to answer than another. It wins question by
    question; a question it omits falls back to `per_answer_cost`, then to that
    question's API token cost. The numerator stays the mean over answered items,
    so a flat cost gives the same figure it gives today, while the spread is
    kept as a column for the paired ratio bootstrap.
    gbp_usd: dollars to the pound, applied only to the API-token fallback, which
    is the one path priced from a USD sheet.
    """
    answerable = [r for r in records if r.linked and r.answerable]
    by_query = per_answer_cost_by_query or {}

    per_query_joint: Dict[str, int] = {}
    faithful_col: Dict[str, int] = {}
    cited_col: Dict[str, int] = {}
    version_col: Dict[str, int] = {}
    cost_col: Dict[str, float] = {}

    for rec in answerable:
        f = faithfulness_per_query.get(rec.query_id)
        faithful = int(f["faithful"]) if f else 0
        cited = is_cited(rec)
        version = is_right_version(rec)
        faithful_col[rec.query_id] = faithful
        cited_col[rec.query_id] = cited
        version_col[rec.query_id] = version
        per_query_joint[rec.query_id] = int(faithful and cited and version)
        supplied = by_query.get(rec.query_id)
        if supplied is not None:
            cost_col[rec.query_id] = float(supplied)
        elif per_answer_cost is not None:
            cost_col[rec.query_id] = float(per_answer_cost)
        else:
            cost_col[rec.query_id] = gbp_from_usd(_per_answer_api_cost(rec),
                                                  gbp_usd=gbp_usd)

    n = len(answerable)
    p_joint = (sum(per_query_joint.values()) / n) if n else None

    # cost numerator: the mean of the per-answer column, one entry per answer
    served = [cost_col[r.query_id] for r in answerable]
    cost_num = (sum(served) / len(served)) if served else 0.0
    if by_query:
        cost_source = "supplied per-question serving cost (GBP)"
    elif per_answer_cost is not None:
        cost_source = "measured cost per answer (GBP): electricity + any API bill"
    else:
        cost_source = f"api-tokens converted at {gbp_usd} USD/GBP (0.0 for local/Ollama)"
    if len(set(served)) > 1:
        cost_source += "; varies per question"

    if p_joint in (None, 0.0):
        cfca = None
        cfca_note = "P(faithful·cited·right-version) = 0 or undefined; CFCA undefined."
    else:
        cfca = round(cost_num / p_joint, 8)
        cfca_note = None

    return {
        "n_answerable": n,
        "cost_per_answer_gbp": round(cost_num, 8),
        "currency": "GBP",
        "cost_source": cost_source,
        "p_faithful_cited_version": round(p_joint, 4) if p_joint is not None else None,
        "cfca": cfca,
        "note": cfca_note,
        "components_mean": {
            "faithful": round(sum(faithful_col.values()) / n, 4) if n else None,
            "cited": round(sum(cited_col.values()) / n, 4) if n else None,
            "right_version": round(sum(version_col.values()) / n, 4) if n else None,
        },
        "per_query": {
            "joint": per_query_joint,          # for bootstrap CI on P
            "cost": cost_col,                  # numerator column for the CFCA delta
            "faithful": faithful_col,
            "cited": cited_col,
            "right_version": version_col,
        },
    }
