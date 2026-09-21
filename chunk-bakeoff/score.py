"""Retrieval metrics at k: the four the methodology names (methodnow-rescoped.md:92)."""

import math
from typing import Dict, List, Set, Tuple


def chunk_is_relevant(chunk, gold_pages: Set[Tuple[str, int]]) -> bool:
    """Gold is page-level: (document, page). A chunk hits if its page span covers one."""
    return any(
        doc == chunk["document"] and chunk["page_start"] <= page <= chunk["page_end"]
        for doc, page in gold_pages
    )


def score_ranking(
    ranked: List[dict],
    gold_pages: Set[Tuple[str, int]],
    k: int = 5,
    n_relevant_total: int = 0,
) -> Dict[str, float]:
    """
    Args:
        ranked: retrieved chunks, best first, each {document, page_start, page_end}.
        gold_pages: the question's gold (document, page) pairs.
        k: cutoff; the app generator reads top_n = 5.
        n_relevant_total: relevant chunks in the whole corpus for this question
            (for the nDCG ideal); computed by the caller once per strategy.
    """
    top = ranked[:k]
    rel = [1 if chunk_is_relevant(c, gold_pages) else 0 for c in top]

    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_n = min(k, n_relevant_total)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_n))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    hit_pages = {
        (doc, page)
        for doc, page in gold_pages
        for c in top
        if doc == c["document"] and c["page_start"] <= page <= c["page_end"]
    }
    recall_pages = len(hit_pages) / len(gold_pages) if gold_pages else 0.0

    mrr = 0.0
    for i, r in enumerate(rel):
        if r:
            mrr = 1.0 / (i + 1)
            break

    return {
        "ndcg": ndcg,
        "recall_pages": recall_pages,
        "mrr": mrr,
        "hit": 1.0 if any(rel) else 0.0,
    }
