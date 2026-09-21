"""
Chunking strategy bake-off over the app's own builders and embedder.

Three strategies (methodnow-rescoped.md:88-92), scored with nDCG@5 (winner),
recall breaking ties, MRR@5 and hit@5 reported:
  fixed_token      chunking/token_budget.py     (verbatim app copy)
  sentence_window  chunking/chunk_builder.py    (verbatim app copy, the control)
  structure        chunking/structure_aware.py  (new, drop-in style)

Embedders (--embedders): `app` = the app's own EmbeddingsService copy
(Xenova/all-MiniLM-L6-v2 int8, truncate 256) and `gemma` =
EmbeddingGemma-300m ONNX (2048-token window, retrieval prompts). The chunking
strategies are the inner test: chunks are cut per embedder with that
embedder's own token counter, and the budget is capped to the embedder's
usable window (window - document prompt prefix - special tokens), so no chunk
is ever truncated at embed time; this matches the app, which counts
CHUNK_MAX_TOKENS with the embedder's own tokenizer. In-memory cosine, no
Chroma, no re-ingest.

`--sweep t1000_o40,t2048_o40` adds fixed_token budget variants
(fixed_token_t{budget}_o{overlap}) to answer how far the budget should rise
toward the gemma 2048 window.

  python run.py --pages pages.json --bench ../benchmarkv2.yaml
"""

import argparse
import asyncio
import json
import re
from collections import defaultdict

import numpy as np
import yaml

from app_embeddings import EmbeddingsService
from chunking.chunk_builder import build_chunks
from chunking.parser import parse_pages_to_elements
from chunking.structure_aware import build_chunks_structure_aware
from chunking.token_budget import build_chunks_token_budgeted
from gemma_embeddings import GemmaEmbeddingsService
from score import chunk_is_relevant, score_ranking


async def embed(svc, texts, kind):
    """Gemma needs its retrieval prompt per input kind; the app service does not."""
    if hasattr(svc, "QUERY_PREFIX"):
        return await svc.generate_embeddings(texts, kind=kind)
    return await svc.generate_embeddings(texts)


def load_pages(path):
    """pages.json: [{document, page, text}] -> {document: [{page_number, text}]}"""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    docs = defaultdict(list)
    for r in rows:
        docs[r["document"]].append({"page_number": r["page"], "text": r["text"] or ""})
    for pages in docs.values():
        pages.sort(key=lambda p: p["page_number"])
    return docs


def load_queries(path):
    """benchmarkv2.yaml `queries:`; only items with gold_evidence are scorable."""
    with open(path, encoding="utf-8") as f:
        bench = yaml.safe_load(f)
    out = []
    for q in bench["queries"]:
        gold = {
            (ev["document"], int(ev["page"]))
            for ev in (q.get("gold_evidence") or [])
            if "document" in ev and "page" in ev
        }
        if gold:
            out.append({"id": q["id"], "text": q["text"], "gold": gold})
    return out


# Budget-sweep variant of fixed_token: name carries the spec, so every knob is
# explicit in the results table (e.g. fixed_token_t2048_o40 = 2048-token budget,
# 40-token overlap, same builder as fixed_token).
_FIXED_VARIANT = re.compile(r"^fixed_token_t(\d+)_o(\d+)$")


def usable_budget(requested, window, prefix_tokens, special_reserve=2):
    """Largest chunk budget that embeds with zero truncation: the service
    tokenizes prefix + chunk and cuts at `window`, so the chunk itself may
    claim only window - prefix_tokens - special_reserve tokens.

    Args:
        requested: the budget the test asked for.
        window: the embedder's _max_length.
        prefix_tokens: tokens the service prepends per document
            (0 when the service has no prompt prefix).
        special_reserve: BOS/EOS or CLS/SEP margin the tokenizer adds.
    """
    return min(requested, window - prefix_tokens - special_reserve)


def build_all(docs, strategy, tc, a, budget_cap=None):
    """Chunk every doc for one strategy. `tc` is the token counter the budget
    is measured with; pass the target embedder's counter so boundaries are
    native to it. `budget_cap` (int -> int) caps any requested budget to the
    embedder's usable window; None applies budgets uncapped."""
    cap = budget_cap or (lambda t: t)
    chunks = []
    variant = _FIXED_VARIANT.match(strategy)
    for doc_name, pages in docs.items():
        elements = parse_pages_to_elements(pages)
        if strategy == "fixed_token":
            built = build_chunks_token_budgeted(
                doc_name, iter(elements), token_counter=tc,
                max_tokens=cap(a.max_tokens), overlap_tokens=a.overlap_tokens,
            )
        elif variant:
            built = build_chunks_token_budgeted(
                doc_name, iter(elements), token_counter=tc,
                max_tokens=cap(int(variant.group(1))),
                overlap_tokens=int(variant.group(2)),
            )
        elif strategy == "sentence_window":
            built = build_chunks(
                doc_name, iter(elements),
                max_words=a.max_words, overlap_words=a.overlap_words,
            )
        elif strategy == "structure":
            built = build_chunks_structure_aware(
                doc_name, iter(elements), token_counter=tc,
                max_tokens=cap(a.st_max_tokens or a.max_tokens),
            )
        else:
            raise ValueError(strategy)
        for c in built:
            chunks.append({
                "document": doc_name,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
                "contains_table": c.contains_table,
            })
    return chunks


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pages", required=True, help="pages.json dumped from the app DB")
    p.add_argument("--bench", default="../benchmarkv2.yaml")
    p.add_argument("--k", type=int, default=5, help="app generator reads top_n = 5")
    p.add_argument("--max-tokens", type=int, default=400, help="fixed_token budget")
    p.add_argument("--overlap-tokens", type=int, default=40)
    p.add_argument("--max-words", type=int, default=310, help="control cell size")
    p.add_argument("--overlap-words", type=int, default=62)
    p.add_argument("--st-max-tokens", type=int, default=0,
                   help="structure budget; 0 = same as --max-tokens")
    p.add_argument("--strategies", default="fixed_token,sentence_window,structure")
    p.add_argument("--sweep", default="",
                   help="extra fixed_token budgets, comma list of t{budget}_o{overlap} "
                        "specs, e.g. t1000_o40,t2048_o205; each adds strategy "
                        "fixed_token_t{budget}_o{overlap}")
    p.add_argument("--embedders", default="app,gemma",
                   help="app = MiniLM app copy; gemma = EmbeddingGemma-300m ONNX")
    p.add_argument("--gemma-dtype", default="quantized",
                   choices=["fp32", "fp16", "quantized", "q4", "q4f16"])
    p.add_argument("--out", default="results.json")
    a = p.parse_args()

    docs = load_pages(a.pages)
    queries = load_queries(a.bench)

    app_svc = EmbeddingsService()
    embedders = {}
    for name in a.embedders.split(","):
        if name == "app":
            embedders[name] = app_svc
        elif name == "gemma":
            embedders[name] = GemmaEmbeddingsService(dtype=a.gemma_dtype)
        else:
            raise ValueError(name)

    # Chunks are cut per embedder, in that embedder's own tokens, with the
    # budget capped to its usable window, so nothing is truncated at embed
    # time and every budget spec means the same thing the app's ingest means.
    strategies = [s for s in a.strategies.split(",") if s]
    strategies += [f"fixed_token_{s}" for s in a.sweep.split(",") if s]

    results = {}
    for e_name, svc in embedders.items():
        e_tc = await svc.get_token_counter()
        prefix_toks = (e_tc(svc.DOCUMENT_PREFIX)
                       if hasattr(svc, "DOCUMENT_PREFIX") else 0)
        cap = lambda t, w=svc._max_length, p=prefix_toks: usable_budget(t, w, p)
        q_vecs = np.array(await embed(svc, [q["text"] for q in queries], "query"))
        for strategy in strategies:
            chunks = build_all(docs, strategy, e_tc, a, budget_cap=cap)
            vecs = np.array(await embed(svc, [c["text"] for c in chunks], "document"))
            sims = q_vecs @ vecs.T  # vectors are L2-normalized by both services
            per_q = {}
            for qi, q in enumerate(queries):
                order = np.argsort(-sims[qi])[: max(a.k, 50)]
                ranked = [chunks[i] for i in order]
                n_rel = sum(1 for c in chunks if chunk_is_relevant(c, q["gold"]))
                per_q[q["id"]] = score_ranking(ranked, q["gold"], a.k, n_rel)
            agg = {m: sum(s[m] for s in per_q.values()) / len(per_q)
                   for m in ("ndcg", "recall_pages", "mrr", "hit")}
            toks = [e_tc(c["text"]) for c in chunks]
            agg.update(
                chunks=len(chunks),
                avg_tokens=round(sum(toks) / len(toks), 1),
                over_embed_window=sum(
                    1 for t in toks
                    if t > svc._max_length - prefix_toks - 2),
                tables=sum(1 for c in chunks if c["contains_table"]),
            )
            key = f"{strategy}+{e_name}"
            results[key] = {"aggregate": agg, "per_question": per_q}
            print(f"{key:24s} " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in agg.items()))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    ranked = sorted(results, key=lambda s: (results[s]["aggregate"]["ndcg"],
                                            results[s]["aggregate"]["recall_pages"]),
                    reverse=True)
    print(f"winner by nDCG@{a.k} (recall tiebreak): {ranked[0]}")
    print(f"scored {len(queries)} gold questions -> {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
