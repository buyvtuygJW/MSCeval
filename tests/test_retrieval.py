import pytest

pytest.importorskip("ranx")

from veridic_eval.retrieval_eval import build_qrels_run, evaluate_retrieval  # noqa: E402
from veridic_eval.extract import QueryRecord, ServedEvidence  # noqa: E402


def _ev(chunk_id, rank):
    return ServedEvidence(chunk_id=chunk_id, document_name="D", page=1,
                          retrieval_score=0.5, rerank_score=0.9, served_rank=rank)


def _rec(qid, gold, served):
    return QueryRecord(
        query_id=qid, condition="c", question="?", answerable=True, category="x",
        judged_chunk_ids=gold, gold_answer=None, governing_doc=None, superseded_doc=None,
        served_evidence=served, linked=True,
    )


def test_build_qrels_run_skips_ungolded():
    recs = [
        _rec("q1", ["A"], [_ev("A", 0)]),
        _rec("q2", [], [_ev("B", 0)]),          # no gold => not judged
    ]
    qrels, run, judged = build_qrels_run(recs)
    assert judged == ["q1"]
    assert qrels["q1"] == {"A": 1}
    assert "q2" not in run


def test_build_qrels_run_placeholder_for_empty_served():
    recs = [_rec("q1", ["A"], [])]
    _, run, _ = build_qrels_run(recs)
    assert run["q1"] == {"__none__": 0.0}


def test_evaluate_retrieval_hit_and_miss():
    recs = [
        _rec("q1", ["A"], [_ev("A", 0), _ev("B", 1)]),   # hit at rank 0
        _rec("q2", ["Z"], [_ev("C", 0), _ev("D", 1)]),   # miss
    ]
    out = evaluate_retrieval(recs, k=5)
    assert out["n_judged"] == 2
    agg = out["aggregate"]
    assert agg["recall@5"] == pytest.approx(0.5)     # 1 of 2 queries
    assert agg["hit_rate@5"] == pytest.approx(0.5)
    assert agg["mrr@5"] == pytest.approx(0.5)        # (1/1 + 0)/2
    # per-query booleans available for stats.py
    assert out["per_query"]["recall@5"]["q1"] == pytest.approx(1.0)
    assert out["per_query"]["recall@5"]["q2"] == pytest.approx(0.0)
