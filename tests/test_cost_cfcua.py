import pytest

from veridic_eval.cost_cfcua import compute_cfcua, is_cited, is_right_version
from veridic_eval.extract import QueryRecord, ServedEvidence


def _ev(doc, rank, chunk="c"):
    return ServedEvidence(chunk_id=f"{doc}-{rank}", document_name=doc, page=1,
                          retrieval_score=0.5, rerank_score=0.9, served_rank=rank,
                          chunk_text=chunk)


def _rec(qid, answerable=True, evidence=None, governing=None, superseded=None,
         provider="ollama"):
    return QueryRecord(
        query_id=qid, condition="c", question="?", answerable=answerable,
        category="x", judged_chunk_ids=[], gold_answer=None,
        governing_doc=governing, superseded_doc=superseded,
        answer_text="ans", served_evidence=evidence or [], linked=True,
        provider=provider, tokens_total=100,
    )


def test_is_cited():
    assert is_cited(_rec("q", evidence=[_ev("A", 0)])) == 1
    assert is_cited(_rec("q", evidence=[])) == 0


def test_right_version_not_applicable_passes():
    assert is_right_version(_rec("q", evidence=[_ev("A", 0)])) == 1


def test_right_version_governing_served_first():
    ev = [_ev("GAS_2024.pdf", 0), _ev("GAS_2019.pdf", 1)]
    rec = _rec("q", evidence=ev, governing="GAS_2024.pdf", superseded="GAS_2019.pdf")
    assert is_right_version(rec) == 1


def test_right_version_superseded_ranked_above():
    ev = [_ev("GAS_2019.pdf", 0), _ev("GAS_2024.pdf", 1)]
    rec = _rec("q", evidence=ev, governing="GAS_2024.pdf", superseded="GAS_2019.pdf")
    assert is_right_version(rec) == 0


def test_right_version_governing_absent():
    rec = _rec("q", evidence=[_ev("OTHER.pdf", 0)], governing="GAS_2024.pdf")
    assert is_right_version(rec) == 0


def test_cfcua_local_ollama_zero_cost():
    recs = [_rec("q1", evidence=[_ev("A", 0)]), _rec("q2", evidence=[_ev("B", 0)])]
    faith = {"q1": {"faithful": 1}, "q2": {"faithful": 1}}
    out = compute_cfcua(recs, faith)
    assert out["p_faithful_cited_version"] == pytest.approx(1.0)
    assert out["cost_per_answer_usd"] == 0.0
    # Local/Ollama => 0 per-answer API cost; CFCUA = 0.0 / 1.0 = 0.0 (defined).
    assert out["cfcua"] == pytest.approx(0.0)
    assert out["note"] is None


def test_cfcua_with_supplied_cost():
    recs = [_rec("q1", evidence=[_ev("A", 0)]),
            _rec("q2", evidence=[_ev("B", 0)])]
    faith = {"q1": {"faithful": 1}, "q2": {"faithful": 0}}  # P = 0.5
    out = compute_cfcua(recs, faith, per_answer_cost=0.01)
    assert out["p_faithful_cited_version"] == pytest.approx(0.5)
    assert out["cfcua"] == pytest.approx(0.02)  # 0.01 / 0.5
    assert out["cost_source"].startswith("supplied")


def test_cfcua_undefined_when_p_zero():
    recs = [_rec("q1", evidence=[])]  # not cited => joint 0 => P=0
    faith = {"q1": {"faithful": 1}}
    out = compute_cfcua(recs, faith, per_answer_cost=0.01)
    assert out["p_faithful_cited_version"] == 0.0
    assert out["cfcua"] is None
    assert "undefined" in out["note"]
