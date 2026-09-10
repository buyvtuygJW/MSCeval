import pytest

from veridic_eval.abstention import score_abstention
from veridic_eval.config import settings
from veridic_eval.extract import QueryRecord

MARKER = settings.abstention_marker


def _rec(qid, answerable, answer):
    return QueryRecord(
        query_id=qid, condition="c", question="?", answerable=answerable,
        category="x", judged_chunk_ids=[], gold_answer=None, governing_doc=None,
        superseded_doc=None, answer_text=answer, linked=True,
    )


def test_abstention_metrics_python_path():
    records = [
        _rec("u1", False, MARKER),                                   # correct refusal (marker)
        _rec("u2", False, "The documents do not contain relevant information."),  # refusal (net)
        _rec("a1", True, "The main stairwell requires FD30 fire doors."),         # answered
        _rec("a2", True, "I cannot answer, insufficient information in documents."),  # over-abstain
    ]
    out = score_abstention(records, use_promptfoo=False)
    agg = out["aggregate"]
    assert out["tool"] == "python"
    assert agg["abstention_recall"] == pytest.approx(1.0)          # 2/2 unanswerable refused
    assert agg["over_abstention_rate"] == pytest.approx(0.5)       # 1/2 answerable refused
    assert agg["abstention_precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_per_query_columns_split_by_class():
    records = [
        _rec("u1", False, MARKER),
        _rec("a1", True, "FD30 doors."),
    ]
    out = score_abstention(records, use_promptfoo=False)
    assert out["per_query"]["abstention_recall"] == {"u1": 1}
    assert out["per_query"]["over_abstention_rate"] == {"a1": 0}
