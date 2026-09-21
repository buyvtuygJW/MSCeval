"""
Inter-rater / detector agreement: the raw agreement rate over two label columns.

Three checks need it, all the same computation over two
label columns:
  1. faithfulness detector vs a small hand-labelled human sample (the accuracy
     figure that justifies trusting the detector);
  2. main faithfulness detector vs the *other* candidate detector (the appendix
     agreement check that defuses the "result is detector-specific" objection);
  3. abstention `regex` vs `is-refusal` on flagged disagreements (human spot-check).

Cohen's kappa is absent on purpose: gold labelling is single-annotator, so no
inter-annotator agreement statistic is reported and nothing here computes one.

Implemented with the stdlib only (no sklearn/pandas), so it never pulls a heavy
dependency and never needs the DB.
"""
from __future__ import annotations

import csv
from typing import Dict, List, Optional, Sequence


def raw_agreement(a: Sequence, b: Sequence) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def agreement_report(a: Sequence, b: Sequence, label_a: str = "A", label_b: str = "B") -> Dict:
    """The two figures reported: n and the raw agreement rate. The rate is
    emitted unrounded — 2/3 stays 2/3 — and any rounding is display-side."""
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n": sum(1 for x, y in zip(a, b) if x is not None and y is not None),
        "raw_agreement": raw_agreement(a, b),
    }


def _coerce(v: str):
    s = (v or "").strip()
    low = s.lower()
    if low in {"true", "1", "yes", "faithful", "y"}:
        return 1
    if low in {"false", "0", "no", "hallucinated", "n"}:
        return 0
    return s or None


def agreement_from_csv(path: str, col_a: str, col_b: str) -> Dict:
    """Load two label columns from a CSV and report agreement."""
    a: List = []
    b: List = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if col_a not in (reader.fieldnames or []) or col_b not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {col_a!r} and {col_b!r}; "
                             f"found {reader.fieldnames}")
        for row in reader:
            a.append(_coerce(row.get(col_a, "")))
            b.append(_coerce(row.get(col_b, "")))
    return agreement_report(a, b, col_a, col_b)


def detector_agreement(per_query_a: Dict[str, Dict], per_query_b: Dict[str, Dict],
                       key: str = "faithful") -> Dict:
    """
    Agreement between two faithfulness backends over shared query ids, the
    second-detector appendix check. Feed the `per_query` dicts from two
    ``score_faithfulness(...)`` runs.
    """
    common = [q for q in per_query_a if q in per_query_b]
    a = [per_query_a[q].get(key) for q in common]
    b = [per_query_b[q].get(key) for q in common]
    return agreement_report(a, b, "detector_a", "detector_b")
