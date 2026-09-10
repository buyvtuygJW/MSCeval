"""
Benchmark (gold set) loader.

The benchmark is authored by the researcher and lives OUTSIDE the database - it
holds the gold answers, relevance judgments (qrels), and the answerable /
unanswerable / outdated labels the metrics need. The app stores none of this,
so it cannot be derived from logs; it is supplied as a YAML/JSON file.

Schema (benchmark.yaml)::

    queries:
      - id: q001
        text: "What fire door rating does the assessment record for the stairwell?"
        category: direct_lookup          # free-form label
        answerable: true                 # false => positive class for abstention
        gold_answer: "FD30 ..."          # for answer-accuracy (optional)
        gold_evidence:                   # what the answer should cite => IR qrels
          - document: "EICR 2.pdf"       # documents.filename
            page: 6                      # 1-based, metadata unless page_window set
            text: "4.16 Protection against electromagnetic effects ..."
        governing_doc: "<current>.pdf"   # optional, only for a corpus that ships
        superseded_doc: "<older>.pdf"    # both versions; unused here, so unscored
      - id: q002
        text: "..."
        answerable: false                # unanswerable => must abstain

Only ``id`` and ``text`` are strictly required. ``answerable`` defaults to True.

Relevance is authored as quoted text only: document, 1-based page, text.
Raw ``chunks.id`` UUIDs are not accepted anywhere in this file, because the app
writes fresh chunk rows on every ingest and the 2x2 flips the chunker, so a UUID
list is stale the moment the corpus is re-uploaded and wrong in three of the
four cells. ``veridic_eval.gold_evidence`` matches that text to chunk ids of
whichever ingest is live at the moment a cell is dumped, which is the only point
where a chunk id is known to exist. A ``gold_chunk_ids`` key raises here rather
than being ignored, so an old benchmark file cannot score a silent zero.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkQuery:
    id: str
    text: str
    category: str = "uncategorised"
    answerable: bool = True
    gold_answer: Optional[str] = None
    # the only qrels input: [{document, page, text, ...}] matched by
    # gold_evidence.py against the live ingest, per cell, at dump time
    gold_evidence: List[dict] = field(default_factory=list)
    governing_doc: Optional[str] = None
    superseded_doc: Optional[str] = None
    # optional explicit per-condition message linkage: {condition_name: message_id}
    message_ids: Dict[str, str] = field(default_factory=dict)


def _normalise_query(raw: dict) -> BenchmarkQuery:
    if "gold_chunk_ids" in raw:
        raise ValueError(
            f"query {raw.get('id')!r}: gold_chunk_ids is not supported. Chunk "
            "UUIDs are recreated by every ingest, so a pasted list cannot be "
            "the same after a re-upload and is wrong in the other cells. "
            "Author the evidence as gold_evidence: [{document, page, text}]."
        )
    if "gold_anchors" in raw:
        raise ValueError(
            f"query {raw.get('id')!r}: gold_anchors was renamed to gold_evidence, "
            "and its `span:` field to `text:`. Rename both rather than leaving "
            "the old key, which would be read as no gold at all."
        )
    return BenchmarkQuery(
        id=str(raw["id"]),
        text=str(raw["text"]).strip(),
        category=str(raw.get("category", "uncategorised")),
        answerable=bool(raw.get("answerable", True)),
        gold_answer=raw.get("gold_answer"),
        gold_evidence=[dict(a) for a in (raw.get("gold_evidence") or [])],
        governing_doc=raw.get("governing_doc"),
        superseded_doc=raw.get("superseded_doc"),
        message_ids={str(k): str(v) for k, v in (raw.get("message_ids") or {}).items()},
    )


def load_benchmark(path: str) -> List[BenchmarkQuery]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. "
            "Author one (see veridic_eval/benchmark.py docstring) or set "
            "VERIDIC_EVAL_BENCHMARK."
        )
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            import yaml  # lazy: only needed for yaml benchmarks

            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)

    queries = data.get("queries", data) if isinstance(data, dict) else data
    if not isinstance(queries, list):
        raise ValueError("Benchmark must contain a list under `queries`.")
    return [_normalise_query(q) for q in queries]


#: Characters trimmed from both ends of a question before it is matched to a
#: logged turn. A question pasted into the chat keeps whatever quotes it was
#: copied inside and the app stores the turn verbatim, so the two texts differ
#: by those characters alone: the qdora q002 turn `ac4ab100` carries a trailing
#: '"' and is identical otherwise, which cost that cell its ninth scored answer.
MATCH_TRIM_CHARS = " \t\r\n\"'`“”‘’«»"

#: The same set plus sentence punctuation, looser than the linker uses. A turn
#: that matches only here is a wording difference to print, not a link to make.
MATCH_TRIM_CHARS_LOOSE = MATCH_TRIM_CHARS + "?!.,"


def match_key(
    s: str,
    *,
    lower: bool = True,
    collapse_whitespace: bool = True,
    trim_chars: str = MATCH_TRIM_CHARS,
    trim_left: bool = True,
    trim_right: bool = True,
) -> str:
    """The comparison key that matches a benchmark question to a logged turn.

    `extract.content_match_sql` builds the SQL side from these same arguments,
    so the two halves of the comparison cannot drift apart.

    Args:
        lower: case-fold, since the app stores the turn exactly as typed.
        collapse_whitespace: every run of whitespace becomes one space and the
            ends are cleared, which covers a soft-wrapped or re-indented paste.
        trim_chars: characters stripped from the ends afterwards, repeatedly.
            An empty string leaves the ends alone and gives the strict form.
        trim_left / trim_right: which end `trim_chars` applies to. A question
            pasted inside quotes carries one at each end; a stray key while
            sending carries one only at the end that is still being typed.
    """
    out = s or ""
    if lower:
        out = out.lower()
    if collapse_whitespace:
        out = " ".join(out.split())
    if trim_chars:
        if trim_left and trim_right:
            out = out.strip(trim_chars)
        elif trim_left:
            out = out.lstrip(trim_chars)
        elif trim_right:
            out = out.rstrip(trim_chars)
    return out
