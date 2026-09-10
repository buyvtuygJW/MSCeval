"""
Gold anchors: condition-stable relevance judgments for the IR metrics.

Raw ``chunks.id`` UUIDs cannot carry the qrels in this study. The app writes new
chunk rows on every ingest, and the 2x2 design flips the chunker (RCTS vs
optimised), so a UUID list copied from one ingest is wrong in the other three
cells. An anchor instead names evidence the way the source PDF does (document
filename, 1-based page, verbatim span) and is resolved against whatever chunk
rows the current ingest produced, so the same anchors serve every cell.

Span source is the PPOCRLabel ``Label.txt`` shipped with the gold dataset, one
line per page::

    pages/page_0001.png<TAB>[{"transcription": "...", "points": [[x,y], ...],
                             "difficult": false, "key_cls": "paragraph",
                             "section_id": 1, "section_title": "..."}, ...]

Two properties of the live corpus drive the defaults here:

* ``chunks.page_number`` is the page a chunk *starts* on, and chunks span several
  pages (``EICR 2.pdf`` holds 6 chunks of ~6 kB covering 12 pages, recorded at
  pages 1, 4, 6, 7, 9, 12). Filtering candidates by page equality therefore
  discards the correct chunk, so ``page_window=None`` (text decides) is the
  default and page is metadata unless the caller opts in.
* Label.txt spans are Tesseract OCR at 144 dpi with per-page ``min_ocr_conf``
  down to 30, while chunk text comes from the app's own extraction, and tables
  arrive as reconstructed markup on both sides. Exact substring equality is
  therefore a best case, not a contract, hence the shingle-coverage fallback.

Nothing in this module writes to the database or to the benchmark file.
"""
from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text as sql_text

from .db import session_scope

# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_PAGE_NUM_RE = re.compile(r"(\d+)")

DEFAULT_KEEP_PUNCT = "./%:"
"""Kept because regulation references and measurements are the discriminating
tokens in this corpus (``411.3.3``, ``0.39``, ``93.09%``, ``522.6.202``)."""


def normalise_for_match(
    s: str,
    *,
    case_fold: bool = True,
    unescape_html: bool = True,
    strip_tags: bool = True,
    keep_punct: str = DEFAULT_KEEP_PUNCT,
    collapse_ws: bool = True,
    drop_table_pipes: bool = True,
) -> str:
    """Fold OCR text and chunk text onto one comparable surface.

    Every step is separately switchable so a caller can tighten matching for a
    clean text layer or loosen it for a noisy scan without editing this file.
    """
    out = s or ""
    if unescape_html:
        out = html.unescape(html.unescape(out))  # Label.txt double-escapes &#x27;
    if strip_tags:
        out = _TAG_RE.sub(" ", out)
    if case_fold:
        out = out.casefold()
    if drop_table_pipes:
        out = out.replace("|", " ")
    kept = []
    for ch in out:
        if ch.isalnum() or ch.isspace() or ch in keep_punct:
            kept.append(ch)
        else:
            kept.append(" ")
    out = "".join(kept)
    if collapse_ws:
        out = " ".join(out.split())
    return out


def _tokens(normalised: str) -> List[str]:
    return normalised.split()


def _shingles(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 1 or len(tokens) <= n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------


@dataclass
class GoldAnchor:
    """One labelled evidence span, addressed independently of any ingest."""

    anchor_id: str
    document: str                      # documents.filename in the app DB
    span: str                          # verbatim Label.txt transcription
    page: Optional[int] = None         # 1-based, hint only by default
    key_cls: Optional[str] = None
    section_id: Optional[int] = None
    section_title: Optional[str] = None
    source: Optional[str] = None       # Label.txt path this came from


DEFAULT_EXCLUDE_KEY_CLS = ("page_header", "page_footer")
"""Running headers and footers repeat on every page ("OSE Test Results 2015 09
22 EICR Example EICR"), so they match everywhere and judge nothing."""


def load_label_anchors(
    label_path: str,
    *,
    document: str,
    page_offset: int = 0,
    pages: Optional[Iterable[int]] = None,
    include_key_cls: Optional[Sequence[str]] = None,
    exclude_key_cls: Sequence[str] = DEFAULT_EXCLUDE_KEY_CLS,
    skip_difficult: bool = True,
    min_span_chars: int = 30,
    max_span_chars: Optional[int] = None,
    section_ids: Optional[Iterable[int]] = None,
    anchor_id_prefix: Optional[str] = None,
    strict: bool = True,
) -> List[GoldAnchor]:
    """Read a PPOCRLabel ``Label.txt`` into anchors for one document.

    Args:
        document: ``documents.filename`` this label file describes. Verified
            mapping for the ingested corpus: ``eicr2pages`` -> ``EICR 2.pdf``,
            ``fra2pages`` -> ``FRA2.pdf``, ``hands1pages`` -> ``HandS1_img.pdf``.
        page_offset: added to the number parsed from ``page_0001.png``; use it
            when the labelled render dropped a cover page.
        pages: keep only these 1-based pages (after offset).
        include_key_cls: whitelist; ``None`` keeps everything not excluded.
        min_span_chars: short spans ("N/A", "44.4") match dozens of chunks and
            produce false qrels, so they are dropped by default.
        section_ids: keep only spans from these Label.txt sections.
        strict: raise on a malformed line instead of skipping it.
    """
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    want_pages = set(pages) if pages is not None else None
    want_sections = set(section_ids) if section_ids is not None else None
    include = tuple(include_key_cls) if include_key_cls else None
    prefix = anchor_id_prefix if anchor_id_prefix is not None else os.path.basename(
        os.path.dirname(os.path.abspath(label_path))
    )

    anchors: List[GoldAnchor] = []
    with open(label_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            img, _, payload = line.partition("\t")
            if not payload:
                if strict:
                    raise ValueError(f"{label_path}:{lineno} has no tab separator")
                continue
            found = _PAGE_NUM_RE.findall(os.path.basename(img))
            page = (int(found[-1]) + page_offset) if found else None
            if want_pages is not None and page not in want_pages:
                continue
            try:
                regions = json.loads(payload)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"{label_path}:{lineno} bad JSON: {exc}") from exc
                continue
            for idx, reg in enumerate(regions):
                span = str(reg.get("transcription") or "")
                key_cls = reg.get("key_cls")
                sec_id = reg.get("section_id")
                if skip_difficult and bool(reg.get("difficult")):
                    continue
                if key_cls in exclude_key_cls:
                    continue
                if include is not None and key_cls not in include:
                    continue
                if want_sections is not None and sec_id not in want_sections:
                    continue
                if len(span.strip()) < min_span_chars:
                    continue
                if max_span_chars is not None and len(span.strip()) > max_span_chars:
                    continue
                anchors.append(
                    GoldAnchor(
                        anchor_id=f"{prefix}:p{page or 0:04d}:s{idx:03d}",
                        document=document,
                        span=span,
                        page=page,
                        key_cls=key_cls,
                        section_id=sec_id,
                        section_title=reg.get("section_title"),
                        source=label_path,
                    )
                )
    return anchors


def anchors_from_benchmark(
    raw_anchors: Iterable[dict],
    *,
    query_id: str,
    default_document: Optional[str] = None,
) -> List[GoldAnchor]:
    """Build anchors from inline ``gold_anchors:`` entries in benchmark.yaml.

    Each entry needs ``span`` plus a ``document`` (or a ``default_document``);
    ``page``, ``key_cls``, ``section_id`` are optional metadata.
    """
    out: List[GoldAnchor] = []
    for i, raw in enumerate(raw_anchors or []):
        doc = raw.get("document") or default_document
        if not doc:
            raise ValueError(f"{query_id}: gold anchor {i} has no document")
        span = str(raw.get("span") or "").strip()
        if not span:
            raise ValueError(f"{query_id}: gold anchor {i} has no span")
        page = raw.get("page")
        out.append(
            GoldAnchor(
                anchor_id=str(raw.get("anchor_id") or f"{query_id}:a{i:02d}"),
                document=str(doc),
                span=span,
                page=int(page) if page is not None else None,
                key_cls=raw.get("key_cls"),
                section_id=raw.get("section_id"),
                section_title=raw.get("section_title"),
                source="benchmark",
            )
        )
    return out


# --------------------------------------------------------------------------
# chunk fetch
# --------------------------------------------------------------------------


@dataclass
class ChunkRow:
    chunk_id: str
    document: str
    page_number: int
    chunk_index: int
    text: str
    normalised: str = ""


_DOC_PREDICATES = {
    "exact": "d.filename = :doc",
    "casefold": "lower(d.filename) = lower(:doc)",
    "basename": "lower(regexp_replace(d.filename, '^.*[/\\\\]', '')) = lower(:doc)",
    "contains": "d.filename ILIKE '%' || :doc || '%'",
}


def fetch_document_chunks(
    sess,
    document: str,
    *,
    document_match: str = "exact",
    document_status: Optional[Sequence[str]] = ("completed",),
    normalise: Optional[Callable[[str], str]] = None,
) -> List[ChunkRow]:
    """All chunks of one document in reading order, newest ingest included.

    ``document_status`` guards against scoring against a half-ingested copy of
    the same filename; pass ``None`` to accept any status.
    """
    if document_match not in _DOC_PREDICATES:
        raise ValueError(
            f"document_match must be one of {sorted(_DOC_PREDICATES)}, got {document_match!r}"
        )
    where = [_DOC_PREDICATES[document_match]]
    params: Dict[str, object] = {"doc": document}
    if document_status:
        where.append("d.status = ANY(:statuses)")
        params["statuses"] = list(document_status)
    stmt = sql_text(
        "SELECT c.id, d.filename, c.page_number, c.chunk_index, c.text "
        "FROM chunks c JOIN documents d ON d.id = c.document_id "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY c.page_number ASC, c.chunk_index ASC"
    )
    norm = normalise or normalise_for_match
    rows = sess.execute(stmt, params).fetchall()
    return [
        ChunkRow(
            chunk_id=str(r[0]),
            document=str(r[1]),
            page_number=int(r[2]),
            chunk_index=int(r[3]),
            text=r[4] or "",
            normalised=norm(r[4] or ""),
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


@dataclass
class AnchorMatch:
    anchor_id: str
    chunk_id: str
    document: str
    chunk_page: int
    chunk_index: int
    score: float
    method: str          # "containment" | "shingle" | "token"


@dataclass
class ResolutionReport:
    """Auditable record of how gold anchors became chunk ids."""

    matches: Dict[str, List[AnchorMatch]] = field(default_factory=dict)
    unresolved: Dict[str, str] = field(default_factory=dict)   # anchor_id -> reason
    chunks_seen: Dict[str, int] = field(default_factory=dict)  # document -> n chunks

    def chunk_ids(self, anchor_ids: Optional[Iterable[str]] = None) -> List[str]:
        """Deduplicated chunk ids, first-seen order, for the given anchors."""
        wanted = list(anchor_ids) if anchor_ids is not None else list(self.matches)
        out: List[str] = []
        for aid in wanted:
            for m in self.matches.get(aid, []):
                if m.chunk_id not in out:
                    out.append(m.chunk_id)
        return out

    def summary(self) -> dict:
        n_multi = sum(1 for v in self.matches.values() if len(v) > 1)
        return {
            "anchors_resolved": len(self.matches),
            "anchors_unresolved": len(self.unresolved),
            "anchors_multi_chunk": n_multi,
            "distinct_chunks": len(self.chunk_ids()),
            "chunks_seen": dict(self.chunks_seen),
            "unresolved_reasons": dict(self.unresolved),
        }


def resolve_anchors(
    anchors: Sequence[GoldAnchor],
    *,
    sess=None,
    session_factory: Callable = session_scope,
    document_match: str = "exact",
    document_status: Optional[Sequence[str]] = ("completed",),
    normalise: Optional[Callable[[str], str]] = None,
    match_mode: str = "containment_then_shingle",
    shingle_n: int = 5,
    min_shingle_coverage: float = 0.6,
    min_token_coverage: float = 0.9,
    page_window: Optional[int] = None,
    page_required: bool = False,
    max_chunks_per_anchor: int = 0,
    min_score: float = 0.0,
    tie_margin: float = 0.0,
    on_unresolved: str = "collect",
    chunk_cache: Optional[Dict[str, List[ChunkRow]]] = None,
) -> ResolutionReport:
    """Map anchors onto the chunk ids of the ingest that is live right now.

    Args:
        sess: existing read-only session; when ``None`` one is opened per call.
        match_mode: ``containment`` (normalised span must be a substring),
            ``shingle`` (token n-gram coverage only), ``containment_then_shingle``
            (substring first, coverage as fallback), or ``token`` (bag-of-tokens
            coverage, last resort for heavy OCR noise).
        shingle_n: n-gram length; lower it for short spans, raise it to cut
            accidental matches in repetitive checklist tables.
        min_shingle_coverage: fraction of the span's n-grams that must appear in
            the chunk.
        min_token_coverage: threshold for ``token`` mode and for spans shorter
            than ``shingle_n`` tokens.
        page_window: if set, only chunks whose recorded start page lies within
            ``anchor.page +- page_window`` are considered. Default ``None``
            because chunks span pages and record only their first page.
        page_required: drop anchors with no page instead of matching on text.
        max_chunks_per_anchor: cap the qrels per anchor (0 = keep all above
            threshold). A span legitimately lands in 2 chunks when the chunker
            overlaps windows.
        tie_margin: keep only matches within this score margin of the best one.
        on_unresolved: ``collect`` (record and continue) or ``raise``.

    Returns:
        ResolutionReport. Unresolved anchors are reported, never silently
        dropped: missing gold deflates recall without any error surfacing.
    """
    valid_modes = ("containment", "shingle", "containment_then_shingle", "token")
    if match_mode not in valid_modes:
        raise ValueError(f"match_mode must be one of {valid_modes}, got {match_mode!r}")
    if on_unresolved not in ("collect", "raise"):
        raise ValueError("on_unresolved must be 'collect' or 'raise'")

    norm = normalise or normalise_for_match
    cache: Dict[str, List[ChunkRow]] = chunk_cache if chunk_cache is not None else {}
    report = ResolutionReport()

    def _run(session) -> None:
        for a in anchors:
            if a.document not in cache:
                cache[a.document] = fetch_document_chunks(
                    session,
                    a.document,
                    document_match=document_match,
                    document_status=document_status,
                    normalise=norm,
                )
                report.chunks_seen[a.document] = len(cache[a.document])
            chunks = cache[a.document]
            if not chunks:
                _fail(a, f"no chunks for document {a.document!r}")
                continue
            if page_required and a.page is None:
                _fail(a, "page_required=True and anchor has no page")
                continue

            candidates = chunks
            if page_window is not None and a.page is not None:
                candidates = [
                    c for c in chunks
                    if a.page - page_window <= c.page_number <= a.page + page_window
                ]
                if not candidates:
                    _fail(a, f"no chunk within page_window={page_window} of page {a.page}")
                    continue

            span_n = norm(a.span)
            if not span_n:
                _fail(a, "span is empty after normalisation")
                continue
            span_tokens = _tokens(span_n)
            span_shingles = set(_shingles(span_tokens, shingle_n))
            span_token_set = set(span_tokens)

            scored: List[AnchorMatch] = []
            for c in candidates:
                score, method = 0.0, ""
                if match_mode in ("containment", "containment_then_shingle"):
                    if span_n and span_n in c.normalised:
                        score, method = 1.0, "containment"
                if not method and match_mode in ("shingle", "containment_then_shingle"):
                    if span_shingles and len(span_tokens) >= shingle_n:
                        chunk_shingles = set(_shingles(_tokens(c.normalised), shingle_n))
                        cov = len(span_shingles & chunk_shingles) / len(span_shingles)
                        if cov >= min_shingle_coverage:
                            score, method = cov, "shingle"
                if not method and (
                    match_mode == "token"
                    or (
                        match_mode in ("shingle", "containment_then_shingle")
                        and len(span_tokens) < shingle_n
                    )
                ):
                    if span_token_set:
                        chunk_tokens = set(_tokens(c.normalised))
                        cov = len(span_token_set & chunk_tokens) / len(span_token_set)
                        if cov >= min_token_coverage:
                            score, method = cov, "token"
                if method and score >= min_score:
                    scored.append(
                        AnchorMatch(
                            anchor_id=a.anchor_id,
                            chunk_id=c.chunk_id,
                            document=c.document,
                            chunk_page=c.page_number,
                            chunk_index=c.chunk_index,
                            score=score,
                            method=method,
                        )
                    )

            if not scored:
                _fail(
                    a,
                    f"no chunk matched (mode={match_mode}, shingle_n={shingle_n}, "
                    f"min_shingle_coverage={min_shingle_coverage})",
                )
                continue

            scored.sort(key=lambda m: (-m.score, m.chunk_page, m.chunk_index))
            if tie_margin > 0.0:
                best = scored[0].score
                scored = [m for m in scored if best - m.score <= tie_margin]
            if max_chunks_per_anchor > 0:
                scored = scored[:max_chunks_per_anchor]
            report.matches[a.anchor_id] = scored

    def _fail(a: GoldAnchor, reason: str) -> None:
        if on_unresolved == "raise":
            raise LookupError(f"{a.anchor_id}: {reason}")
        report.unresolved[a.anchor_id] = reason

    if sess is not None:
        _run(sess)
    else:
        with session_factory() as session:
            _run(session)
    return report


# --------------------------------------------------------------------------
# benchmark wiring
# --------------------------------------------------------------------------


def resolve_query_gold(
    queries: Sequence[object],
    *,
    default_document: Optional[str] = None,
    require_all: bool = False,
    **resolve_kwargs,
) -> Tuple[Dict[str, List[str]], ResolutionReport]:
    """Per-query gold chunk ids for the live ingest, from inline anchors.

    Reads ``query.gold_anchors`` (list of dicts) and ignores queries without
    any. Returns ``{query_id: [chunk_id, ...]}`` plus the audit report.

    Args:
        require_all: raise if any anchor of any query failed to resolve. Use it
            on the final thesis run: an unresolved anchor is scored as a
            retrieval miss, which reads as a recall drop that is not real.
    """
    per_query_anchors: Dict[str, List[GoldAnchor]] = {}
    flat: List[GoldAnchor] = []
    for q in queries:
        raw = getattr(q, "gold_anchors", None) or []
        if not raw:
            continue
        built = anchors_from_benchmark(raw, query_id=getattr(q, "id"), default_document=default_document)
        per_query_anchors[getattr(q, "id")] = built
        flat.extend(built)

    if not flat:
        return {}, ResolutionReport()

    report = resolve_anchors(flat, **resolve_kwargs)
    if require_all and report.unresolved:
        raise LookupError(
            f"{len(report.unresolved)} gold anchors unresolved: {report.unresolved}"
        )
    return (
        {
            qid: report.chunk_ids([a.anchor_id for a in anchors])
            for qid, anchors in per_query_anchors.items()
        },
        report,
    )


def apply_gold_to_records(
    records_by_condition: Dict[str, list],
    queries: Sequence[object],
    **resolve_kwargs,
) -> Dict[str, ResolutionReport]:
    """Fill ``rec.judged_chunk_ids`` in place, per condition, from anchors.

    Call this once per condition *after* that cell's corpus is the live ingest;
    chunk ids are only valid for the ingest that is in the database at the time
    of the call. The assignment is unconditional, so a record can never keep an
    id that belongs to a different ingest: a query whose anchors resolve to
    nothing ends up with an empty list and is left unjudged.
    """
    reports: Dict[str, ResolutionReport] = {}
    for cond_name, records in records_by_condition.items():
        gold, report = resolve_query_gold(queries, **resolve_kwargs)
        reports[cond_name] = report
        for rec in records:
            rec.judged_chunk_ids = list(gold.get(getattr(rec, "query_id"), []))
    return reports
