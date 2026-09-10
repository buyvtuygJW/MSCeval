"""
One-time ingest cost, measured off the database instead of typed by hand.

``onetime_gpu_hours`` in the CFCA is the wall clock a condition spent being
built once: parse, chunk, embed, index. `power_meter` never sees that work,
because it happens before a single question is asked, so until now the number
was whatever the yaml declared. The app stamps every document and every chunk it
writes, so those stamps are the measurement: `read_ingest_documents` pulls them,
`ingest_span` turns them into hours, `ingest_cost_inputs` hands the CFCA its two
one-time quantities, and `apply_measured_ingest` drops them into the cost blocks
the same way `apply_measured_power` drops in measured watts.

The database only ever holds the newest ingest: the chunking correction that
`chunk_opt` needs deletes and rewrites every row. A cell is therefore measured
while it is live, at dump time, and the numbers are read back off its snapshot
afterwards. Nothing here re-measures a cell whose ingest has been overwritten.

Every choice that changes the number is an argument: which stamps bound the
span, whether documents ingested in parallel are counted once or twice, and how
many characters make a token. Nothing is guessed silently.

The token side is logged as well, so it need not be estimated at all: the app's
chunker writes ``token_count`` into every chunk's metadata and the rag-service
logs ``corpus_tokens`` per build run. `read_chunk_token_counts` and
`read_index_build_logs` read those, ``tokens_total`` names which one, and the
character divisor stays as the fallback for a corpus that predates the
instrumentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .db import bind_cast, session_scope

#: How a document's ingest stretch is bounded.
#:
#: ``lifecycle``   ``documents.created_at`` -> ``documents.updated_at``: the whole
#:                 ingest of that file, queue wait included.
#: ``chunks``      first -> last ``chunks.created_at``: the chunk write only, which
#:                 floors at zero for a one-chunk document.
#: ``first_touch`` ``documents.created_at`` -> last ``chunks.created_at``: from the
#:                 upload to the last chunk landing, ignoring later status edits.
INGEST_BASES: Tuple[str, ...] = ("lifecycle", "chunks", "first_touch")

#: Rule of thumb for English prose, the fallback when the logged counts are not
#: there. The app does log them, so prefer ``tokens_total="chunks"`` or
#: ``tokens_total="index_build_logs"`` over this divisor, see `LOGGED_TOKEN_SOURCES`.
CHARS_PER_TOKEN: float = 4.0

INGEST_GPU_HOURS_KEY: str = "onetime_gpu_hours"
INGEST_EMBED_MTOK_KEY: str = "onetime_embed_mtok"

#: Where `resolve_logged_tokens` parks the whole `index_build_totals` block, so
#: one call that asked for a logged token count also carries the app's own
#: build seconds and the label that build ran under.
LOGGED_INDEX_BUILD_KEY: str = "index_build"

#: Which clock `merged_ingest_cost_inputs` bills.
#:
#: ``span``          the stretch between two database stamps, today's number.
#: ``logged``        the rag-service's own timed ``chunk_and_index()`` run.
#: ``prefer_logged`` the logged run when there is one, else the span.
#: ``prefer_span``   the span when it is measured, else the logged run.
HOURS_SOURCES: Tuple[str, ...] = ("span", "logged", "prefer_logged", "prefer_span")

#: What a policy label is allowed to differ by before `check_ingest_policies`
#: calls it a mismatch: the whole string, the family prefix, or the unit letter
#: that says which builder ran (``w`` words, ``t`` tokens).
POLICY_MATCHES: Tuple[str, ...] = ("exact", "prefix", "unit")


# --------------------------------------------------------------------------
# 1. read
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IngestDoc:
    """One document's ingest stamps, as the app left them."""

    doc_id: str
    filename: str
    status: str = ""
    page_count: Optional[int] = None
    n_chunks: int = 0
    chars: int = 0
    doc_created: Optional[datetime] = None
    doc_updated: Optional[datetime] = None
    first_chunk: Optional[datetime] = None
    last_chunk: Optional[datetime] = None

    def bounds(self, basis: str = "lifecycle") -> Optional[Tuple[datetime, datetime]]:
        """The document's stretch under one basis; None when a stamp is absent."""
        if basis not in INGEST_BASES:
            raise ValueError(f"basis must be one of {list(INGEST_BASES)}, got {basis!r}")
        if basis == "lifecycle":
            pair = (self.doc_created, self.doc_updated)
        elif basis == "chunks":
            pair = (self.first_chunk, self.last_chunk)
        else:
            pair = (self.doc_created, self.last_chunk)
        if pair[0] is None or pair[1] is None:
            return None
        return (pair[0], pair[1])

    def as_dict(self) -> Dict[str, Any]:
        def iso(ts: Optional[datetime]) -> Optional[str]:
            return ts.isoformat() if ts is not None else None

        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "status": self.status,
            "page_count": self.page_count,
            "n_chunks": self.n_chunks,
            "chars": self.chars,
            "doc_created": iso(self.doc_created),
            "doc_updated": iso(self.doc_updated),
            "first_chunk": iso(self.first_chunk),
            "last_chunk": iso(self.last_chunk),
        }


def _identifier(name: str, what: str) -> str:
    if not name.isidentifier():
        raise ValueError(f"{what} must be a bare identifier, got {name!r}")
    return name


def ingest_columns(
    *,
    tables: Sequence[str] = ("documents", "chunks"),
    schema: str = "public",
    session: Any = None,
    session_factory: Callable[[], Any] = session_scope,
) -> Dict[str, List[str]]:
    """
    The columns each ingest table actually has, ``{table: [column, ...]}``.

    One read-only look at ``information_schema`` so a missing stamp column is a
    sentence rather than a driver traceback. A table absent from the database
    comes back as an empty list.
    """
    from sqlalchemy import text as sql_text

    for t in tables:
        _identifier(t, "table name")
    stmt = sql_text(
        "SELECT table_name, column_name FROM information_schema.columns "
        f"WHERE table_schema = :schema AND table_name = ANY({bind_cast('tables', sql_type='text', array=True)}) "
        "ORDER BY table_name, ordinal_position"
    )
    params = {"schema": schema, "tables": list(tables)}
    out: Dict[str, List[str]] = {t: [] for t in tables}

    def _run(sess: Any) -> None:
        for row in sess.execute(stmt, params):
            out.setdefault(row.table_name, []).append(row.column_name)

    if session is not None:
        _run(session)
    else:
        with session_factory() as sess:
            _run(sess)
    return out


def require_ingest_columns(
    needed: Dict[str, Sequence[str]],
    *,
    present: Optional[Dict[str, List[str]]] = None,
    schema: str = "public",
    session: Any = None,
    session_factory: Callable[[], Any] = session_scope,
) -> Dict[str, List[str]]:
    """
    Raise unless every named column exists; return what is there.

    Args:
        needed: ``{table: [column, ...]}`` the read depends on.
        present: a previous `ingest_columns` result, to skip the second query.
    """
    have = present if present is not None else ingest_columns(
        tables=tuple(needed), schema=schema, session=session, session_factory=session_factory,
    )
    missing = {
        t: [c for c in cols if c not in set(have.get(t) or ())]
        for t, cols in needed.items()
    }
    missing = {t: cols for t, cols in missing.items() if cols}
    if missing:
        detail = "; ".join(f"{t}: {', '.join(cols)}" for t, cols in sorted(missing.items()))
        raise ValueError(f"ingest stamps absent from the database ({detail}). "
                         f"Name the real columns, or declare onetime_gpu_hours by hand.")
    return have


def read_ingest_documents(
    *,
    filenames: Optional[Sequence[str]] = None,
    statuses: Optional[Sequence[str]] = ("completed",),
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    documents_table: str = "documents",
    chunks_table: str = "chunks",
    id_column: str = "id",
    chunk_id_column: str = "id",
    document_id_column: str = "document_id",
    created_column: str = "created_at",
    updated_column: str = "updated_at",
    chunk_created_column: str = "created_at",
    chunk_text_column: str = "text",
    filename_column: str = "filename",
    status_column: str = "status",
    page_count_column: str = "page_count",
    schema: str = "public",
    check_columns: bool = True,
    assume_tz: Optional[timezone] = timezone.utc,
    session: Any = None,
    session_factory: Callable[[], Any] = session_scope,
) -> List[IngestDoc]:
    """
    Every document's ingest stamps and chunk totals, one row per document.

    Read-only, one query, grouped in the database rather than in Python so a
    corpus of any size costs one round trip.

    Args:
        filenames: restrict to these files; None takes the whole corpus. The
            served-document scope of a cell goes here.
        statuses: document statuses that count as ingested; None takes every
            status, which will include a failed half-ingest.
        since / until: bound on the document's own ``created_at``, for a
            corpus that carries files from an earlier build.
        documents_table ... page_count_column: names, for a schema that differs.
        check_columns: verify the stamp columns exist before querying.
        assume_tz: timezone for a naive stamp; None leaves it naive.
        session: an open read-only session; None opens one per call.
    """
    from sqlalchemy import text as sql_text

    docs_t = _identifier(documents_table, "documents_table")
    chunks_t = _identifier(chunks_table, "chunks_table")
    cols = {
        "id_column": id_column,
        "chunk_id_column": chunk_id_column,
        "document_id_column": document_id_column,
        "created_column": created_column,
        "updated_column": updated_column,
        "chunk_created_column": chunk_created_column,
        "chunk_text_column": chunk_text_column,
        "filename_column": filename_column,
        "status_column": status_column,
        "page_count_column": page_count_column,
    }
    for what, name in cols.items():
        _identifier(name, what)

    if check_columns:
        require_ingest_columns(
            {
                docs_t: [id_column, filename_column, status_column, page_count_column,
                         created_column, updated_column],
                chunks_t: [chunk_id_column, document_id_column, chunk_created_column,
                           chunk_text_column],
            },
            schema=schema, session=session, session_factory=session_factory,
        )

    where: List[str] = []
    params: Dict[str, Any] = {}
    if statuses is not None:
        where.append(f"d.{status_column} = ANY({bind_cast('statuses', sql_type='text', array=True)})")
        params["statuses"] = list(statuses)
    if filenames is not None:
        where.append(f"d.{filename_column} = ANY({bind_cast('filenames', sql_type='text', array=True)})")
        params["filenames"] = list(filenames)
    if since is not None:
        where.append(f"d.{created_column} >= :since")
        params["since"] = since
    if until is not None:
        where.append(f"d.{created_column} <= :until")
        params["until"] = until
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    stmt = sql_text(
        f"""
        SELECT CAST(d.{id_column} AS text) AS doc_id,
               d.{filename_column}   AS filename,
               d.{status_column}     AS status,
               d.{page_count_column} AS page_count,
               d.{created_column}    AS doc_created,
               d.{updated_column}    AS doc_updated,
               count(c.{chunk_id_column})           AS n_chunks,
               coalesce(sum(length(c.{chunk_text_column})), 0) AS chars,
               min(c.{chunk_created_column})        AS first_chunk,
               max(c.{chunk_created_column})        AS last_chunk
        FROM {docs_t} d
        LEFT JOIN {chunks_t} c ON c.{document_id_column} = d.{id_column}
        {clause}
        GROUP BY d.{id_column}, d.{filename_column}, d.{status_column}, d.{page_count_column},
                 d.{created_column}, d.{updated_column}
        ORDER BY d.{created_column}
        """
    )

    def _stamp(ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
        if ts.tzinfo is None and assume_tz is not None:
            ts = ts.replace(tzinfo=assume_tz)
        return ts

    def _run(sess: Any) -> List[IngestDoc]:
        rows = sess.execute(stmt, params).fetchall()
        return [
            IngestDoc(
                doc_id=str(r.doc_id),
                filename=str(r.filename or ""),
                status=str(r.status or ""),
                page_count=int(r.page_count) if r.page_count is not None else None,
                n_chunks=int(r.n_chunks or 0),
                chars=int(r.chars or 0),
                doc_created=_stamp(r.doc_created),
                doc_updated=_stamp(r.doc_updated),
                first_chunk=_stamp(r.first_chunk),
                last_chunk=_stamp(r.last_chunk),
            )
            for r in rows
        ]

    if session is not None:
        return _run(session)
    with session_factory() as sess:
        return _run(sess)


def served_documents(records: Sequence[Any], *, evidence_field: str = "served_evidence",
                     name_field: str = "document_name") -> List[str]:
    """The distinct document names this cell's answers were actually served from."""
    names = {
        str(getattr(ev, name_field, "") or "")
        for rec in records
        for ev in (getattr(rec, evidence_field, None) or ())
    }
    return sorted(n for n in names if n)


# --------------------------------------------------------------------------
# 2. span
# --------------------------------------------------------------------------
def ingest_span(
    docs: Sequence[IngestDoc],
    *,
    basis: str = "lifecycle",
    merge_overlaps: bool = True,
    require_chunks: bool = True,
    max_doc_seconds: Optional[float] = None,
    on_overrun: str = "clamp",
    min_doc_seconds: float = 0.0,
    min_seconds: float = 0.0,
    require_all: bool = False,
    round_seconds: int = 3,
) -> Dict[str, Any]:
    """
    Ingest wall clock over a set of documents, in seconds and hours.

    Args:
        basis: which stamps bound each document; see `INGEST_BASES`.
        merge_overlaps: True unions overlapping documents, so files ingested at
            the same time are charged once, which is what a wall clock reads.
            False sums them, which is the right number when the work was
            serialised on one device and you want device-seconds.
        require_chunks: a document with no chunk row never finished chunking;
            True leaves it out of the span and counts it under ``skipped``.
        max_doc_seconds: ceiling per document, for a file that sat in a queue.
            None keeps whatever the stamps say.
        on_overrun: past the ceiling, ``clamp`` to it, ``skip`` the document,
            or ``raise``.
        min_doc_seconds: a document below this contributes nothing; ``chunks``
            basis returns exactly 0 for a one-chunk file.
        min_seconds: total below this is not credible, so the span comes back
            unmeasured instead of a rounding artefact.
        require_all: raise when any document yielded no usable stretch.
        round_seconds: decimals on the reported seconds.

    Returns:
        ``{measured, basis, hours, seconds, documents, counted, skipped, first,
        last, merged, per_document}``.
    """
    if basis not in INGEST_BASES:
        raise ValueError(f"basis must be one of {list(INGEST_BASES)}, got {basis!r}")
    if on_overrun not in ("clamp", "skip", "raise"):
        raise ValueError("on_overrun must be 'clamp', 'skip' or 'raise'")

    spans: List[Tuple[datetime, datetime]] = []
    per_doc: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for d in docs:
        bounds = d.bounds(basis)
        if require_chunks and d.n_chunks <= 0:
            skipped.append({"filename": d.filename, "reason": "no chunk rows"})
            continue
        if bounds is None:
            skipped.append({"filename": d.filename, "reason": f"{basis} stamp absent"})
            continue
        start, end = bounds
        seconds = (end - start).total_seconds()
        if seconds < 0:
            skipped.append({"filename": d.filename, "reason": f"end before start ({seconds}s)"})
            continue
        if max_doc_seconds is not None and seconds > float(max_doc_seconds):
            if on_overrun == "raise":
                raise ValueError(f"document {d.filename!r}: {basis} span {seconds}s over "
                                 f"max_doc_seconds {max_doc_seconds}")
            if on_overrun == "skip":
                skipped.append({"filename": d.filename,
                                "reason": f"{round(seconds, 1)}s over max_doc_seconds"})
                continue
            end = start + timedelta(seconds=float(max_doc_seconds))
            seconds = float(max_doc_seconds)
        if seconds < min_doc_seconds:
            skipped.append({"filename": d.filename,
                            "reason": f"{round(seconds, 3)}s under min_doc_seconds"})
            continue
        spans.append((start, end))
        per_doc.append({
            "filename": d.filename,
            "seconds": round(seconds, round_seconds),
            "n_chunks": d.n_chunks,
            "chars": d.chars,
            "pages": d.page_count,
            "start": start.isoformat(),
            "end": end.isoformat(),
        })

    if require_all and skipped:
        detail = "; ".join(f"{s['filename']}: {s['reason']}" for s in skipped)
        raise ValueError(f"ingest span incomplete under basis {basis!r} ({detail})")

    if not spans:
        return {
            "measured": False, "basis": basis, "reason": "no document carries an ingest span",
            "hours": None, "seconds": 0.0, "documents": len(docs), "counted": 0,
            "skipped": skipped, "first": None, "last": None, "merged": 0,
            "per_document": per_doc,
        }

    spans.sort(key=lambda p: p[0])
    merged = 0
    if merge_overlaps:
        union: List[List[datetime]] = [list(spans[0])]
        for start, end in spans[1:]:
            if start <= union[-1][1]:
                union[-1][1] = max(union[-1][1], end)
                merged += 1
            else:
                union.append([start, end])
        total = sum((b - a).total_seconds() for a, b in union)
    else:
        total = sum((b - a).total_seconds() for a, b in spans)

    measured = total >= min_seconds and total > 0.0
    return {
        "measured": bool(measured),
        "basis": basis,
        "reason": None if measured else f"{round(total, 3)}s under min_seconds {min_seconds}",
        "hours": round(total / 3600.0, 6) if measured else None,
        "seconds": round(total, round_seconds),
        "documents": len(docs),
        "counted": len(spans),
        "skipped": skipped,
        "first": min(a for a, _ in spans).isoformat(),
        "last": max(b for _, b in spans).isoformat(),
        "merged": merged,
        "per_document": per_doc,
    }


def ingest_tokens(
    docs: Sequence[IngestDoc],
    *,
    chars_per_token: float = CHARS_PER_TOKEN,
    tokens_total: Optional[float] = None,
    embed_passes: float = 1.0,
    require_chunks: bool = True,
    round_mtok: int = 6,
) -> Dict[str, Any]:
    """
    Tokens embedded during the ingest, in millions.

    Args:
        chars_per_token: divisor over ``sum(length(chunks.text))``. The chunk
            text is what the embedder saw, so the estimate is over the right
            characters, and it is only needed where the logged count is absent.
        tokens_total: a real tokeniser count, which overrides the estimate.
            `resolve_logged_tokens` reads that count off the app's own log.
        embed_passes: times the corpus was embedded (a re-embed after a
            chunking change is a second pass).
        require_chunks: skip documents with no chunk row, matching `ingest_span`.
        round_mtok: decimals on the reported millions.
    """
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be > 0, got {chars_per_token!r}")
    if embed_passes <= 0:
        raise ValueError(f"embed_passes must be > 0, got {embed_passes!r}")
    used = [d for d in docs if d.n_chunks > 0 or not require_chunks]
    chars = sum(int(d.chars or 0) for d in used)
    chunks = sum(int(d.n_chunks or 0) for d in used)
    estimated = tokens_total is None
    tokens = (chars / float(chars_per_token)) if estimated else float(tokens_total)
    mtok = tokens * float(embed_passes) / 1_000_000.0
    return {
        "documents": len(used),
        "chunks": chunks,
        "chars": chars,
        "chars_per_token": float(chars_per_token),
        "estimated": estimated,
        "embed_passes": float(embed_passes),
        "tokens": round(tokens * float(embed_passes), 1),
        "mtok": round(mtok, round_mtok),
    }


def ingest_cost_inputs(
    span: Dict[str, Any],
    tokens: Optional[Dict[str, Any]] = None,
    *,
    gpu_hours_key: str = INGEST_GPU_HOURS_KEY,
    embed_mtok_key: str = INGEST_EMBED_MTOK_KEY,
    include_embed_mtok: bool = True,
    hours_upper_bound: bool = False,
    round_hours: int = 6,
) -> Dict[str, Any]:
    """
    The two one-time CFCA quantities, from a span and a token count.

    Args:
        include_embed_mtok: False leaves ``onetime_embed_mtok`` alone, for a
            local embedder whose token price is 0 and whose hours are already
            inside the span.
        hours_upper_bound: True when the span is known to bill idle queue time,
            which the ``lifecycle`` basis does.

    Returns:
        ``{onetime_gpu_hours, onetime_embed_mtok, source, upper_bound}``, or the
        same shape with a None hour count when the span was unmeasured.
    """
    hours = span.get("hours") if span.get("measured") else None
    out: Dict[str, Any] = {
        gpu_hours_key: round(float(hours), round_hours) if hours is not None else None,
        "upper_bound": bool(hours_upper_bound and hours is not None),
    }
    if include_embed_mtok and tokens is not None:
        out[embed_mtok_key] = tokens.get("mtok")
    if hours is None:
        out["source"] = f"unmeasured: {span.get('reason') or 'no ingest span'}"
        return out
    bits = [f"measured ({span.get('basis')} basis, {span.get('counted')}/{span.get('documents')} docs",
            f"{span.get('seconds')}s"]
    if span.get("merged"):
        bits.append(f"{span.get('merged')} overlaps merged")
    if include_embed_mtok and tokens is not None:
        if tokens.get("source"):
            bits.append(f"{tokens.get('mtok')} Mtok {tokens['source']}")
        else:
            how = "estimated" if tokens.get("estimated") else "counted"
            bits.append(f"{tokens.get('mtok')} Mtok {how} at {tokens.get('chars_per_token')} chars/token")
    out["source"] = ", ".join(bits) + ")"
    return out


def logged_build_hours(
    tokens: Optional[Dict[str, Any]] = None,
    *,
    logged_key: str = LOGGED_INDEX_BUILD_KEY,
    gpu_hours_key: str = INGEST_GPU_HOURS_KEY,
    min_hours: float = 0.0,
    min_runs: int = 1,
    round_hours: int = 6,
) -> Tuple[Optional[float], str]:
    """
    The build hours the app itself logged, read off a token block.

    The rag-service times its own ``chunk_and_index()`` call and writes the
    milliseconds into `INDEX_BUILD_TABLE`, so this bills the rebuild that
    happened instead of the stretch between two document stamps. A re-ingest
    writes a fresh row and never touches ``documents``, which is why the
    ``lifecycle`` and ``first_touch`` bases cannot see one.

    Args:
        logged_key: where the token block parks `index_build_totals`.
        gpu_hours_key: hour key inside that block and in the returned figure.
        min_hours: hour counts at or below this are refused as not credible,
            which a 0.0 always is.
        min_runs: build rows the block must carry before its seconds are used.
        round_hours: decimals on the returned figure.

    Returns:
        ``(hours, reason)``. hours is None when the log cannot bill the build
        and the reason names which refusal applied, so a caller can print it.
    """
    block = (tokens or {}).get(logged_key) if isinstance(tokens, dict) else None
    if not isinstance(block, dict):
        return None, f"no {logged_key} block in the token count"
    runs = int(block.get("runs") or 0)
    if runs < int(min_runs):
        return None, f"{INDEX_BUILD_TABLE} has {runs} run(s) in scope, {min_runs} needed"
    raw = block.get(gpu_hours_key)
    if raw is None or not block.get("measured"):
        return None, f"{INDEX_BUILD_TABLE} logged {block.get('seconds')}s in scope"
    hours = float(raw)
    if hours <= float(min_hours):
        return None, f"{INDEX_BUILD_TABLE} logged {hours} h, at or below min_hours={min_hours}"
    policies = list(block.get("policies") or ())
    return round(hours, round_hours), (
        f"measured ({INDEX_BUILD_TABLE}, {runs} run(s), {block.get('seconds')}s"
        + (f", policy {policies[0]}" if len(policies) == 1 else "")
        + ")"
    )


def merged_ingest_cost_inputs(
    span: Dict[str, Any],
    tokens: Optional[Dict[str, Any]] = None,
    *,
    hours_source: str = "span",
    gpu_hours_key: str = INGEST_GPU_HOURS_KEY,
    embed_mtok_key: str = INGEST_EMBED_MTOK_KEY,
    include_embed_mtok: bool = True,
    hours_upper_bound: bool = False,
    logged_upper_bound: bool = False,
    on_missing_logged: str = "span",
    logged_kwargs: Optional[Dict[str, Any]] = None,
    round_hours: int = 6,
) -> Dict[str, Any]:
    """
    The same two CFCA quantities as `ingest_cost_inputs`, with the hour count
    taken from whichever clock the caller names.

    ``hours_source="span"`` returns `ingest_cost_inputs` unchanged, so the
    default is today's number. Every other value reaches for the app's logged
    build seconds through `logged_build_hours`, and ``source`` always names the
    clock that won and the one that lost.

    Args:
        hours_source: one of `HOURS_SOURCES`.
        hours_upper_bound: the span bills idle queue time, as ``lifecycle`` does.
        logged_upper_bound: the logged seconds bill more than the build; False,
            because the rag-service times the call itself.
        on_missing_logged: no logged row and ``hours_source="logged"``: ``span``
            falls back and says so, ``unmeasured`` drops the hours so the
            declared number survives, ``raise`` stops the run.
        logged_kwargs: `logged_build_hours` knobs, ``min_hours`` / ``min_runs``.

    Returns:
        ``{onetime_gpu_hours, onetime_embed_mtok, source, upper_bound}``.
    """
    if hours_source not in HOURS_SOURCES:
        raise ValueError(f"hours_source must be one of {list(HOURS_SOURCES)}, got {hours_source!r}")
    if on_missing_logged not in ("span", "unmeasured", "raise"):
        raise ValueError(
            f"on_missing_logged must be 'span', 'unmeasured' or 'raise', got {on_missing_logged!r}"
        )
    out = ingest_cost_inputs(
        span, tokens,
        gpu_hours_key=gpu_hours_key, embed_mtok_key=embed_mtok_key,
        include_embed_mtok=include_embed_mtok, hours_upper_bound=hours_upper_bound,
        round_hours=round_hours,
    )
    if hours_source == "span":
        return out
    span_hours = out.get(gpu_hours_key)
    hours, note = logged_build_hours(
        tokens, gpu_hours_key=gpu_hours_key, round_hours=round_hours,
        **(logged_kwargs or {}),
    )
    span_note = f"span {span.get('basis')} basis {span.get('seconds')}s"
    if hours_source == "prefer_span" and span_hours is not None:
        out["source"] = f"{out['source']}; {INDEX_BUILD_TABLE} not read (span preferred)"
        return out
    if hours is not None:
        out[gpu_hours_key] = hours
        out["upper_bound"] = bool(logged_upper_bound)
        out["source"] = f"{note}; {span_note} not used"
        return out
    if hours_source == "prefer_logged" and span_hours is not None:
        out["source"] = f"{out['source']}; {note}"
        return out
    if on_missing_logged == "raise":
        raise ValueError(
            f"hours_source={hours_source!r} found no logged build hours: {note}. Re-ingest with "
            f"the instrumentation on, or name a different hours_source"
        )
    if on_missing_logged == "unmeasured":
        out[gpu_hours_key] = None
        out["upper_bound"] = False
        out["source"] = f"unmeasured: {note}"
        return out
    out["source"] = f"{out['source']}; {note}"
    return out


# --------------------------------------------------------------------------
# 3. one call
# --------------------------------------------------------------------------
def measure_ingest(
    *,
    filenames: Optional[Sequence[str]] = None,
    docs: Optional[Sequence[IngestDoc]] = None,
    basis: str = "lifecycle",
    merge_overlaps: bool = True,
    include_embed_mtok: bool = True,
    chars_per_token: float = CHARS_PER_TOKEN,
    tokens_total: Optional[Any] = None,
    embed_passes: float = 1.0,
    hours_upper_bound: Optional[bool] = None,
    hours_source: str = "span",
    hours_kwargs: Optional[Dict[str, Any]] = None,
    keep_per_document: bool = False,
    read_kwargs: Optional[Dict[str, Any]] = None,
    span_kwargs: Optional[Dict[str, Any]] = None,
    token_kwargs: Optional[Dict[str, Any]] = None,
    printer: Optional[Callable[[str], None]] = None,
    cell: str = "",
    session: Any = None,
) -> Dict[str, Any]:
    """
    One cell's measured ingest: read the stamps, span them, price the quantities.

    Args:
        filenames: the scope; None takes every ingested document.
        docs: already-read stamps, to score several cells off one query.
        tokens_total: a number, or one of `LOGGED_TOKEN_SOURCES` to read the
            count the app logged instead of dividing characters.
        hours_upper_bound: None marks the ``lifecycle`` basis as an upper bound
            (it bills queue wait) and the others as exact.
        hours_source: which clock bills the build, one of `HOURS_SOURCES`. A
            re-ingest writes chunk rows and a build log row only, so ``span``
            under the ``lifecycle`` basis reads the upload and not the rebuild;
            ``logged`` needs ``tokens_total="index_build_logs"`` to have carried
            the build block in.
        hours_kwargs: `merged_ingest_cost_inputs` knobs, ``on_missing_logged``
            and ``logged_kwargs`` among them.
        keep_per_document: keep the per-file table in the result, which is worth
            having in a snapshot and noise in a report.
        read_kwargs / span_kwargs / token_kwargs: `read_ingest_documents`,
            `ingest_span` and `resolve_logged_tokens` knobs.
        printer: one line naming the hours; None stays silent.
        cell: name for that line and for the result.
        session: an open read-only session to reuse.

    Returns:
        ``{cell, scope, basis, documents, span, tokens, cost_inputs}``.
    """
    rows = list(docs) if docs is not None else read_ingest_documents(
        filenames=filenames, session=session, **(read_kwargs or {}),
    )
    span = ingest_span(rows, basis=basis, merge_overlaps=merge_overlaps, **(span_kwargs or {}))
    if isinstance(tokens_total, str):
        tokens = resolve_logged_tokens(
            tokens_total, docs=rows, filenames=filenames, session=session,
            chars_per_token=chars_per_token, embed_passes=embed_passes,
            **(token_kwargs or {}),
        )
    else:
        tokens = ingest_tokens(rows, chars_per_token=chars_per_token, tokens_total=tokens_total,
                               embed_passes=embed_passes)
    upper = (basis == "lifecycle") if hours_upper_bound is None else bool(hours_upper_bound)
    cost_inputs = merged_ingest_cost_inputs(
        span, tokens, hours_source=hours_source, include_embed_mtok=include_embed_mtok,
        hours_upper_bound=upper, **(hours_kwargs or {}),
    )
    lean = dict(span)
    if not keep_per_document:
        lean.pop("per_document", None)
    out = {
        "cell": cell,
        "scope": "corpus" if filenames is None else "documents",
        "basis": basis,
        "documents": [d.as_dict() for d in rows] if keep_per_document else len(rows),
        "span": lean,
        "tokens": tokens,
        "cost_inputs": cost_inputs,
    }
    if printer:
        hours = cost_inputs.get(INGEST_GPU_HOURS_KEY)
        printer(f"  {cell or 'ingest'}: {cost_inputs['source']}"
                + (f" -> {hours} h" if hours is not None else ""))
    return out


def measure_cell_ingest(
    cell: str,
    records: Sequence[Any],
    *,
    scope: str = "corpus",
    filenames: Optional[Sequence[str]] = None,
    as_of_from_records: bool = True,
    printer: Optional[Callable[[str], None]] = None,
    session: Any = None,
    **measure_kwargs: Any,
) -> Dict[str, Any]:
    """
    `measure_ingest` for one dumped cell, scoped by what the cell was served.

    Args:
        scope: ``corpus`` prices every ingested document, which is the build the
            condition ran on; ``served`` narrows to the documents this cell's
            answers actually cited, which under-counts a corpus the questions
            never touched; ``documents`` uses ``filenames`` verbatim.
        filenames: the list for ``documents`` scope.
        as_of_from_records: bound the build log at this cell's first answer, so
            a logged source reads the build that served it and not a later
            re-ingest. False leaves the bound to ``token_kwargs['as_of']``.
    """
    if scope not in ("corpus", "served", "documents"):
        raise ValueError("scope must be 'corpus', 'served' or 'documents'")
    if scope == "corpus":
        names: Optional[List[str]] = None
    elif scope == "served":
        names = served_documents(records)
        if not names:
            names = []
    else:
        if filenames is None:
            raise ValueError("scope 'documents' needs filenames")
        names = list(filenames)
    if as_of_from_records and isinstance(measure_kwargs.get("tokens_total"), str):
        tok = dict(measure_kwargs.get("token_kwargs") or {})
        if "as_of" not in tok:
            stamp = earliest_answered_at(records)
            if stamp is not None:
                tok["as_of"] = stamp
                measure_kwargs["token_kwargs"] = tok
    out = measure_ingest(filenames=names, printer=printer, cell=cell, session=session,
                         **measure_kwargs)
    out["scope"] = scope
    return out


# --------------------------------------------------------------------------
# 4. into the cost blocks
# --------------------------------------------------------------------------
def apply_measured_ingest(
    cost_inputs: Dict[str, Dict[str, float]],
    measured: Dict[str, Optional[Dict[str, Any]]],
    *,
    keys: Sequence[str] = (INGEST_GPU_HOURS_KEY, INGEST_EMBED_MTOK_KEY),
    prefer: str = "measured",
    accept_upper_bound: bool = True,
    require_measured: Sequence[str] = (),
    printer: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Put measured ingest quantities into each cell's cost block.

    The sibling of `apply_measured_power`, same vocabulary: ``measured``
    overwrites, ``declared`` keeps the file's number, ``missing`` fills only a
    key the cell never declared.

    Args:
        keys: which measured quantities may be written.
        accept_upper_bound: False keeps the declared number when the span was
            flagged as an upper bound.
        require_measured: cells that must carry a measurement, else raise.
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
        got = (measured or {}).get(name)
        usable = {k: got.get(k) for k in keys if got and got.get(k) is not None}
        if not usable:
            provenance[name] = "declared (no ingest measurement)"
            missing.append(name)
            continue
        if got.get("upper_bound") and not accept_upper_bound:
            provenance[name] = f"declared (ingest span was an upper bound: {got.get('source')})"
            continue
        if prefer == "declared":
            provenance[name] = ("declared; measured "
                                + ", ".join(f"{k}={v}" for k, v in sorted(usable.items()))
                                + " not applied")
            continue
        wrote: List[str] = []
        for key, value in usable.items():
            if prefer == "missing" and key in block:
                continue
            block[key] = float(value)
            wrote.append(f"{key}={value}")
        if not wrote:
            provenance[name] = "declared (every key already declared)"
            continue
        provenance[name] = str(got.get("source") or "measured")
        if printer:
            printer(f"  {name}: {', '.join(wrote)} measured, {provenance[name]}")

    hard = [n for n in require_measured if n in missing]
    if hard:
        raise ValueError(f"cells {sorted(hard)} have no measured ingest: "
                         f"dump them with a reachable database, or drop them from "
                         f"require_measured")
    return out, provenance


# --------------------------------------------------------------------------
# 5. the token counts the app already logs
# --------------------------------------------------------------------------
#: Sources of a real token count, accepted by ``tokens_total`` in place of a
#: number so the estimate is replaced rather than joined by a second knob.
#:
#: ``chunks``            sums ``chunks.metadata->>'token_count'``, written per
#:                       chunk by the app's chunker under the embedder's own
#:                       tokeniser, which is the tokeniser the embed term prices.
#: ``index_build_logs``  reads ``corpus_tokens`` off the app's per-run build log,
#:                       one row per ``chunk_and_index()`` call, which also
#:                       carries the build milliseconds and the index size.
LOGGED_TOKEN_SOURCES: Tuple[str, ...] = ("chunks", "index_build_logs")

#: Keys the app writes into ``chunks.metadata``.
TOKEN_METADATA_KEY: str = "token_count"
OVERFLOW_METADATA_KEY: str = "token_overflow"
POLICY_METADATA_KEY: str = "chunking_policy"

#: The app's own build log, created by the rag-service on startup.
INDEX_BUILD_TABLE: str = "index_build_logs"

#: Postgres pattern a metadata value must match before it is cast to an
#: integer, so one hand-edited row cannot fail the whole query.
NUMERIC_TEXT_RE: str = "^[0-9]+$"


@dataclass(frozen=True)
class ChunkTokens:
    """One document's logged chunk token counts, as the chunker left them."""

    doc_id: str
    filename: str
    n_chunks: int = 0
    n_counted: int = 0
    tokens: int = 0
    overflow_chunks: int = 0
    overflow_tokens: int = 0
    policies: Tuple[str, ...] = ()
    first_chunk: Optional[datetime] = None
    last_chunk: Optional[datetime] = None

    @property
    def coverage(self) -> float:
        """Share of this document's chunks carrying a logged count."""
        return (self.n_counted / self.n_chunks) if self.n_chunks else 0.0

    @property
    def complete(self) -> bool:
        return self.n_chunks > 0 and self.n_counted == self.n_chunks

    def as_dict(self) -> Dict[str, Any]:
        def iso(ts: Optional[datetime]) -> Optional[str]:
            return ts.isoformat() if ts is not None else None

        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "n_chunks": self.n_chunks,
            "n_counted": self.n_counted,
            "coverage": round(self.coverage, 4),
            "tokens": self.tokens,
            "overflow_chunks": self.overflow_chunks,
            "overflow_tokens": self.overflow_tokens,
            "policies": list(self.policies),
            "first_chunk": iso(self.first_chunk),
            "last_chunk": iso(self.last_chunk),
        }


def read_chunk_token_counts(
    *,
    filenames: Optional[Sequence[str]] = None,
    chunking_policy: Optional[str] = None,
    statuses: Optional[Sequence[str]] = ("completed",),
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    require_numeric: bool = True,
    documents_table: str = "documents",
    chunks_table: str = "chunks",
    id_column: str = "id",
    chunk_id_column: str = "id",
    document_id_column: str = "document_id",
    filename_column: str = "filename",
    status_column: str = "status",
    created_column: str = "created_at",
    chunk_created_column: str = "created_at",
    metadata_column: str = "metadata",
    token_key: str = TOKEN_METADATA_KEY,
    overflow_key: str = OVERFLOW_METADATA_KEY,
    policy_key: str = POLICY_METADATA_KEY,
    schema: str = "public",
    check_columns: bool = True,
    assume_tz: Optional[timezone] = timezone.utc,
    session: Any = None,
    session_factory: Callable[[], Any] = session_scope,
) -> List[ChunkTokens]:
    """
    Every document's logged chunk token counts, one row per document.

    The count is the embedder's own tokeniser count, which is the quantity the
    ``onetime_embed_mtok`` term prices. The generator's prompt tokens are a
    different number on a different tokeniser, logged per answer by the app and
    read by `cfca_metric.serving_token_totals`.

    Args:
        filenames: restrict to these files; None takes the whole corpus, the
            same scope vocabulary as `read_ingest_documents`.
        chunking_policy: keep only chunks stamped with this policy label. The
            predicate sits in the join, so a document whose rows carry another
            label still returns with a zero count instead of vanishing.
        statuses: document statuses that count as ingested.
        since / until: bound on the document's own ``created_at``, matching
            `read_ingest_documents`. The chunk stamp is deliberately not the
            bound: a re-ingest upserts the chunk row and keeps the original
            ``chunks.created_at``, so only the policy label separates two
            chunkings of one file.
        require_numeric: cast only values matching `NUMERIC_TEXT_RE`. False
            casts every value and fails loudly on a hand-edited row.
        documents_table ... policy_key: names, for a schema that differs.
        check_columns: verify the columns exist before querying.
        assume_tz: timezone for a naive stamp; None leaves it naive.
        session: an open read-only session; None opens one per call.

    Returns:
        One `ChunkTokens` per document, ordered by the document stamp.
    """
    from sqlalchemy import text as sql_text

    docs_t = _identifier(documents_table, "documents_table")
    chunks_t = _identifier(chunks_table, "chunks_table")
    for what, name in {
        "id_column": id_column,
        "chunk_id_column": chunk_id_column,
        "document_id_column": document_id_column,
        "filename_column": filename_column,
        "status_column": status_column,
        "created_column": created_column,
        "chunk_created_column": chunk_created_column,
        "metadata_column": metadata_column,
    }.items():
        _identifier(name, what)
    for what, key in {"token_key": token_key, "overflow_key": overflow_key,
                      "policy_key": policy_key}.items():
        if "'" in key:
            raise ValueError(f"{what} must not contain a quote, got {key!r}")

    if check_columns:
        require_ingest_columns(
            {
                docs_t: [id_column, filename_column, status_column, created_column],
                chunks_t: [chunk_id_column, document_id_column, chunk_created_column,
                           metadata_column],
            },
            schema=schema, session=session, session_factory=session_factory,
        )

    tok = f"c.{metadata_column}->>'{token_key}'"
    ovf = f"c.{metadata_column}->>'{overflow_key}'"
    pol = f"c.{metadata_column}->>'{policy_key}'"
    guard = f" ~ '{NUMERIC_TEXT_RE}'" if require_numeric else " IS NOT NULL"

    join = [f"c.{document_id_column} = d.{id_column}"]
    params: Dict[str, Any] = {}
    if chunking_policy is not None:
        join.append(f"{pol} = :policy")
        params["policy"] = chunking_policy

    where: List[str] = []
    if statuses is not None:
        where.append(f"d.{status_column} = ANY({bind_cast('statuses', sql_type='text', array=True)})")
        params["statuses"] = list(statuses)
    if filenames is not None:
        where.append(f"d.{filename_column} = ANY({bind_cast('filenames', sql_type='text', array=True)})")
        params["filenames"] = list(filenames)
    if since is not None:
        where.append(f"d.{created_column} >= :since")
        params["since"] = since
    if until is not None:
        where.append(f"d.{created_column} <= :until")
        params["until"] = until
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    stmt = sql_text(
        f"""
        SELECT CAST(d.{id_column} AS text) AS doc_id,
               d.{filename_column} AS filename,
               count(c.{chunk_id_column}) AS n_chunks,
               count(CASE WHEN {tok}{guard} THEN 1 END) AS n_counted,
               coalesce(sum(CASE WHEN {tok}{guard} THEN ({tok})::bigint END), 0) AS tokens,
               count(CASE WHEN {ovf}{guard}
                          THEN CASE WHEN ({ovf})::bigint > 0 THEN 1 END END) AS overflow_chunks,
               coalesce(sum(CASE WHEN {ovf}{guard} THEN ({ovf})::bigint END), 0) AS overflow_tokens,
               min(c.{chunk_created_column}) AS first_chunk,
               max(c.{chunk_created_column}) AS last_chunk,
               array_remove(array_agg(DISTINCT {pol}), NULL) AS policies
        FROM {docs_t} d
        LEFT JOIN {chunks_t} c ON {' AND '.join(join)}
        {clause}
        GROUP BY d.{id_column}, d.{filename_column}, d.{created_column}
        ORDER BY d.{created_column}
        """
    )

    def _stamp(ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
        if ts.tzinfo is None and assume_tz is not None:
            ts = ts.replace(tzinfo=assume_tz)
        return ts

    def _run(sess: Any) -> List[ChunkTokens]:
        rows = sess.execute(stmt, params).fetchall()
        return [
            ChunkTokens(
                doc_id=str(r.doc_id),
                filename=str(r.filename or ""),
                n_chunks=int(r.n_chunks or 0),
                n_counted=int(r.n_counted or 0),
                tokens=int(r.tokens or 0),
                overflow_chunks=int(r.overflow_chunks or 0),
                overflow_tokens=int(r.overflow_tokens or 0),
                policies=tuple(sorted(str(p) for p in (r.policies or []) if p)),
                first_chunk=_stamp(r.first_chunk),
                last_chunk=_stamp(r.last_chunk),
            )
            for r in rows
        ]

    if session is not None:
        return _run(session)
    with session_factory() as sess:
        return _run(sess)


def ingest_tokens_logged(
    counts: Sequence[ChunkTokens],
    *,
    docs: Optional[Sequence[IngestDoc]] = None,
    embed_passes: float = 1.0,
    require_chunks: bool = True,
    min_coverage: float = 1.0,
    on_incomplete: str = "fallback",
    chars_per_token: float = CHARS_PER_TOKEN,
    round_mtok: int = 6,
    round_coverage: int = 4,
) -> Dict[str, Any]:
    """
    Tokens embedded during the ingest, in millions, off the logged counts.

    Returns the shape `ingest_tokens` returns, so `ingest_cost_inputs` consumes
    either one, with the coverage the caller needs to defend the number added
    beside it.

    Args:
        counts: `read_chunk_token_counts` output.
        docs: `read_ingest_documents` output, needed for the character estimate
            that ``on_incomplete='fallback'`` falls back to.
        embed_passes: times the corpus was embedded; a re-embed after a chunking
            change is a second pass.
        require_chunks: skip documents with no chunk row, matching `ingest_span`.
        min_coverage: share of chunks that must carry a logged count, over the
            documents kept. 1.0 demands every chunk.
        on_incomplete: under ``min_coverage``, ``fallback`` prices the character
            estimate instead, ``partial`` prices the counted chunks and says so,
            ``raise`` stops.
        chars_per_token: divisor for the fallback only.
        round_mtok / round_coverage: decimals on the reported figures.
    """
    if on_incomplete not in ("fallback", "partial", "raise"):
        raise ValueError("on_incomplete must be 'fallback', 'partial' or 'raise'")
    if embed_passes <= 0:
        raise ValueError(f"embed_passes must be > 0, got {embed_passes!r}")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage!r}")

    used = [c for c in counts if c.n_chunks > 0 or not require_chunks]
    chunks = sum(c.n_chunks for c in used)
    counted = sum(c.n_counted for c in used)
    tokens = sum(c.tokens for c in used)
    coverage = (counted / chunks) if chunks else 0.0
    policies = sorted({p for c in used for p in c.policies})
    by_name = {d.filename: d for d in (docs or ())}
    chars = sum(int(by_name[c.filename].chars or 0) for c in used if c.filename in by_name)
    short = chunks == 0 or coverage < min_coverage
    reason = (
        f"{counted}/{chunks} chunks carry {TOKEN_METADATA_KEY}"
        f" (coverage {round(coverage, round_coverage)} under {min_coverage})"
    )

    if short and on_incomplete == "raise":
        raise ValueError(
            f"logged token counts incomplete: {reason}; backfill the metadata, "
            f"lower min_coverage, or pass on_incomplete='fallback'"
        )
    estimated = bool(short and on_incomplete == "fallback")
    if estimated:
        if not docs:
            raise ValueError(
                "on_incomplete='fallback' needs docs=read_ingest_documents(...) "
                "for the character estimate"
            )
        if chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {chars_per_token!r}")
        base = chars / float(chars_per_token)
        source = f"estimated at {chars_per_token} chars/token: {reason}"
    else:
        base = float(tokens)
        source = (
            f"counted from {TOKEN_METADATA_KEY} on {counted}/{chunks} chunks"
            + (f", policy {policies[0]}" if len(policies) == 1 else "")
            + (f", partial: {reason}" if short else "")
        )

    total = base * float(embed_passes)
    return {
        "documents": len(used),
        "chunks": chunks,
        "chars": chars,
        "chars_per_token": float(chars_per_token),
        "estimated": estimated,
        "embed_passes": float(embed_passes),
        "tokens": round(total, 1),
        "mtok": round(total / 1_000_000.0, round_mtok),
        "source_name": "chunks",
        "counted_chunks": counted,
        "coverage": round(coverage, round_coverage),
        "complete": bool(chunks and counted == chunks),
        "partial": bool(short and not estimated),
        "overflow_chunks": sum(c.overflow_chunks for c in used),
        "overflow_tokens": sum(c.overflow_tokens for c in used),
        "policies": policies,
        "source": source,
    }


@dataclass(frozen=True)
class IndexBuildRun:
    """One ``chunk_and_index()`` run, as the app logged it."""

    document_id: str
    chunking_policy: str = ""
    chunk_count: int = 0
    corpus_tokens: int = 0
    truncated_chunks: int = 0
    chunk_build_ms: float = 0.0
    embed_ms: float = 0.0
    index_gb: float = 0.0
    embed_model: Optional[str] = None
    created_at: Optional[datetime] = None
    run_id: Optional[int] = None

    @property
    def seconds(self) -> float:
        """Non-idle build time this run measured: chunk build plus embed."""
        return (float(self.chunk_build_ms) + float(self.embed_ms)) / 1000.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "document_id": self.document_id,
            "chunking_policy": self.chunking_policy,
            "chunk_count": self.chunk_count,
            "corpus_tokens": self.corpus_tokens,
            "truncated_chunks": self.truncated_chunks,
            "chunk_build_ms": self.chunk_build_ms,
            "embed_ms": self.embed_ms,
            "index_gb": self.index_gb,
            "embed_model": self.embed_model,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def read_index_build_logs(
    *,
    document_ids: Optional[Sequence[str]] = None,
    chunking_policy: Optional[str] = None,
    since: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
    table: str = INDEX_BUILD_TABLE,
    id_column: str = "id",
    document_id_column: str = "document_id",
    policy_column: str = "chunking_policy",
    created_column: str = "created_at",
    chunk_count_column: str = "chunk_count",
    corpus_tokens_column: str = "corpus_tokens",
    truncated_chunks_column: str = "truncated_chunks",
    chunk_build_ms_column: str = "chunk_build_ms",
    embed_ms_column: str = "embed_ms",
    index_gb_column: str = "index_gb",
    embed_model_column: str = "embed_model",
    on_missing: str = "empty",
    limit: int = 5000,
    schema: str = "public",
    assume_tz: Optional[timezone] = timezone.utc,
    session: Any = None,
    session_factory: Callable[[], Any] = session_scope,
) -> List[IndexBuildRun]:
    """
    The app's own build-log rows, newest last.

    Args:
        document_ids: restrict to these documents; None takes every row.
        chunking_policy: keep one policy label, which is what separates the
            baseline build from the re-chunked one.
        since: lower bound on the row stamp.
        as_of: upper bound on the row stamp. A cell is served by the build that
            existed when it was asked, so this takes the cell's own window
            start; a build logged after the answers did not serve them.
        table ... embed_model_column: names, for a schema that differs. Every
            one is checked before the query, so a table missing a column falls
            under ``on_missing`` instead of raising raw SQL.
        on_missing: ``empty`` returns no rows when the table is absent, which is
            an app that never ran the instrumentation; ``raise`` stops.
        limit: ceiling on rows returned.
        session: an open read-only session; None opens one per call.
    """
    from sqlalchemy import text as sql_text

    if on_missing not in ("empty", "raise"):
        raise ValueError("on_missing must be 'empty' or 'raise'")
    table_t = _identifier(table, "table")
    columns = {
        "id_column": id_column,
        "document_id_column": document_id_column,
        "policy_column": policy_column,
        "created_column": created_column,
        "chunk_count_column": chunk_count_column,
        "corpus_tokens_column": corpus_tokens_column,
        "truncated_chunks_column": truncated_chunks_column,
        "chunk_build_ms_column": chunk_build_ms_column,
        "embed_ms_column": embed_ms_column,
        "index_gb_column": index_gb_column,
        "embed_model_column": embed_model_column,
    }
    for what, name in columns.items():
        _identifier(name, what)

    try:
        require_ingest_columns(
            {table_t: sorted(set(columns.values()))},
            schema=schema, session=session, session_factory=session_factory,
        )
    except Exception:
        if on_missing == "raise":
            raise
        return []

    where: List[str] = []
    params: Dict[str, Any] = {}
    if document_ids is not None:
        where.append(
            f"CAST(b.{document_id_column} AS text) = "
            f"ANY({bind_cast('document_ids', sql_type='text', array=True)})"
        )
        params["document_ids"] = [str(d) for d in document_ids]
    if chunking_policy is not None:
        where.append(f"b.{policy_column} = :policy")
        params["policy"] = chunking_policy
    if since is not None:
        where.append(f"b.{created_column} >= :since")
        params["since"] = since
    if as_of is not None:
        where.append(f"b.{created_column} <= :as_of")
        params["as_of"] = as_of
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    stmt = sql_text(
        f"""
        SELECT b.{id_column} AS run_id,
               CAST(b.{document_id_column} AS text) AS document_id,
               b.{policy_column} AS chunking_policy,
               b.{chunk_count_column} AS chunk_count,
               b.{corpus_tokens_column} AS corpus_tokens,
               b.{truncated_chunks_column} AS truncated_chunks,
               b.{chunk_build_ms_column} AS chunk_build_ms,
               b.{embed_ms_column} AS embed_ms,
               b.{index_gb_column} AS index_gb,
               b.{embed_model_column} AS embed_model,
               b.{created_column} AS created_at
        FROM {table_t} b
        {clause}
        ORDER BY b.{created_column} ASC
        LIMIT {int(limit)}
        """
    )

    def _stamp(ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
        if ts.tzinfo is None and assume_tz is not None:
            ts = ts.replace(tzinfo=assume_tz)
        return ts

    def _run(sess: Any) -> List[IndexBuildRun]:
        rows = sess.execute(stmt, params).fetchall()
        return [
            IndexBuildRun(
                run_id=int(r.run_id) if r.run_id is not None else None,
                document_id=str(r.document_id),
                chunking_policy=str(r.chunking_policy or ""),
                chunk_count=int(r.chunk_count or 0),
                corpus_tokens=int(r.corpus_tokens or 0),
                truncated_chunks=int(r.truncated_chunks or 0),
                chunk_build_ms=float(r.chunk_build_ms or 0.0),
                embed_ms=float(r.embed_ms or 0.0),
                index_gb=float(r.index_gb or 0.0),
                embed_model=str(r.embed_model) if r.embed_model else None,
                created_at=_stamp(r.created_at),
            )
            for r in rows
        ]

    if session is not None:
        return _run(session)
    with session_factory() as sess:
        return _run(sess)


def index_build_totals(
    runs: Sequence[IndexBuildRun],
    *,
    per_document: str = "latest",
    embed_passes: float = 1.0,
    include_chunk_build_ms: bool = True,
    include_embed_ms: bool = True,
    index_gb_mode: str = "sum",
    gpu_hours_key: str = INGEST_GPU_HOURS_KEY,
    embed_mtok_key: str = INGEST_EMBED_MTOK_KEY,
    index_gb_key: str = "index_gb",
    round_hours: int = 6,
    round_mtok: int = 6,
    round_gb: int = 6,
) -> Dict[str, Any]:
    """
    One build's measured quantities, ready for a cost block.

    Args:
        per_document: ``latest`` keeps the newest run per document, which is the
            build that served the cell; ``all`` sums every run, which prices a
            re-ingest as extra work; ``first`` keeps the original build.
        embed_passes: multiplier on the token total, for a corpus embedded more
            than once outside these rows.
        include_chunk_build_ms / include_embed_ms: which measured milliseconds
            enter the hours. Dropping the embed leaves the chunking alone.
        index_gb_mode: ``sum`` adds the per-document estimates, ``latest`` takes
            the newest row's own figure, ``none`` reports no index size.
        gpu_hours_key / embed_mtok_key / index_gb_key: names written at the top
            level, so `apply_measured_ingest` can pick them straight out.
        round_hours / round_mtok / round_gb: decimals on the reported figures.
    """
    if per_document not in ("latest", "all", "first"):
        raise ValueError("per_document must be 'latest', 'all' or 'first'")
    if index_gb_mode not in ("sum", "latest", "none"):
        raise ValueError("index_gb_mode must be 'sum', 'latest' or 'none'")
    if embed_passes <= 0:
        raise ValueError(f"embed_passes must be > 0, got {embed_passes!r}")

    def _order(run: IndexBuildRun) -> Tuple[int, float, int]:
        """Stamp order that never compares a naive stamp with an aware one."""
        ts = run.created_at
        if ts is None:
            return (1, 0.0, int(run.run_id or 0))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (0, ts.timestamp(), int(run.run_id or 0))

    ordered = sorted(runs, key=_order)
    if per_document == "all":
        kept = list(ordered)
    else:
        picked: Dict[str, IndexBuildRun] = {}
        for run in ordered:
            if per_document == "latest" or run.document_id not in picked:
                picked[run.document_id] = run
        kept = list(picked.values())

    seconds = sum(
        (float(r.chunk_build_ms) if include_chunk_build_ms else 0.0)
        + (float(r.embed_ms) if include_embed_ms else 0.0)
        for r in kept
    ) / 1000.0
    tokens = sum(r.corpus_tokens for r in kept) * float(embed_passes)
    if index_gb_mode == "sum":
        gb: Optional[float] = sum(r.index_gb for r in kept)
    elif index_gb_mode == "latest":
        gb = kept[-1].index_gb if kept else None
    else:
        gb = None

    policies = sorted({r.chunking_policy for r in kept if r.chunking_policy})
    stamps = [r.created_at for r in kept if r.created_at is not None]
    measured = bool(kept) and seconds > 0.0
    out: Dict[str, Any] = {
        "source_name": "index_build_logs",
        "measured": measured,
        "runs": len(kept),
        "runs_seen": len(ordered),
        "per_document": per_document,
        "documents": len({r.document_id for r in kept}),
        "policies": policies,
        "embed_models": sorted({r.embed_model for r in kept if r.embed_model}),
        "chunks": sum(r.chunk_count for r in kept),
        "truncated_chunks": sum(r.truncated_chunks for r in kept),
        "corpus_tokens": sum(r.corpus_tokens for r in kept),
        "seconds": round(seconds, 3),
        "first": min(stamps).isoformat() if stamps else None,
        "last": max(stamps).isoformat() if stamps else None,
        "upper_bound": False,
        gpu_hours_key: round(seconds / 3600.0, round_hours) if measured else None,
        embed_mtok_key: round(tokens / 1_000_000.0, round_mtok),
    }
    if gb is not None:
        out[index_gb_key] = round(gb, round_gb)
    out["source"] = (
        f"{'measured' if measured else 'unmeasured'} "
        f"({INDEX_BUILD_TABLE}, {len(kept)}/{len(ordered)} runs, "
        f"{per_document} per document, {round(seconds, 3)}s, "
        f"{out[embed_mtok_key]} Mtok counted"
        + (f", policy {policies[0]}" if len(policies) == 1 else "")
        + ")"
    )
    return out


def resolve_logged_tokens(
    source: str,
    *,
    docs: Optional[Sequence[IngestDoc]] = None,
    filenames: Optional[Sequence[str]] = None,
    chunking_policy: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
    embed_passes: float = 1.0,
    chars_per_token: float = CHARS_PER_TOKEN,
    min_coverage: float = 1.0,
    on_incomplete: str = "fallback",
    per_document: str = "latest",
    on_missing: str = "empty",
    chunk_kwargs: Optional[Dict[str, Any]] = None,
    log_kwargs: Optional[Dict[str, Any]] = None,
    totals_kwargs: Optional[Dict[str, Any]] = None,
    session: Any = None,
) -> Dict[str, Any]:
    """
    The `ingest_tokens` shape from a logged count instead of a divisor.

    Args:
        source: one of `LOGGED_TOKEN_SOURCES`.
        docs: the already-read stamps, for the fallback estimate and for
            scoping the build log to this cell's documents.
        filenames / chunking_policy / since / until: the chunk scope.
        as_of: build-log upper bound, the cell's window start.
        chunk_kwargs / log_kwargs / totals_kwargs: the three called functions'
            own knobs, for a schema or a rounding this signature does not name.
        session: an open read-only session; None opens one per call.
    """
    name = (source or "").strip().lower()
    if name not in LOGGED_TOKEN_SOURCES:
        raise ValueError(
            f"tokens_total source must be one of {list(LOGGED_TOKEN_SOURCES)} "
            f"or a number, got {source!r}"
        )
    if name == "chunks":
        counts = read_chunk_token_counts(
            filenames=filenames, chunking_policy=chunking_policy,
            since=since, until=until, session=session, **(chunk_kwargs or {}),
        )
        out = ingest_tokens_logged(
            counts, docs=docs, embed_passes=embed_passes, min_coverage=min_coverage,
            on_incomplete=on_incomplete, chars_per_token=chars_per_token,
        )
        out["per_document_counts"] = [c.as_dict() for c in counts]
        return out

    runs = read_index_build_logs(
        document_ids=[d.doc_id for d in docs] if docs else None,
        chunking_policy=chunking_policy, since=since, as_of=as_of,
        on_missing=on_missing, session=session, **(log_kwargs or {}),
    )
    totals = index_build_totals(
        runs, per_document=per_document, embed_passes=embed_passes,
        **(totals_kwargs or {}),
    )
    chars = sum(int(d.chars or 0) for d in (docs or ()))
    tokens = float(totals.get("corpus_tokens") or 0.0) * float(embed_passes)
    if not runs:
        if on_incomplete == "raise":
            raise ValueError(
                f"{INDEX_BUILD_TABLE} has no row in scope: run the ingest with the "
                f"instrumentation on, or pass a number for tokens_total"
            )
        if on_incomplete == "fallback":
            if not docs:
                raise ValueError(
                    "on_incomplete='fallback' needs docs=read_ingest_documents(...) "
                    "for the character estimate"
                )
            tokens = (chars / float(chars_per_token)) * float(embed_passes)
    estimated = not runs and on_incomplete == "fallback"
    return {
        "documents": totals.get("documents") if runs else len(docs or ()),
        "chunks": totals.get("chunks", 0),
        "chars": chars,
        "chars_per_token": float(chars_per_token),
        "estimated": estimated,
        "embed_passes": float(embed_passes),
        "tokens": round(tokens, 1),
        "mtok": round(tokens / 1_000_000.0, 6),
        "source_name": "index_build_logs",
        "policies": totals.get("policies", []),
        "truncated_chunks": totals.get("truncated_chunks", 0),
        "source": (
            f"estimated at {chars_per_token} chars/token: {INDEX_BUILD_TABLE} empty in scope"
            if estimated else totals.get("source")
        ),
        LOGGED_INDEX_BUILD_KEY: totals,
    }


# --------------------------------------------------------------------------
# 6. the label check
# --------------------------------------------------------------------------
def latest_build_policy(
    *,
    since: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
    session: Any = None,
    read_kwargs: Optional[Dict[str, Any]] = None,
    on_error: str = "note",
) -> Tuple[Optional[str], str]:
    """
    The label on the newest build row, for a check before the numbers are frozen.

    `check_ingest_policies` reads a dump, which is after the fact. This reads the
    live table instead, so ``veridic-eval sit`` can say that the answers just
    given were served by a corpus the other builder chunked, while re-asking
    them is still cheap.

    Args:
        since / as_of: bound the rows, for a window that is already known.
        session: an open read-only session to reuse.
        read_kwargs: `read_index_build_logs` knobs.
        on_error: ``note`` returns the exception as the reason, ``raise`` lets
            it out, for a caller with no database in reach.

    Returns:
        ``(label, reason)``; label is None when no row carries one.
    """
    if on_error not in ("note", "raise"):
        raise ValueError(f"on_error must be 'note' or 'raise', got {on_error!r}")
    try:
        runs = read_index_build_logs(
            since=since, as_of=as_of, session=session, **(read_kwargs or {}),
        )
    except Exception as exc:
        if on_error == "raise":
            raise
        return None, f"{INDEX_BUILD_TABLE} unreadable ({type(exc).__name__}: {exc})"
    if not runs:
        return None, f"{INDEX_BUILD_TABLE} has no row in scope"
    labelled = [r for r in runs if r.chunking_policy]
    if not labelled:
        return None, f"{INDEX_BUILD_TABLE} newest row carries no label"
    newest = labelled[-1]
    return str(newest.chunking_policy), (
        f"newest build {newest.created_at.isoformat() if newest.created_at else '(no stamp)'}"
        f", {len(labelled)} labelled row(s) in scope"
    )


def ingest_policy_block(
    measured: Optional[Dict[str, Any]] = None,
    *,
    logged_key: str = LOGGED_INDEX_BUILD_KEY,
    gpu_hours_key: str = INGEST_GPU_HOURS_KEY,
    keep_runs: bool = True,
) -> Dict[str, Any]:
    """
    The label side of one `measure_ingest` result, small enough to dump.

    The cost block carries hours and Mtok and nothing about which builder ran,
    so a cell dumped without this block cannot be checked afterwards. The
    policy label is the bucket key every cost query groups by, which is what
    makes it the thing to check.

    Args:
        logged_key: where the token block parks `index_build_totals`.
        keep_runs: keep the build row count and seconds, for a report line.

    Returns:
        ``{policies, chunks, truncated_chunks, corpus_tokens, tokens_source,
        basis, hours_source}``, with unknown fields left as None or ().
    """
    meta = measured or {}
    tokens = meta.get("tokens") if isinstance(meta.get("tokens"), dict) else {}
    block = tokens.get(logged_key) if isinstance(tokens.get(logged_key), dict) else {}
    cost = meta.get("cost_inputs") if isinstance(meta.get("cost_inputs"), dict) else {}
    policies = list(block.get("policies") or tokens.get("policies") or ())
    out: Dict[str, Any] = {
        "policies": policies,
        "chunks": block.get("chunks", tokens.get("chunks")),
        "truncated_chunks": block.get("truncated_chunks", tokens.get("truncated_chunks")),
        "corpus_tokens": block.get("corpus_tokens", tokens.get("tokens")),
        "tokens_source": tokens.get("source_name") or tokens.get("source"),
        "basis": meta.get("basis"),
        "hours": cost.get(gpu_hours_key),
        "hours_source": cost.get("source"),
    }
    if keep_runs:
        out["runs"] = block.get("runs")
        out["seconds"] = block.get("seconds")
        out["embed_models"] = list(block.get("embed_models") or ())
    return out


def policy_unit(label: str) -> str:
    """
    The builder letter inside a policy label: ``w`` words, ``t`` tokens.

    `resolve_chunking_policy` writes ``context_rag_w1000_o200`` for the word
    builder and ``context_rag_t256_o64`` for the token one, so this letter is
    the one character that says which builder produced a bucket.
    """
    for part in str(label or "").split("_"):
        if len(part) > 1 and part[0] in ("w", "t") and part[1:].isdigit():
            return part[0]
    return ""


def policy_family(label: str) -> str:
    """The label with its sizes dropped: ``context_rag_w1000_o200`` -> ``context_rag``."""
    parts = []
    for part in str(label or "").split("_"):
        if len(part) > 1 and part[0] in ("w", "t", "o") and part[1:].isdigit():
            break
        parts.append(part)
    return "_".join(parts)


def policies_agree(found: str, expected: str, *, match: str = "exact") -> Tuple[bool, str]:
    """
    Whether one found label satisfies one expected label under one rule.

    Args:
        match: one of `POLICY_MATCHES`. ``unit`` accepts any size as long as the
            builder letter agrees, which is the check that catches a forgotten
            ``CHUNKING_MODE`` flip; ``prefix`` accepts any size and unit inside
            the same family; ``exact`` accepts the string only.

    Returns:
        ``(ok, reason)``, the reason naming what was compared.
    """
    if match not in POLICY_MATCHES:
        raise ValueError(f"match must be one of {list(POLICY_MATCHES)}, got {match!r}")
    found_s, expected_s = str(found or ""), str(expected or "")
    if match == "exact":
        return found_s == expected_s, f"{found_s or '(none)'} vs {expected_s or '(none)'}"
    if match == "prefix":
        fam_f, fam_e = policy_family(found_s), policy_family(expected_s)
        return bool(fam_f) and fam_f == fam_e, f"family {fam_f or '(none)'} vs {fam_e or '(none)'}"
    unit_f, unit_e = policy_unit(found_s), policy_unit(expected_s)
    if not unit_f or not unit_e:
        return found_s == expected_s, (
            f"no unit letter in {found_s or '(none)'} or {expected_s or '(none)'}, compared whole"
        )
    return unit_f == unit_e, f"unit {unit_f} vs {unit_e} ({found_s} vs {expected_s})"


def check_ingest_policies(
    measured: Dict[str, Optional[Dict[str, Any]]],
    expected: Optional[Dict[str, str]] = None,
    *,
    match: str = "exact",
    policies_key: str = "policies",
    truncated_key: str = "truncated_chunks",
    on_mismatch: str = "warn",
    on_mixed: str = "warn",
    on_unlabelled: str = "warn",
    on_unmeasured: str = "warn",
    on_truncated: str = "ignore",
    require: Sequence[str] = (),
    skip: Sequence[str] = (),
    printer: Optional[Callable[[str], None]] = print,
    label: str = "ingest policy",
) -> Tuple[bool, Dict[str, str], List[str]]:
    """
    Read the label each cell's ingest actually ran under and say so out loud.

    Nothing here edits a file or a number. A cell whose ``CHUNKING_MODE`` was
    never flipped ingests under the other cell's builder and lands in the other
    cell's bucket with no warning at all, and this is the call that catches it,
    off the blocks `ingest_policy_block` dumped beside each cell.

    Args:
        measured: ``{cell: policy block or None}``, as read back from the dumps.
        expected: ``{cell: label}`` the run declared; a cell absent from it is
            reported and not judged.
        match: how strictly a found label must equal the expected one, one of
            `POLICY_MATCHES`.
        on_mismatch / on_mixed / on_unlabelled / on_unmeasured: ``raise``,
            ``warn`` or ``ignore``, per fault. ``mixed`` is more than one label
            in one cell, which means two builds were summed; ``unlabelled`` is
            an empty label, which buckets on its own key.
        on_truncated: a cell whose logged build truncated chunks; the token
            builder exists to hold this at zero, so ``warn`` is worth it once
            the token cells are the ones being read.
        require: cells that must pass whatever the four settings say.
        skip: cells to leave unjudged, off-grid ones for instance.
        printer: one line per verdict; None stays silent.
        label: what the lines call this check.

    Returns:
        ``(ok, verdicts, faults)``: ok is False when any judged cell failed,
        verdicts maps every cell to its one-line reading, and faults lists the
        cells that failed.
    """
    for name, value in (("on_mismatch", on_mismatch), ("on_mixed", on_mixed),
                        ("on_unlabelled", on_unlabelled), ("on_unmeasured", on_unmeasured),
                        ("on_truncated", on_truncated)):
        if value not in ("raise", "warn", "ignore"):
            raise ValueError(f"{name} must be 'raise', 'warn' or 'ignore', got {value!r}")
    if match not in POLICY_MATCHES:
        raise ValueError(f"match must be one of {list(POLICY_MATCHES)}, got {match!r}")

    want = dict(expected or {})
    skipped = set(skip or ())
    hard = set(require or ())
    verdicts: Dict[str, str] = {}
    faults: List[str] = []
    raises: List[str] = []

    def fault(cell: str, policy: str, note: str) -> None:
        verdicts[cell] = f"{note}"
        if policy == "ignore" and cell not in hard:
            return
        faults.append(cell)
        if policy == "raise" or cell in hard:
            raises.append(f"{cell}: {note}")

    for cell in sorted(measured or {}):
        block = measured.get(cell)
        if cell in skipped:
            verdicts[cell] = "skipped"
            continue
        if not isinstance(block, dict):
            fault(cell, on_unmeasured, "no ingest policy block dumped for this cell")
            continue
        found = list(block.get(policies_key) or ())
        if not found:
            fault(cell, on_unlabelled, f"no label in {INDEX_BUILD_TABLE} for this cell's scope")
            continue
        if len(found) > 1:
            fault(cell, on_mixed, f"{len(found)} labels summed into one cell: {found}")
            continue
        got = str(found[0])
        if not got.strip():
            fault(cell, on_unlabelled, "empty label: the build wrote no chunking_policy")
            continue
        if cell not in want:
            verdicts[cell] = f"ran {got}, nothing expected"
            continue
        ok, why = policies_agree(got, want[cell], match=match)
        if not ok:
            fault(cell, on_mismatch, f"ran under {got}, expected {want[cell]} ({match}: {why})")
            continue
        truncated = block.get(truncated_key)
        if truncated is None:
            tail = ""
        elif truncated:
            tail = f", {truncated} chunk(s) truncated"
        else:
            tail = ", nothing truncated"
        if truncated and on_truncated != "ignore":
            fault(cell, on_truncated, f"ran {got} as expected{tail}")
            continue
        verdicts[cell] = f"ran {got} as expected ({match}: {why}){tail}"

    if printer:
        for cell in sorted(verdicts):
            mark = "FAIL" if cell in faults else "ok"
            printer(f"  {label} {mark} {cell}: {verdicts[cell]}")
    if raises:
        raise ValueError(f"{label}: " + "; ".join(raises))
    return (not faults), verdicts, faults


def earliest_answered_at(
    records: Sequence[Any],
    *,
    field: str = "answered_at",
    assume_tz: Optional[timezone] = timezone.utc,
) -> Optional[datetime]:
    """
    The first answer stamp in a cell, which dates the build that served it.

    The records are the turns the cell's declared window selected, so this is
    that window's own start as the database recorded it, with no second bound
    to keep in step with `veridic-eval sit`.
    """
    stamps: List[datetime] = []
    for rec in records or ():
        raw = getattr(rec, field, None)
        if raw is None:
            continue
        ts = raw
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        if not isinstance(ts, datetime):
            continue
        if ts.tzinfo is None and assume_tz is not None:
            ts = ts.replace(tzinfo=assume_tz)
        stamps.append(ts)
    return min(stamps) if stamps else None
