"""
Extraction: join benchmark queries to persisted logs, per condition.

This is the ONLY module that touches the app database, and it is strictly
read-only (see db.py). It never replays the index or calls the LLM - it reads
exactly what the app already served (logs-only, k <= top_n).

For each (condition, benchmark query) it produces a ``QueryRecord`` carrying:
  * the served evidence set in served order (rerank desc, retrieval desc) -
    matching chat-service routes/messages.get_messages
  * full chunk text for each served chunk (for faithfulness context rebuild)
  * the assistant answer text
  * the completion log (provider/model/tokens/latency) for cost
  * the rag log (top_k/top_n/latencies)

Linking a benchmark query to a logged answer:
  1. explicit: benchmark `message_ids[condition]` if provided, else
  2. text match: the user message whose normalised content == the query text,
     within the condition window / conversation set, then the next assistant
     message in that conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from .benchmark import BenchmarkQuery, MATCH_TRIM_CHARS, match_key
from .config import DEFAULT_RUN_ID, Condition
from .db import ID_SQL_TYPE, bind_cast, session_scope


@dataclass
class ServedEvidence:
    chunk_id: str
    document_name: str
    page: int
    retrieval_score: Optional[float]
    rerank_score: Optional[float]
    served_rank: int           # 0-based, in served order
    chunk_text: str = ""       # full chunks.text (for faithfulness)


def render_served_context(
    evidence: Sequence[ServedEvidence],
    *,
    passages: str = "joined",
    order_by: Optional[str] = "served_rank",
    document_header: Optional[str] = "=== Document: {document} ===",
    page_line: Optional[str] = "Page {page}:",
    group_by_document: bool = True,
    passage_sep: str = "\n\n",
    document_sep: str = "\n\n\n",
    trailing: str = "\n",
    line_sep: str = "\n",
    unknown_document: str = "Unknown",
    unknown_page: Any = 0,
    skip_empty_text: bool = True,
) -> List[str]:
    """The served chunks framed the way the app framed them for the model.

    The app wraps every retrieved chunk before the LLM reads it
    (rag-service context_assembler._build_context_string)::

        === Document: FRA2.pdf ===
        Page 21:
        <chunk text>

    and the chat system prompt orders the model to cite exactly those names and
    pages (chat-service chat_handler._get_system_prompt). A judge handed the bare chunk text
    scores every correct citation as unsupported, so the defaults here rebuild
    the served frame. Pass ``document_header=None, page_line=None`` for the bare
    text a pre-2026-09-02 report was scored on.

    Args:
      passages: ``"joined"`` returns ONE passage holding the whole context, the
        single string the app sent. ``"per_chunk"`` returns one passage per
        chunk, each carrying its own header and page line so provenance
        survives when the detector budgets passages separately.
      order_by: attribute to sort by; None keeps the given order.
      document_header / page_line: format strings over ``{document}`` and
        ``{page}``. None drops that line.
      group_by_document: emit the header only when the document changes, as the
        app does. False repeats it per chunk. Ignored under ``"per_chunk"``,
        where a standalone passage always carries its own header.
      passage_sep: between chunk blocks of the same document.
      document_sep: between chunk blocks when the document changes (the app's
        extra blank line).
      trailing: appended to the joined string (the app's trailing blank line).
      line_sep: between header, page line and text inside one block.
      unknown_document / unknown_page: substituted when the row carries none.
      skip_empty_text: drop evidence rows whose chunk text is empty, which is
        what a dump written with ``include_chunk_text=False`` holds.

    Returns the passage list. ``"joined"`` always returns one element, empty
    string included, so a caller never has to special-case no evidence.
    """
    if passages not in ("joined", "per_chunk"):
        raise ValueError("passages must be 'joined' or 'per_chunk'")
    items = [e for e in evidence if e.chunk_text or not skip_empty_text]
    if order_by:
        items = sorted(items, key=lambda e: getattr(e, order_by))

    blocks: List[str] = []
    seps: List[str] = []
    current_doc = None
    for e in items:
        doc = e.document_name or unknown_document
        page = e.page if e.page is not None else unknown_page
        new_doc = doc != current_doc
        lines: List[str] = []
        if document_header is not None and (
            passages == "per_chunk" or not group_by_document or new_doc
        ):
            lines.append(document_header.format(document=doc, page=page))
        if page_line is not None:
            lines.append(page_line.format(document=doc, page=page))
        lines.append(e.chunk_text)
        blocks.append(line_sep.join(lines))
        seps.append(document_sep if (new_doc and current_doc is not None) else passage_sep)
        current_doc = doc

    if passages == "per_chunk":
        return blocks
    joined = ""
    for i, block in enumerate(blocks):
        joined += (seps[i] if i else "") + block
    return [joined + trailing if joined else ""]


@dataclass
class QueryRecord:
    query_id: str
    condition: str
    question: str
    answerable: bool
    category: str
    gold_answer: Optional[str]
    governing_doc: Optional[str]
    superseded_doc: Optional[str]

    # qrels for THIS cell's ingest, written by gold_evidence.apply_gold_to_records
    # after extraction. Never read from the benchmark file: chunk UUIDs do not
    # survive a re-upload, so an authored list would be stale here.
    judged_chunk_ids: List[str] = field(default_factory=list)

    # which repeat of the cell answered this question. A cell asked once carries
    # ``r1`` on every record; repeats are scored apart, then averaged per
    # question, so the same query_id appears once per run in a pooled cell.
    run_id: str = DEFAULT_RUN_ID

    message_id: Optional[str] = None
    answer_text: str = ""
    served_evidence: List[ServedEvidence] = field(default_factory=list)

    # completion log
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_total: Optional[int] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    latency_ms: Optional[int] = None

    # how the completion log was attached: ``message_id`` for an exact link,
    # ``conversation+time`` for a row the provider wrote before the assistant
    # message existed. None means no completion log was found at all.
    completion_link: Optional[str] = None

    # when the assistant message landed, ISO 8601. ``latency_ms`` covers
    # generation only, so the serving stretch this answer occupied is the sum of
    # retrieval, rerank and generation, and that is the interval the power log
    # is integrated over to price the answer's electricity.
    answered_at: Optional[str] = None

    # rag log
    top_k: Optional[int] = None
    top_n: Optional[int] = None
    retrieval_latency_ms: Optional[int] = None
    rerank_latency_ms: Optional[int] = None

    linked: bool = False       # False => no matching answer found for this cell

    @property
    def served_chunk_ids(self) -> List[str]:
        return [e.chunk_id for e in self.served_evidence]

    @property
    def context_text(self) -> str:
        """The served context as the model received it (faithfulness context).

        `render_served_context` defaults: document header, page line, served
        order. Backends that need another shape call it directly.
        """
        return render_served_context(self.served_evidence)[0]

    @property
    def has_evidence(self) -> bool:
        return len(self.served_evidence) > 0


# ---- SQL ------------------------------------------------------------------

def content_match_sql(
    column: str = "m.content",
    *,
    lower: bool = True,
    collapse_whitespace: bool = True,
    trim_chars: str = MATCH_TRIM_CHARS,
    trim_left: bool = True,
    trim_right: bool = True,
) -> str:
    """SQL that normalises a logged message, mirroring `benchmark.match_key`.

    Both sides of the comparison take the same arguments and are built from
    them, so a question that survives one survives the other.

    Args:
        column: the message column to normalise, aliased as the caller's query
            aliases it.
        lower / collapse_whitespace / trim_chars / trim_left / trim_right: read
            `benchmark.match_key`; each argument does the same work here.

    Returns:
        A SQL expression for embedding in a SQLAlchemy `text()`. A quote in
        ``trim_chars`` is doubled here; a colon has to be written ``\\:`` by the
        caller, which is why the shipped sets carry none.
    """
    expr = column
    if lower:
        expr = f"lower({expr})"
    if collapse_whitespace:
        expr = f"btrim(regexp_replace({expr}, '\\s+', ' ', 'g'))"
    fn = {(True, True): "btrim", (True, False): "ltrim", (False, True): "rtrim"}.get(
        (trim_left, trim_right)
    )
    if trim_chars and fn:
        expr = f"{fn}({expr}, '{trim_chars.replace(chr(39), chr(39) * 2)}')"
    return expr


#: Normalisation applied to the logged message before it is compared to the
#: benchmark question; mirrors `benchmark.match_key` in SQL.
DEFAULT_CONTENT_SQL = content_match_sql()


def user_message_sql(
    *,
    include_deleted_conversations: bool = False,
    role: str = "user",
    content_sql: str = DEFAULT_CONTENT_SQL,
    order_by: str = "m.created_at ASC",
    limit: int = 1,
    id_sql_type: str = ID_SQL_TYPE,
):
    """The query that finds the logged turn a benchmark question was asked in.

    Every filter the cell can carry is a bound parameter, so one statement serves
    a cell pinned by conversation id, by window, or by nothing.

    Args:
        include_deleted_conversations: False drops chats the app soft-deleted
            (`conversations.deleted_at`), which is what a re-run of a cell
            usually wants: delete the bad chat in the app, re-ask, re-dump.
        role: the message role that carries the question.
        content_sql: SQL that normalises the stored message for comparison.
        order_by / limit: first match wins by default; widen only if you want a
            different tie-break between two chats holding the same question.
        id_sql_type: type the ``conv_ids`` list is cast to, see `db.bind_cast`.
            ``conversations.id`` is a uuid, so the bound strings need the cast.
    """
    deleted = "" if include_deleted_conversations else "AND (c.deleted_at IS NULL)\n      "
    conv_ids = bind_cast("conv_ids", sql_type=id_sql_type, array=True)
    return text(
        f"""
    SELECT m.id AS user_id, m.conversation_id, m.created_at
    FROM messages m
    LEFT JOIN conversations c ON c.id = m.conversation_id
    WHERE m.role = '{role}'
      AND {content_sql} = :qtext
      {deleted}AND (:has_window = false OR (m.created_at >= :start AND m.created_at <= :end))
      AND (:has_convs = false OR m.conversation_id = ANY({conv_ids}))
    ORDER BY {order_by}
    LIMIT {int(limit)}
    """
    )


_FIND_USER_MSG = user_message_sql()

_NEXT_ASSISTANT = text(
    f"""
    SELECT m.id, m.content
    FROM messages m
    WHERE m.conversation_id = {bind_cast('conversation_id')}
      AND m.role = 'assistant'
      AND m.created_at >= :after
    ORDER BY m.created_at ASC
    LIMIT 1
    """
)

# Served-order reconstruction mirrors chat-service routes/messages.get_messages exactly.
_EVIDENCE = text(
    f"""
    SELECT me.chunk_id, me.document_name, me.page,
           me.retrieval_score, me.rerank_score,
           c.text AS chunk_text
    FROM message_evidence me
    LEFT JOIN chunks c ON c.id = me.chunk_id
    WHERE me.message_id = {bind_cast('message_id')}
    ORDER BY me.rerank_score DESC NULLS LAST, me.retrieval_score DESC NULLS LAST
    """
)

_COMPLETION = text(
    f"""
    SELECT provider, model, tokens_input, tokens_output, tokens_total, latency_ms
    FROM completion_logs
    WHERE message_id = {bind_cast('message_id')}
    ORDER BY created_at DESC
    LIMIT 1
    """
)

# The provider writes its own completion log the moment generation returns
# (chat-service services/completion_log.py), which is before routes/messages.py
# inserts the assistant message, so message_id is NULL on those rows:
# completion_logs.message_id carries a foreign key to messages(id) and the row
# does not exist yet. Such a log is attributed by conversation plus created_at
# proximity, the same association rag_logs already needs, and the skew bound
# keeps a second answer in the same conversation from claiming it.
_COMPLETION_NEAR = text(
    f"""
    SELECT provider, model, tokens_input, tokens_output, tokens_total, latency_ms,
           abs(extract(epoch FROM (created_at - :ts))) AS skew_seconds
    FROM completion_logs
    WHERE message_id IS NULL
      AND conversation_id = {bind_cast('conversation_id')}
      AND abs(extract(epoch FROM (created_at - :ts))) <= :max_skew
    ORDER BY skew_seconds ASC, created_at DESC
    LIMIT 1
    """
)

# rag_logs.message_id is inserted NULL by rag-service (retrieval.retrieve_evidence),
# and the table is app-wide: /api/v1/retrieve serves the chat at top_n=5
# (chat-service chat_handler._retrieve_evidence) and the compliance dashboard at
# top_n=100 (ingestion-service routes/compliance.get_all_finding_evidence and
# get_finding_evidence), whose panel polls while
# questions are being asked. Nearest-in-time alone therefore hands an answer a
# compliance call's pool width. The scores decide instead: message_evidence
# carries the retrieval score of every row this answer was served, and only the
# call that served them lists all of those scores.
_RAG_LOG_BY_SCORES = text(
    f"""
    SELECT l.top_k, l.top_n, l.retrieval_latency_ms, l.rerank_latency_ms,
           (SELECT count(*) FROM message_evidence e
             WHERE e.message_id = {bind_cast('message_id')}
               AND e.retrieval_score IS NOT NULL
               AND EXISTS (
                   SELECT 1 FROM jsonb_array_elements_text(l.retrieval_scores) s
                   WHERE round(s::numeric, :score_digits)
                       = round(e.retrieval_score::numeric, :score_digits)
               )) AS matched_scores
    FROM rag_logs l
    WHERE jsonb_typeof(l.retrieval_scores) = 'array'
      AND abs(extract(epoch FROM (l.created_at - :ts))) <= :max_skew
    ORDER BY matched_scores DESC, abs(extract(epoch FROM (l.created_at - :ts)))
    LIMIT 1
    """
)

_RAG_LOG_NEAR = text(
    """
    SELECT top_k, top_n, retrieval_latency_ms, rerank_latency_ms
    FROM rag_logs
    WHERE abs(extract(epoch FROM (created_at - :ts))) <= :max_skew
    ORDER BY abs(extract(epoch FROM (created_at - :ts)))
    LIMIT 1
    """
)

_EVIDENCE_SCORE_COUNT = text(
    f"""
    SELECT count(*) AS n FROM message_evidence
    WHERE message_id = {bind_cast('message_id')} AND retrieval_score IS NOT NULL
    """
)

_MSG_META = text(
    f"SELECT conversation_id, content, created_at FROM messages WHERE id = {bind_cast('id')}"
)


def _load_evidence(sess, message_id: str) -> List[ServedEvidence]:
    rows = sess.execute(_EVIDENCE, {"message_id": message_id}).fetchall()
    out: List[ServedEvidence] = []
    for rank, r in enumerate(rows):
        out.append(
            ServedEvidence(
                chunk_id=str(r.chunk_id),
                document_name=r.document_name,
                page=int(r.page) if r.page is not None else 0,
                retrieval_score=float(r.retrieval_score) if r.retrieval_score is not None else None,
                rerank_score=float(r.rerank_score) if r.rerank_score is not None else None,
                served_rank=rank,
                chunk_text=r.chunk_text or "",
            )
        )
    return out


def completion_log_for_message(
    sess,
    message_id: str,
    *,
    conversation_id: Optional[str] = None,
    answered_at=None,
    allow_unlinked: bool = True,
    max_skew_seconds: float = 120.0,
) -> Tuple[Optional[Any], Optional[str]]:
    """Return ``(completion_logs row, how it was linked)`` for one answer.

    The exact link is tried first and always wins, so a row the app wrote with a
    message_id is never displaced by a proximity match.

    Args:
        message_id: assistant message whose completion is wanted.
        conversation_id: conversation that message belongs to; the proximity
            path is scoped to one conversation. None reads it off the message
            row, and only when the exact link already missed.
        answered_at: that message's ``created_at`` (a datetime, as the driver
            returns it). The proximity path measures skew against it. None
            reads it off the message row, as with conversation_id.
        allow_unlinked: False restricts the lookup to rows carrying the exact
            message_id, which is what a reproduction of the pre-provider logging
            behaviour needs. True also accepts a provider-written row whose
            message_id is NULL.
        max_skew_seconds: how far a NULL-message_id row may sit from the
            assistant message and still be accepted. The provider logs a few
            milliseconds before the message row is inserted; a bound this wide
            absorbs clock skew between the two writers while staying far below
            the gap between two answers served by hand.

    Returns:
        (row, "message_id") for an exact link, (row, "conversation+time") for a
        proximity link, or (None, None) when no completion log matches.
    """
    exact = sess.execute(_COMPLETION, {"message_id": message_id}).first()
    if exact is not None:
        return exact, "message_id"
    if not allow_unlinked:
        return None, None
    if conversation_id is None or answered_at is None:
        meta = sess.execute(_MSG_META, {"id": str(message_id)}).first()
        if meta is None:
            return None, None
        if conversation_id is None:
            conversation_id = meta.conversation_id
        if answered_at is None:
            answered_at = meta.created_at
    if conversation_id is None or answered_at is None:
        return None, None
    near = sess.execute(
        _COMPLETION_NEAR,
        {
            "conversation_id": str(conversation_id),
            "ts": answered_at,
            "max_skew": float(max_skew_seconds),
        },
    ).first()
    if near is None:
        return None, None
    return near, "conversation+time"


def rag_log_for_message(
    sess,
    message_id: str,
    answered_at,
    *,
    match_on_scores: bool = True,
    score_digits: int = 4,
    min_matched_scores: Optional[int] = None,
    max_skew_seconds: float = 120.0,
    allow_nearest_fallback: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    """Return ``(rag_logs row, how it was linked)`` for one answer.

    Args:
        message_id: assistant message whose retrieve call is wanted.
        answered_at: that message's ``created_at``; both paths measure skew
            against it. None returns (None, None), since neither path has a key.
        match_on_scores: True claims the call whose ``retrieval_scores`` list
            covers the scores in ``message_evidence`` for this message, which is
            what separates the chat's own call from any other caller retrieving
            in the same seconds. False leaves only the time path.
        score_digits: decimals both sides are rounded to before comparison.
            message_evidence stores a rounded score while rag_logs keeps full
            precision, so an exact float comparison never matches; 4 is the
            column's own scale.
        min_matched_scores: how many of this answer's served scores the call must
            list to be accepted. None requires all of them, which is the only
            value that cannot be reached by a different query hitting the same
            chunk at the same score.
        max_skew_seconds: how far a row may sit from the assistant message and
            still be considered, on either path.
        allow_nearest_fallback: True accepts the nearest row in the window when
            the score match fails (an answer served no evidence, or the app
            logged no scores), which reproduces the pre-scores behaviour. False
            reports no rag log rather than a possibly foreign one.

    Returns:
        (row, "evidence_scores") for a score match, (row, "time") for the
        proximity fallback, or (None, None) when neither path answers.
    """
    if answered_at is None:
        return None, None
    params = {"ts": answered_at, "max_skew": float(max_skew_seconds)}
    if match_on_scores:
        wanted = min_matched_scores
        if wanted is None:
            row = sess.execute(
                _EVIDENCE_SCORE_COUNT, {"message_id": str(message_id)}
            ).first()
            wanted = int(row.n) if row is not None else 0
        if wanted > 0:
            hit = sess.execute(
                _RAG_LOG_BY_SCORES,
                dict(params, message_id=str(message_id), score_digits=int(score_digits)),
            ).first()
            if hit is not None and int(hit.matched_scores) >= wanted:
                return hit, "evidence_scores"
    if not allow_nearest_fallback:
        return None, None
    near = sess.execute(_RAG_LOG_NEAR, params).first()
    return (near, "time") if near is not None else (None, None)


def _fill_logs(
    sess,
    rec: QueryRecord,
    message_id: str,
    msg_created_at,
    conversation_id: Optional[str] = None,
    **completion_kwargs,
) -> None:
    comp, link = completion_log_for_message(
        sess,
        message_id,
        conversation_id=conversation_id,
        answered_at=msg_created_at,
        **completion_kwargs,
    )
    if comp:
        rec.provider = comp.provider
        rec.model = comp.model
        rec.tokens_input = comp.tokens_input
        rec.tokens_output = comp.tokens_output
        rec.tokens_total = comp.tokens_total
        rec.latency_ms = comp.latency_ms
        rec.completion_link = link
    if msg_created_at is not None:
        rec.answered_at = (
            msg_created_at.isoformat() if hasattr(msg_created_at, "isoformat")
            else str(msg_created_at)
        )
        rag, _rag_link = rag_log_for_message(sess, message_id, msg_created_at)
        if rag:
            rec.top_k = rag.top_k
            rec.top_n = rag.top_n
            rec.retrieval_latency_ms = rag.retrieval_latency_ms
            rec.rerank_latency_ms = rag.rerank_latency_ms


def _link_message_id(
    sess,
    q: BenchmarkQuery,
    cond: Condition,
    *,
    statement=None,
) -> tuple[Optional[str], object]:
    """Return (assistant_message_id, created_at) or (None, None).

    ``statement`` overrides the default query, so a caller that wants the
    soft-deleted chats builds one with `user_message_sql` and passes it here.
    """
    # 1) explicit linkage wins
    explicit = q.message_ids.get(cond.name)
    if explicit:
        meta = sess.execute(_MSG_META, {"id": explicit}).first()
        return explicit, (meta.created_at if meta else None)

    # 2) text match on the user turn, then the following assistant turn
    # One derivation of the id list, so the gate and the array cannot disagree.
    # Blank entries are dropped before the cast, which would reject them.
    conv_ids = [str(c).strip() for c in (cond.conversation_ids or ()) if str(c).strip()]
    user = sess.execute(
        statement if statement is not None else _FIND_USER_MSG,
        {
            "qtext": match_key(q.text),
            "has_window": bool(cond.start and cond.end),
            "start": cond.start,
            "end": cond.end,
            "has_convs": bool(conv_ids),
            "conv_ids": conv_ids or None,
        },
    ).first()
    if not user:
        return None, None
    asst = sess.execute(
        _NEXT_ASSISTANT,
        {"conversation_id": str(user.conversation_id), "after": user.created_at},
    ).first()
    if not asst:
        return None, None
    # created_at of the assistant message for rag-log proximity
    meta = sess.execute(_MSG_META, {"id": str(asst.id)}).first()
    return str(asst.id), (meta.created_at if meta else user.created_at)


def extract_condition(
    queries: List[BenchmarkQuery],
    cond: Condition,
    *,
    include_deleted_conversations: bool = False,
    run_id: Optional[str] = None,
) -> List[QueryRecord]:
    """Build one QueryRecord per benchmark query for the given condition.

    Args:
        include_deleted_conversations: passed to `user_message_sql`; True keeps
            the chats the app soft-deleted.
        run_id: stamped on every record so a pooled cell keeps its repeats apart.
            None takes the window's own ``cond.run_id``.
    """
    rid = cond.run_id if run_id is None else run_id
    statement = (
        _FIND_USER_MSG
        if not include_deleted_conversations
        else user_message_sql(include_deleted_conversations=True)
    )
    records: List[QueryRecord] = []
    with session_scope() as sess:
        for q in queries:
            rec = QueryRecord(
                query_id=q.id,
                condition=cond.name,
                run_id=rid,
                question=q.text,
                answerable=q.answerable,
                category=q.category,
                gold_answer=q.gold_answer,
                governing_doc=q.governing_doc,
                superseded_doc=q.superseded_doc,
            )
            msg_id, created_at = _link_message_id(sess, q, cond, statement=statement)
            if msg_id:
                asst = sess.execute(_MSG_META, {"id": msg_id}).first()
                rec.message_id = msg_id
                rec.answer_text = (asst.content if asst else "") or ""
                rec.served_evidence = _load_evidence(sess, msg_id)
                _fill_logs(
                    sess,
                    rec,
                    msg_id,
                    created_at,
                    conversation_id=(asst.conversation_id if asst else None),
                )
                rec.linked = True
            records.append(rec)
    return records


def extract_cell_runs(
    queries: List[BenchmarkQuery],
    runs: Sequence[Condition],
    *,
    include_deleted_conversations: bool = False,
    run_ids: Optional[Sequence[str]] = None,
    require_distinct: bool = True,
    flatten: bool = False,
) -> Dict[str, List[QueryRecord]]:
    """Extract every repeat of one cell, keyed by run id, so the same question
    comes back once per run instead of the last repeat overwriting the first.

    Args:
        runs: the cell's windows, in the order asked.
        run_ids: override their ids, position by position.
        require_distinct: raise when two repeats claim one run id.
        flatten: return ``{"": records}``, every run concatenated.
    """
    if run_ids is not None and len(run_ids) != len(runs):
        raise ValueError(f"run_ids has {len(run_ids)} entries for {len(runs)} runs")
    out: Dict[str, List[QueryRecord]] = {}
    for i, cond in enumerate(runs):
        rid = cond.run_id if run_ids is None else str(run_ids[i])
        if require_distinct and rid in out:
            raise ValueError(f"cell '{cond.name}': run id {rid!r} declared twice")
        recs = extract_condition(
            queries, cond,
            include_deleted_conversations=include_deleted_conversations,
            run_id=rid,
        )
        out.setdefault(rid, []).extend(recs)
    if flatten:
        return {"": [r for recs in out.values() for r in recs]}
    return out


def extract_all(queries: List[BenchmarkQuery], conditions: List[Condition]) -> Dict[str, List[QueryRecord]]:
    return {c.name: extract_condition(queries, c) for c in conditions}
