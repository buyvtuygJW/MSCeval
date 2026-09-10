"""
Open a sitting: point one cell at the chat that actually holds its answers.

`veridic-eval sit control` reads the benchmark, finds the logged chat whose user
turns carry those exact questions, and writes that chat's real window (and, by
default, its conversation id) into conditions.yaml under ``control``.

This exists because the starter conditions file ships fabricated dates. A cell
whose window does not cover the sitting links zero answers, and the linking SQL
cannot say why: a window that excludes every row and a question nobody asked
both return no rows (`extract.user_message_sql`). Discovery here runs the same
comparison the linker runs, so a window written by this module is a window the
linker can use, and a question that cannot be found is reported by id instead of
being folded into one silent 0/10.

The app database is read-only, as everywhere else in this package. The only file
written is the conditions file, only after the patched text re-parses, and the
previous copy is kept beside it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text

from .benchmark import (
    MATCH_TRIM_CHARS_LOOSE,
    BenchmarkQuery,
    load_benchmark,
    match_key,
)
from .conditions import (
    CONDITIONS_KEY,
    RUN_ID_KEY,
    RUNS_KEY,
    load_conditions_file,
)
from .db import ID_SQL_TYPE, bind_cast, session_scope
from .ingest_cost import latest_build_policy, policies_agree
from .extract import DEFAULT_CONTENT_SQL, content_match_sql

__all__ = [
    "MatchedTurn",
    "Sitting",
    "SittingResult",
    "matched_turn_sql",
    "find_matched_turns",
    "probe_query_counts",
    "group_sittings",
    "sitting_window",
    "format_bound",
    "find_cell_block",
    "cell_child_indent",
    "normalise_cell_indent",
    "find_key_line",
    "patch_cell",
    "open_sitting",
]

#: Trailing comment written on every line this module rewrites, so a hand-edited
#: bound and a discovered one are told apart in a diff. None writes no comment
#: and leaves whatever comment the line already carried.
DEFAULT_STAMP = "set from the live chat by `veridic-eval sit`"

IDS_YAML_KEY = "conversation_ids"


# --------------------------------------------------------------- discovery

@dataclass
class MatchedTurn:
    """One logged user turn carrying one benchmark question, verbatim."""

    query_id: str
    conversation_id: str
    message_id: str
    created_at: datetime
    title: Optional[str] = None


@dataclass
class Sitting:
    """One candidate sitting: the chats that hold the questions, and when."""

    conversation_ids: List[str]
    #: The title every chat here shares, None when they differ. Printed in the
    #: report so the chats can be recognised; never written to the yaml.
    title: Optional[str]
    matched: List[str]
    missing: List[str]
    first_at: datetime
    last_at: datetime
    turns: List[MatchedTurn] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return len(self.matched)

    @property
    def complete(self) -> bool:
        return not self.missing


def matched_turn_sql(
    *,
    role: str = "user",
    content_sql: str = DEFAULT_CONTENT_SQL,
    include_deleted_conversations: bool = False,
    id_sql_type: str = ID_SQL_TYPE,
    order_by: str = "m.created_at ASC",
    limit: int = 5000,
):
    """Every logged turn whose normalised content is one of the benchmark texts.

    The content comparison is `extract.DEFAULT_CONTENT_SQL` by default, i.e. the
    one the linker uses, so discovery cannot succeed where linking will fail.

    Args:
        role: message role that carries the question.
        content_sql: SQL that normalises the stored message before comparison.
            Change it only to diagnose; the linker's own value is the default.
        include_deleted_conversations: False drops chats the app soft-deleted.
        id_sql_type: type the id lists are cast to, see `db.bind_cast`.
        order_by / limit: ask order and a ceiling on rows returned.
    """
    deleted = "" if include_deleted_conversations else "AND (c.deleted_at IS NULL)\n      "
    conv_ids = bind_cast("conv_ids", sql_type=id_sql_type, array=True)
    excl_ids = bind_cast("exclude_ids", sql_type=id_sql_type, array=True)
    qtexts = bind_cast("qtexts", sql_type="text", array=True)
    return text(
        f"""
    SELECT m.id AS message_id, m.conversation_id, m.created_at,
           c.title AS title, {content_sql} AS norm
    FROM messages m
    LEFT JOIN conversations c ON c.id = m.conversation_id
    WHERE m.role = '{role}'
      AND {content_sql} = ANY({qtexts})
      {deleted}AND (:has_since = false OR m.created_at >= :since)
      AND (:has_until = false OR m.created_at <= :until)
      AND (:has_convs = false OR m.conversation_id = ANY({conv_ids}))
      AND (:has_exclude = false OR NOT (m.conversation_id = ANY({excl_ids})))
    ORDER BY {order_by}
    LIMIT {int(limit)}
    """
    )


def find_matched_turns(
    sess,
    queries: Sequence[BenchmarkQuery],
    *,
    statement=None,
    normalise: Callable[[str], str] = match_key,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    conversation_ids: Sequence[str] = (),
    exclude_conversation_ids: Sequence[str] = (),
    **sql_kwargs: Any,
) -> List[MatchedTurn]:
    """The logged turns that carry these questions, in ask order.

    Args:
        sess: an open session; this function never opens or closes one.
        queries: benchmark items; their `text` is what is matched.
        statement: overrides the built statement, for a caller that wants a
            different `matched_turn_sql` shape.
        normalise: Python side of the comparison, mirroring `content_sql`.
        since / until: bounds on the search itself, not on the cell.
        conversation_ids: restrict to these chats.
        exclude_conversation_ids: skip these chats, e.g. ones another cell has
            already claimed.
        **sql_kwargs: forwarded to `matched_turn_sql`.
    """
    by_norm: Dict[str, List[str]] = {}
    for q in queries:
        by_norm.setdefault(normalise(q.text), []).append(q.id)
    if not by_norm:
        return []
    conv = [str(c).strip() for c in conversation_ids if str(c).strip()]
    excl = [str(c).strip() for c in exclude_conversation_ids if str(c).strip()]
    rows = sess.execute(
        statement if statement is not None else matched_turn_sql(**sql_kwargs),
        {
            "qtexts": list(by_norm),
            "has_since": since is not None,
            "since": since,
            "has_until": until is not None,
            "until": until,
            "has_convs": bool(conv),
            "conv_ids": conv or None,
            "has_exclude": bool(excl),
            "exclude_ids": excl or None,
        },
    ).fetchall()
    out: List[MatchedTurn] = []
    for r in rows:
        for qid in by_norm.get(r.norm, ()):
            out.append(
                MatchedTurn(
                    query_id=qid,
                    conversation_id=str(r.conversation_id),
                    message_id=str(r.message_id),
                    created_at=r.created_at,
                    title=r.title,
                )
            )
    return out


def probe_query_counts(
    sess,
    queries: Sequence[BenchmarkQuery],
    *,
    normalise: Callable[[str], str] = match_key,
    **sql_kwargs: Any,
) -> Dict[str, int]:
    """How many logged turns carry each question anywhere in the DB, unscoped.

    A question with 0 here was never asked, or was asked with different wording;
    a question with a count but no sitting is a scoping problem instead. That is
    the distinction `0/10 queries linked` cannot make on its own.
    """
    turns = find_matched_turns(sess, queries, normalise=normalise, **sql_kwargs)
    counts = {q.id: 0 for q in queries}
    for t in turns:
        counts[t.query_id] = counts.get(t.query_id, 0) + 1
    return counts


def group_sittings(
    turns: Sequence[MatchedTurn],
    queries: Sequence[BenchmarkQuery],
    *,
    min_matched: int = 1,
    order: str = "recent",
    merge: bool = True,
    max_conversations: int = 10,
) -> List[Sitting]:
    """Candidate sittings, best first, the winner spanning every chat it needs.

    One sitting is a set of chats, not one chat: the app opens a fresh chat
    whenever the tab is reloaded, so ten questions asked in one go routinely
    land in three chats. Ten questions found across three chats is a complete
    sitting, and reporting it as the newest chat's three would be a lie about
    the data.

    Args:
        min_matched: drop a chat carrying fewer questions than this. 1 keeps any
            chat that holds a single question, which is what a half-finished
            sitting looks like.
        order: ``recent`` ranks by the last question asked, ``matched`` by how
            many questions the chat holds, ``first`` by the earliest.
        merge: default. Fold further chats into the winner while they add
            questions it does not have, and stop as soon as the benchmark is
            covered. The window then spans them all, which is safe only because
            the ids are pinned beside it: the id list is the scope, the window
            is the second condition. Off gives one sitting per chat, for
            telling two sittings of the same cell apart by hand.
        max_conversations: ceiling on chats merged into one sitting.
    """
    if order not in ("recent", "matched", "first"):
        raise ValueError(f"order must be recent, matched or first, got {order!r}")
    all_ids = [q.id for q in queries]
    per_conv: Dict[str, List[MatchedTurn]] = {}
    for t in turns:
        per_conv.setdefault(t.conversation_id, []).append(t)

    def _build(conv_turns: Sequence[MatchedTurn], ids: Sequence[str]) -> Sitting:
        seen: List[str] = []
        for t in conv_turns:
            if t.query_id not in seen:
                seen.append(t.query_id)
        stamps = [t.created_at for t in conv_turns]
        # Report only: one title when every chat here shares it, so the printed
        # candidate is recognisable. Nothing selects on it.
        titles = {t.title for t in conv_turns}
        return Sitting(
            conversation_ids=list(ids),
            title=titles.pop() if len(titles) == 1 else None,
            matched=seen,
            missing=[q for q in all_ids if q not in seen],
            first_at=min(stamps),
            last_at=max(stamps),
            turns=sorted(conv_turns, key=lambda t: t.created_at),
        )

    singles = [
        _build(v, [k]) for k, v in per_conv.items() if len({t.query_id for t in v}) >= min_matched
    ]
    keys = {
        "recent": lambda s: (s.last_at, s.n_matched),
        "matched": lambda s: (s.n_matched, s.last_at),
        "first": lambda s: (-s.first_at.timestamp(), s.n_matched),
    }
    singles.sort(key=keys[order], reverse=True)
    if not merge or len(singles) < 2:
        return singles

    winner = singles[0]
    ids = list(winner.conversation_ids)
    pooled = list(winner.turns)
    have = set(winner.matched)
    for cand in singles[1:]:
        if len(ids) >= max_conversations or not [q for q in all_ids if q not in have]:
            break
        adds = [t for t in cand.turns if t.query_id not in have]
        if not adds:
            continue
        ids += cand.conversation_ids
        pooled += cand.turns
        have |= {t.query_id for t in adds}
    merged = _build(pooled, ids)
    return [merged] + singles[1:]


def sitting_window(
    sitting: Sitting,
    *,
    pad_before: timedelta = timedelta(seconds=60),
    pad_after: timedelta = timedelta(seconds=60),
    round_to: Optional[timedelta] = None,
) -> Tuple[datetime, datetime]:
    """The window to write for a sitting: its own bounds, padded.

    The linker compares inclusively (``>= start AND <= end``), so zero padding
    already covers every matched turn. The default minute of slack covers a
    question re-asked a moment later without reaching a neighbouring cell.

    Args:
        pad_before / pad_after: slack around the first and last matched turn.
        round_to: floor the start and ceil the end to this granularity, for a
            file that reads in round numbers. None writes the exact bounds.
    """
    start = sitting.first_at - pad_before
    end = sitting.last_at + pad_after
    if round_to is not None:
        secs = round_to.total_seconds()
        if secs <= 0:
            raise ValueError(f"round_to must be positive, got {round_to!r}")
        epoch = datetime(1970, 1, 1, tzinfo=start.tzinfo)
        start -= timedelta(seconds=(start - epoch).total_seconds() % secs)
        over = (end - epoch).total_seconds() % secs
        if over:
            end += timedelta(seconds=secs - over)
    return start, end


def format_bound(
    dt: datetime,
    *,
    aware_format: str = "%Y-%m-%dT%H:%M:%SZ",
    naive_format: str = "%Y-%m-%dT%H:%M:%S",
    to_utc: bool = True,
) -> str:
    """A window bound written the way the DB handed it over.

    An aware timestamp is written in UTC with the ``Z`` the parser expects. A
    naive one is written without a zone, because `conditions.parse_timestamp`
    leaves it naive and the comparison then matches the column it came from.
    Writing ``Z`` on a naive local timestamp is the silent hours-wide shift this
    split avoids.
    """
    if dt.tzinfo is not None:
        return (dt.astimezone(timezone.utc) if to_utc else dt).strftime(aware_format)
    return dt.strftime(naive_format)


# ------------------------------------------------------------ yaml surgery
#
# The conditions file is comment-heavy: the commented-out repeat under `runs:`,
# the per-cell factor notes, the header block. A yaml round trip drops all of
# it, so every write here is a line patch that touches only the bounds.

_KEY_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<hash>#[ \t]?)?(?P<dash>-[ \t]+)?"
    r"(?P<key>[A-Za-z_][\w.\-]*)[ \t]*:(?P<rest>.*)$"
)
_DASH_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<hash>#[ \t]?)?-[ \t]*(?P<rest>.*)$")
_VALUE_RE = re.compile(
    r"^(?P<pre>[ \t]*(?:#[ \t]?)?(?:-[ \t]+)?(?P<key>[A-Za-z_][\w.\-]*)[ \t]*:[ \t]*)"
    r"(?P<val>[^#]*?)(?P<gap>[ \t]*)(?P<comment>#.*)?$"
)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _is_blank(line: str) -> bool:
    return not line.strip()


def _is_comment(line: str) -> bool:
    return line.lstrip(" \t").startswith("#")


def _uncomment(line: str) -> str:
    """Drop one leading ``# `` and keep the column the live form would have."""
    return re.sub(r"^(?P<i>[ \t]*)#[ \t]?", lambda m: m.group("i"), line, count=1)


def _comment_out(line: str, *, prefix: str = "# ") -> str:
    i = _indent_of(line)
    return f"{line[:i]}{prefix}{line[i:]}"


def _set_value(
    line: str,
    value: str,
    *,
    stamp: Optional[str] = DEFAULT_STAMP,
    comment_col: int = 34,
) -> str:
    """Rewrite one ``key: value`` line, keeping its key spelling and column."""
    m = _VALUE_RE.match(line)
    if not m:
        raise ValueError(f"not a scalar yaml line: {line!r}")
    body = f"{m.group('pre')}{value}"
    if stamp is None:
        return body + m.group("gap") + (m.group("comment") or "")
    pad = " " * max(comment_col - len(body), 1)
    return f"{body}{pad}# {stamp}"


def _flow_ids(ids: Sequence[str]) -> str:
    return "[" + ", ".join(f'"{i}"' for i in ids) + "]"


def find_cell_block(
    lines: Sequence[str],
    cell: str,
    *,
    conditions_key: str = CONDITIONS_KEY,
) -> Tuple[int, int, int]:
    """``(first_line, end_line_exclusive, indent)`` of one cell's block.

    Comment lines and blank lines belong to the block they sit inside, so the
    commented-out repeat under ``runs:`` is never mistaken for the end of a cell.

    Raises:
        ValueError: no ``conditions:`` key, or no such cell, naming what the
            file does declare.
    """
    ci = -1
    for i, line in enumerate(lines):
        m = _KEY_RE.match(line)
        if m and not m.group("hash") and m.group("key") == conditions_key:
            ci = i
            break
    if ci < 0:
        raise ValueError(f"no live `{conditions_key}:` key in this file")

    cell_indent: Optional[int] = None
    found: List[str] = []
    start = -1
    for i in range(ci + 1, len(lines)):
        line = lines[i]
        if _is_blank(line) or _is_comment(line):
            continue
        indent = _indent_of(line)
        if indent <= _indent_of(lines[ci]):
            break
        m = _KEY_RE.match(line)
        if not m or m.group("dash"):
            continue
        if cell_indent is None:
            cell_indent = indent
        if indent != cell_indent:
            continue
        found.append(m.group("key"))
        if m.group("key") == cell:
            start = i
            break
    if start < 0:
        raise ValueError(
            f"no cell `{cell}` under `{conditions_key}:`; this file declares {found or 'none'}"
        )

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if _is_blank(line) or _is_comment(line):
            continue
        if _indent_of(line) <= cell_indent:
            end = i
            break
    while end - 1 > start and _is_blank(lines[end - 1]):
        end -= 1
    return start, end, int(cell_indent or 0)


def cell_child_indent(
    lines: Sequence[str],
    start: int,
    end: int,
    cell_indent: int,
    *,
    prefer_live: bool = False,
    comment_probe: bool = True,
    snap_to_live: bool = True,
    snap_slack: int = 1,
    default_extra: int = 2,
) -> str:
    """The column this cell's own keys sit at, as leading spaces.

    A comment probe can read one column deeper than the live keys beside it,
    because `_uncomment` drops the hash and a single space: ``#  start:`` under
    ``    key:`` probes 5 where the mapping is 4. A key written at 5 closes no
    mapping and the file stops parsing, so a comment probe is pulled onto the
    nearest live sibling within ``snap_slack``.

    Args:
        prefer_live: the first live child wins outright, and comments are read
            only when the cell has no live child. False reads the file in order.
        comment_probe: commented children may supply the column. False leaves a
            fully commented cell on ``cell_indent + default_extra``.
        snap_to_live / snap_slack: pull a comment probe onto the nearest live
            sibling column when it is within this many columns.
        default_extra: columns under the cell key when nothing else answers.
    """
    first: Optional[Tuple[int, bool]] = None
    live: List[int] = []
    for i in range(start + 1, end):
        line = lines[i]
        if _is_blank(line):
            continue
        commented = _is_comment(line)
        if commented and not comment_probe:
            continue
        ind = _indent_of(_uncomment(line) if commented else line)
        if ind <= cell_indent:
            continue
        if not commented:
            live.append(ind)
        if first is None:
            first = (ind, commented)
    if prefer_live and live:
        return " " * live[0]
    if first is None:
        return " " * (cell_indent + default_extra)
    ind, commented = first
    if commented and snap_to_live and live:
        near = min(live, key=lambda v: (abs(v - ind), v))
        if abs(near - ind) <= snap_slack:
            ind = near
    return " " * ind


def _reindent(line: str, indent: str) -> str:
    """Move one line to ``indent``, keeping its key, value and comment."""
    return indent + line.lstrip(" \t")


def normalise_cell_indent(
    lines: List[str],
    start: int,
    end: int,
    indent: str,
    *,
    slack: int = 1,
    include_comments: bool = True,
    keys: Optional[Sequence[str]] = None,
    changes: Optional[List[str]] = None,
    where: str = "",
) -> int:
    """Pull this cell's own keys onto ``indent``, and report how many moved.

    One column of drift on a key is invisible to read and fatal to parse: the
    mapping never closes, so every command that loads the file dies on it. A
    key inside ``slack`` columns of the cell's own column belongs to the cell,
    so it is moved there instead of being left for a hand edit.

    Args:
        indent: the column the cell's keys sit at, from `cell_child_indent`.
        slack: how far off ``indent`` a line may be and still be this cell's.
            Keeps a ``runs:`` entry's children, four columns deeper, out of it.
        include_comments: commented keys are moved too, so uncommenting one
            later cannot reintroduce the drift.
        keys: only these key names, None for every key.
        changes / where: one human line per move, appended for the caller.

    Returns:
        The number of lines moved.
    """
    want = len(indent)
    moved = 0
    for i in range(start + 1, end):
        line = lines[i]
        if _is_blank(line):
            continue
        commented = _is_comment(line)
        if commented and not include_comments:
            continue
        m = _KEY_RE.match(line)
        if not m or m.group("dash"):
            continue
        if keys is not None and m.group("key") not in keys:
            continue
        have = _indent_of(_uncomment(line) if commented else line)
        if have == want or abs(have - want) > slack:
            continue
        body = _uncomment(line) if commented else line
        lines[i] = _comment_out(_reindent(body, indent)) if commented else _reindent(body, indent)
        moved += 1
        if changes is not None:
            changes.append(
                f"{where}.{m.group('key')}: column {have} -> {want}"
                if where else f"{m.group('key')}: column {have} -> {want}"
            )
    return moved


def find_key_line(
    lines: Sequence[str],
    start: int,
    end: int,
    key: str,
    *,
    live_only: bool = False,
    exact_indent: Optional[int] = None,
    indent_slack: int = 0,
) -> int:
    """First line declaring ``key`` in a span, commented or not.

    ``exact_indent`` keeps a flat cell's own ``start:`` apart from the ``start:``
    inside a commented-out ``runs:`` entry, which sits four columns deeper.
    ``indent_slack`` widens that to a band, so a hand-edited bound one column off
    is adopted and repaired instead of being missed and duplicated beside itself.
    """
    for i in range(start, end):
        line = lines[i]
        m = _KEY_RE.match(line)
        if not m or m.group("key") != key:
            continue
        if live_only and m.group("hash"):
            continue
        if exact_indent is not None:
            if abs(_indent_of(_uncomment(line)) - exact_indent) > indent_slack:
                continue
        return i
    return -1


def _run_entries(
    lines: Sequence[str],
    runs_at: int,
    end: int,
    *,
    run_id_key: str = RUN_ID_KEY,
) -> List[Dict[str, Any]]:
    """The ``- `` entries under a ``runs:`` key, live and commented alike."""
    runs_indent = _indent_of(lines[runs_at])
    entries: List[Dict[str, Any]] = []
    for i in range(runs_at + 1, end):
        line = lines[i]
        if _is_blank(line):
            continue
        probe = _uncomment(line) if _is_comment(line) else line
        if _indent_of(probe) <= runs_indent and not _is_blank(probe):
            break
        if _DASH_RE.match(line):
            entries.append({"start": i, "end": i + 1, "commented": _is_comment(line), "id": None})
        elif entries:
            entries[-1]["end"] = i + 1
    for e in entries:
        rid = find_key_line(lines, e["start"], e["end"], run_id_key)
        if rid >= 0:
            m = _VALUE_RE.match(lines[rid])
            e["id"] = (m.group("val").strip().strip('"\'') if m else None)
    return entries


def patch_cell(
    source: str,
    cell: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    conversation_ids: Optional[Sequence[str]] = None,
    run: Optional[Any] = None,
    uncomment: bool = True,
    stamp: Optional[str] = DEFAULT_STAMP,
    comment_col: int = 34,
    conditions_key: str = CONDITIONS_KEY,
    runs_key: str = RUNS_KEY,
    run_id_key: str = RUN_ID_KEY,
    start_key: str = "start",
    end_key: str = "end",
    ids_key: str = IDS_YAML_KEY,
    bound_indent_slack: int = 1,
    reindent_bounds: bool = True,
    child_indent_snap: bool = True,
    child_indent_slack: int = 1,
    normalise_indent: bool = True,
    normalise_indent_slack: int = 1,
) -> Tuple[str, List[str]]:
    """Write bounds and ids into one cell, leaving every other line untouched.

    Handles the three shapes this package and its docs emit: a cell with a flat
    ``start``/``end`` pair, a cell with a ``runs:`` list, and a one-line flow
    mapping. A commented-out bound is uncommented rather than duplicated, so the
    starter file's commented repeat can be the target of ``run``.

    Args:
        start / end: preformatted bounds; None leaves that bound alone.
        conversation_ids: pinned chat ids; None leaves the key alone, an empty
            list comments an existing key out.
        run: which ``runs:`` entry to write, by 1-based number or by run id.
            None takes the first live entry, or the first entry when all of them
            are commented out.
        uncomment: a commented target is uncommented before it is written.
        stamp: trailing comment on every rewritten line; None keeps the existing
            comment instead.
        comment_col / conditions_key / runs_key / run_id_key / start_key /
        end_key / ids_key: layout and vocabulary, defaulted from
            `conditions.py` and `templates.py` so a patch cannot drift from what
            the parser reads.
        bound_indent_slack / reindent_bounds: a hand-edited ``start:``/``end:``
            this many columns off the cell's other keys is the target, and it is
            rewritten at their column. 0 / False restores the older behaviour,
            where such a bound was skipped and a second one inserted beside it.
        child_indent_snap / child_indent_slack: forwarded to `cell_child_indent`
            as ``snap_to_live`` / ``snap_slack``.
        normalise_indent / normalise_indent_slack: repair the cell's own key
            columns before writing, so a key drifted by a hand edit is moved
            back instead of failing every command that loads the file. Each move
            is reported in ``changes``.

    Returns:
        ``(new_text, changes)``, changes being one human line per edit.
    """
    newline = "\r\n" if "\r\n" in source else "\n"
    trailing = source.endswith(("\n", "\r"))
    lines = source.splitlines()
    si, ei, cell_indent = find_cell_block(lines, cell, conditions_key=conditions_key)
    changes: List[str] = []

    # ---- flow mapping: control: {start: ..., end: ...}
    head = _KEY_RE.match(lines[si])
    if head and head.group("rest").strip().startswith("{"):
        lines[si] = _patch_flow(
            lines[si], start=start, end=end, conversation_ids=conversation_ids,
            start_key=start_key, end_key=end_key, ids_key=ids_key, changes=changes,
        )
        out = newline.join(lines) + (newline if trailing else "")
        return out, changes

    ind = cell_child_indent(
        lines, si, ei, cell_indent,
        snap_to_live=child_indent_snap, snap_slack=child_indent_slack,
    )
    if normalise_indent:
        normalise_cell_indent(
            lines, si, ei, ind,
            slack=normalise_indent_slack, changes=changes, where=cell,
        )

    # ---- window
    runs_at = find_key_line(lines, si + 1, ei, runs_key, live_only=True)
    if runs_at >= 0:
        entries = _run_entries(lines, runs_at, ei, run_id_key=run_id_key)
        if not entries:
            raise ValueError(f"cell `{cell}` declares `{runs_key}:` with no entries")
        target = _pick_entry(entries, run, cell=cell, runs_key=runs_key)
        if target["commented"] and uncomment:
            for i in range(target["start"], target["end"]):
                if _is_comment(lines[i]):
                    lines[i] = _uncomment(lines[i])
            target["commented"] = False
            changes.append(f"{runs_key}[{target['id'] or '?'}]: uncommented")
        _write_bounds(
            lines, target["start"], target["end"], indent=ind + "    ",
            start=start, end=end, start_key=start_key, end_key=end_key,
            stamp=stamp, comment_col=comment_col, changes=changes,
            where=f"{runs_key}[{target['id'] or '?'}]",
        )
    else:
        _write_bounds(
            lines, si + 1, ei, indent=ind,
            start=start, end=end, start_key=start_key, end_key=end_key,
            stamp=stamp, comment_col=comment_col, changes=changes, where=cell,
            insert_at=si + 1, exact_indent=len(ind),
            indent_slack=bound_indent_slack, reindent=reindent_bounds,
        )

    # ---- pinned ids, at cell level so both shapes inherit them
    if conversation_ids is not None:
        si2, ei2, _ = find_cell_block(lines, cell, conditions_key=conditions_key)
        ii = find_key_line(lines, si2 + 1, ei2, ids_key)
        if not conversation_ids:
            if ii >= 0 and not _is_comment(lines[ii]):
                lines[ii] = _comment_out(lines[ii])
                changes.append(f"{ids_key}: commented out")
        elif ii >= 0:
            if _is_comment(lines[ii]):
                lines[ii] = _uncomment(lines[ii])
            lines[ii] = _set_value(lines[ii], _flow_ids(conversation_ids),
                                   stamp=stamp, comment_col=comment_col)
            changes.append(f"{ids_key}: -> {_flow_ids(conversation_ids)}")
        else:
            lines.insert(si2 + 1, _set_value(
                f"{ind}{ids_key}: x", _flow_ids(conversation_ids),
                stamp=stamp, comment_col=comment_col))
            changes.append(f"{ids_key}: added {_flow_ids(conversation_ids)}")

    out = newline.join(lines) + (newline if trailing else "")
    return out, changes


def _pick_entry(entries: Sequence[Dict[str, Any]], run: Optional[Any], *, cell: str, runs_key: str):
    if run is None:
        for e in entries:
            if not e["commented"]:
                return e
        return entries[0]
    if isinstance(run, int) or (isinstance(run, str) and run.isdigit()):
        n = int(run)
        if not 1 <= n <= len(entries):
            raise ValueError(
                f"cell `{cell}` has {len(entries)} `{runs_key}:` entries, asked for {n}"
            )
        return entries[n - 1]
    for e in entries:
        if e["id"] == run:
            return e
    raise ValueError(
        f"cell `{cell}` has no `{runs_key}:` entry with id {run!r}; "
        f"ids are {[e['id'] for e in entries]}"
    )


def _write_bounds(
    lines: List[str],
    start_i: int,
    end_i: int,
    *,
    indent: str,
    start: Optional[str],
    end: Optional[str],
    start_key: str,
    end_key: str,
    stamp: Optional[str],
    comment_col: int,
    changes: List[str],
    where: str,
    insert_at: Optional[int] = None,
    exact_indent: Optional[int] = None,
    indent_slack: int = 0,
    reindent: bool = False,
) -> None:
    """Set both bounds inside one span, uncommenting or inserting as needed.

    ``indent_slack`` / ``reindent`` adopt a bound that sits a column off ``indent``
    and rewrite it there, rather than leaving it and inserting a second one that
    no parser accepts beside it.
    """
    label = f"{end_key}:".ljust(len(start_key) + 1)
    for key, value, rendered in (
        (start_key, start, f"{indent}{start_key}: x"),
        (end_key, end, f"{indent}{label} x"),
    ):
        if value is None:
            continue
        i = find_key_line(lines, start_i, end_i, key, live_only=True,
                          exact_indent=exact_indent, indent_slack=indent_slack)
        if i < 0:
            i = find_key_line(lines, start_i, end_i, key,
                              exact_indent=exact_indent, indent_slack=indent_slack)
        if i >= 0:
            if _is_comment(lines[i]):
                lines[i] = _uncomment(lines[i])
            moved = reindent and _indent_of(lines[i]) != len(indent)
            if moved:
                lines[i] = _reindent(lines[i], indent)
            lines[i] = _set_value(lines[i], value, stamp=stamp, comment_col=comment_col)
            note = f" (column {len(indent)})" if moved else ""
            changes.append(f"{where}.{key}: -> {value}{note}")
        else:
            at = insert_at if insert_at is not None else end_i
            lines.insert(at, _set_value(rendered, value, stamp=stamp, comment_col=comment_col))
            end_i += 1
            if insert_at is not None:
                insert_at = at + 1
            changes.append(f"{where}.{key}: added {value}")


def _patch_flow(
    line: str,
    *,
    start: Optional[str],
    end: Optional[str],
    conversation_ids: Optional[Sequence[str]],
    start_key: str,
    end_key: str,
    ids_key: str,
    changes: List[str],
) -> str:
    """Rewrite ``cell: {start: ..., end: ...}`` in place, key by key."""
    def _sub(src: str, key: str, value: str) -> str:
        pat = re.compile(rf"(\b{re.escape(key)}\s*:\s*)([^,}}]*)")
        if pat.search(src):
            changes.append(f"{key}: -> {value}")
            return pat.sub(lambda m: m.group(1) + value, src, count=1)
        changes.append(f"{key}: added {value}")
        return re.sub(r"\}\s*$", f", {key}: {value}}}", src, count=1)

    out = line
    if start is not None:
        out = _sub(out, start_key, start)
    if end is not None:
        out = _sub(out, end_key, end)
    if conversation_ids:
        out = _sub(out, ids_key, _flow_ids(conversation_ids))
    return out


# ------------------------------------------------------------ orchestration

@dataclass
class SittingResult:
    """What one `open_sitting` did, or why it did nothing."""

    cell: str
    path: str
    ok: bool = False
    sitting: Optional[Sitting] = None
    candidates: List[Sitting] = field(default_factory=list)
    start: Optional[str] = None
    end: Optional[str] = None
    changes: List[str] = field(default_factory=list)
    written: bool = False
    backup: Optional[str] = None
    reason: str = ""
    probe: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cell": self.cell,
            "path": self.path,
            "ok": self.ok,
            "written": self.written,
            "backup": self.backup,
            "start": self.start,
            "end": self.end,
            "changes": list(self.changes),
            "reason": self.reason,
            "conversation_ids": list(self.sitting.conversation_ids) if self.sitting else [],
            "title": self.sitting.title if self.sitting else None,
            "matched": list(self.sitting.matched) if self.sitting else [],
            "missing": list(self.sitting.missing) if self.sitting else [],
            "probe": dict(self.probe),
        }


def open_sitting(
    cell: str,
    *,
    conditions_path: str = "conditions.yaml",
    benchmark_path: Optional[str] = None,
    queries: Optional[Sequence[BenchmarkQuery]] = None,
    session=None,
    # discovery
    conversation_ids: Sequence[str] = (),
    exclude_conversation_ids: Sequence[str] = (),
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    role: str = "user",
    content_sql: str = DEFAULT_CONTENT_SQL,
    include_deleted_conversations: bool = False,
    min_matched: int = 1,
    order: str = "recent",
    merge: bool = True,
    max_conversations: int = 10,
    probe_on_empty: bool = True,
    probe_content_sql: Sequence[str] = (
        content_match_sql(trim_chars=MATCH_TRIM_CHARS_LOOSE),
    ),
    probe_normalise: Optional[Callable[[str], str]] = None,
    # window
    pad_before: timedelta = timedelta(seconds=60),
    pad_after: timedelta = timedelta(seconds=60),
    round_to: Optional[timedelta] = None,
    aware_format: str = "%Y-%m-%dT%H:%M:%SZ",
    naive_format: str = "%Y-%m-%dT%H:%M:%S",
    to_utc: bool = True,
    # what is written
    write_window: bool = True,
    pin_ids: bool = True,
    run: Optional[Any] = None,
    stamp: Optional[str] = DEFAULT_STAMP,
    comment_col: int = 34,
    conditions_key: str = CONDITIONS_KEY,
    runs_key: str = RUNS_KEY,
    run_id_key: str = RUN_ID_KEY,
    start_key: str = "start",
    end_key: str = "end",
    ids_key: str = IDS_YAML_KEY,
    # file handling
    dry_run: bool = False,
    backup_suffix: Optional[str] = ".bak",
    encoding: str = "utf-8",
    verify: bool = True,
    check_covers: bool = True,
    check_policy: bool = True,
    policy_match: str = "exact",
    policy_kwargs: Optional[Dict[str, Any]] = None,
    printer: Optional[Callable[[str], None]] = print,
) -> SittingResult:
    """Point one cell at the chat that holds its answers, and write it down.

    Discovery, window, and file patch in one call: find the chat whose logged
    turns carry the benchmark questions, derive the window from those turns, and
    write both into the cell. The file is re-parsed afterwards, and restored
    from the backup if the patch made it unreadable.

    Args:
        cell: the cell to open, a key under ``conditions:``.
        conditions_path: file to patch.
        benchmark_path: benchmark to read; None uses `config.settings`.
        queries: preloaded benchmark items, skipping the file read.
        session: an open DB session; None opens one for this call.
        conversation_ids: skip discovery of which chat, and use these.
        exclude_conversation_ids: chats another cell already claimed.
        since / until: bounds on the search, e.g. today only.
        role / content_sql / include_deleted_conversations: the comparison, kept
            identical to the linker's by default.
        min_matched / order / merge / max_conversations: candidate ranking, see
            `group_sittings`.
        probe_on_empty / probe_content_sql / probe_normalise: when nothing
            matches, re-ask with these looser comparisons and report which
            questions they do find. Both halves loosen together: the SQL side
            strips `MATCH_TRIM_CHARS_LOOSE` and so does the Python key unless
            `probe_normalise` says otherwise, so an untrimmed message or a
            question ending in different punctuation is named rather than
            guessed at.
        pad_before / pad_after / round_to: window shaping, see `sitting_window`.
        aware_format / naive_format / to_utc: how a bound is written, see
            `format_bound`.
        write_window: False pins the chat and leaves the window alone.
        pin_ids: also write ``conversation_ids``, so a repeat of the same
            question in another chat cannot be picked up by the window.
        run: which ``runs:`` entry to write, see `patch_cell`.
        stamp / comment_col / *_key: forwarded to `patch_cell`.
        dry_run: compute and report, write nothing.
        backup_suffix: copy kept beside the file; None disables the backup and
            with it the restore on a failed verify.
        verify: re-parse the patched file, restore the backup if it fails.
        check_covers: assert the parsed cell now covers every matched turn.
        check_policy / policy_match / policy_kwargs: after the window is written,
            name the chunking policy the live corpus carries and compare it with
            the cell's ``expect_policy:``, through `say_policy_verdict`. It
            writes nothing and fails nothing: it is the one moment a forgotten
            ``CHUNKING_MODE`` flip is still cheap to fix, because the answers can
            be asked again before the cell is dumped.
        printer: line sink; None silences this call.
    """
    say = printer or (lambda _m: None)
    result = SittingResult(cell=cell, path=conditions_path)
    if queries is None:
        from .config import settings
        queries = load_benchmark(benchmark_path or settings.benchmark_path)
    if not queries:
        result.reason = "the benchmark has no queries"
        say(f"{cell}: {result.reason}")
        return result

    sql_kwargs = dict(
        role=role,
        content_sql=content_sql,
        include_deleted_conversations=include_deleted_conversations,
    )
    own = session is None
    ctx = session_scope() if own else None
    sess = ctx.__enter__() if own else session
    try:
        turns = find_matched_turns(
            sess, queries,
            since=since, until=until,
            conversation_ids=conversation_ids,
            exclude_conversation_ids=exclude_conversation_ids,
            **sql_kwargs,
        )
        result.candidates = group_sittings(
            turns, queries,
            min_matched=min_matched, order=order,
            merge=merge, max_conversations=max_conversations,
        )
        if not result.candidates:
            result.reason = (
                f"no logged chat carries these questions (searched {len(queries)} of them)"
            )
            say(f"{cell}: {result.reason}")
            if probe_on_empty:
                result.probe = probe_query_counts(sess, queries, **sql_kwargs)
                missing = [q for q, n in result.probe.items() if not n]
                if missing:
                    say(f"  never asked, or asked in other words: {', '.join(missing)}")
                probe_key = probe_normalise or (
                    lambda s: match_key(s, trim_chars=MATCH_TRIM_CHARS_LOOSE)
                )
                for probe_sql in probe_content_sql:
                    loose = probe_query_counts(
                        sess,
                        queries,
                        normalise=probe_key,
                        **{**sql_kwargs, "content_sql": probe_sql},
                    )
                    gained = [q for q in loose if loose[q] and not result.probe.get(q)]
                    if gained:
                        say(f"  matched only under `{probe_sql}`: {', '.join(gained)}")
                        say("  the linker compares with the strict form "
                            "(extract.DEFAULT_CONTENT_SQL), so those turns cannot link")
            return result

        sitting = result.candidates[0]
        result.sitting = sitting
    finally:
        if own and ctx is not None:
            ctx.__exit__(None, None, None)

    start_dt, end_dt = sitting_window(
        sitting, pad_before=pad_before, pad_after=pad_after, round_to=round_to
    )
    fmt = dict(aware_format=aware_format, naive_format=naive_format, to_utc=to_utc)
    result.start = format_bound(start_dt, **fmt) if write_window else None
    result.end = format_bound(end_dt, **fmt) if write_window else None

    with open(conditions_path, "r", encoding=encoding) as fh:
        source = fh.read()
    patched, changes = patch_cell(
        source, cell,
        start=result.start, end=result.end,
        conversation_ids=list(sitting.conversation_ids) if pin_ids else None,
        run=run, stamp=stamp, comment_col=comment_col,
        conditions_key=conditions_key, runs_key=runs_key, run_id_key=run_id_key,
        start_key=start_key, end_key=end_key, ids_key=ids_key,
    )
    result.changes = changes

    say(
        f"{cell}: {sitting.n_matched}/{len(queries)} questions found in "
        f"{len(sitting.conversation_ids)} chat(s)"
        + (f" titled {sitting.title!r}" if sitting.title else "")
    )
    if sitting.missing:
        say(f"  not in that chat: {', '.join(sitting.missing)}")
    for c in changes:
        say(f"  {c}")
    if dry_run:
        result.ok = True
        result.reason = "dry run, nothing written"
        say(f"  (dry run) {conditions_path} left as it is")
        return result
    if patched == source:
        result.ok = True
        result.reason = "already pointed at this sitting"
        say(f"  {conditions_path} already says this")
        return result

    if backup_suffix:
        result.backup = conditions_path + backup_suffix
        with open(result.backup, "w", encoding=encoding, newline="") as fh:
            fh.write(source)
    with open(conditions_path, "w", encoding=encoding, newline="") as fh:
        fh.write(patched)
    result.written = True

    if verify:
        try:
            parsed = load_conditions_file(conditions_path, printer=None)
            cond = next(
                (c.condition for c in parsed.cells if c.name == cell), None
            )
            if cond is None:
                raise ValueError(f"cell `{cell}` vanished from the patched file")
            if check_covers and cond.start and cond.end:
                for t in sitting.turns:
                    if not (cond.start <= t.created_at <= cond.end):
                        raise ValueError(
                            f"patched window {cond.start} -> {cond.end} does not cover "
                            f"{t.query_id} asked at {t.created_at}"
                        )
        except Exception as exc:
            if result.backup and os.path.exists(result.backup):
                with open(result.backup, "r", encoding=encoding) as fh:
                    restore = fh.read()
                with open(conditions_path, "w", encoding=encoding, newline="") as fh:
                    fh.write(restore)
                result.written = False
            result.reason = f"patch rejected: {type(exc).__name__}: {exc}"
            say(f"  {result.reason}")
            say(f"  {conditions_path} restored" if result.backup else "  file left as patched")
            return result

    result.ok = True
    say(f"  {conditions_path} updated" + (f" (backup {result.backup})" if result.backup else ""))
    if check_policy:
        say_policy_verdict(
            cell,
            conditions_path=conditions_path,
            since=sitting.first_at,
            as_of=sitting.last_at,
            session=session,
            match=policy_match,
            printer=printer,
            **(policy_kwargs or {}),
        )
    return result


def say_policy_verdict(
    cell: str,
    *,
    conditions_path: str = "conditions.yaml",
    since: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
    session=None,
    match: str = "exact",
    bound_to_sitting: bool = False,
    printer: Optional[Callable[[str], None]] = print,
    latest_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[bool], str]:
    """
    Say which chunking policy served the sitting that was just written down.

    A cell that declares ``expect_policy:`` says which builder was supposed to
    have chunked the corpus these answers were served from. Flipping
    ``CHUNKING_MODE`` and re-ingesting is a step outside this package, and a
    forgotten flip leaves a cell that reads perfectly and measures the other
    cell's builder, so this call names the live label the moment the window is
    known and re-asking the questions is still cheap. It writes nothing.

    Args:
        since / as_of: bound the build rows read; ``bound_to_sitting`` False
            ignores them and takes the newest build in the table, because the
            build that served a sitting is the one before its first answer, not
            one inside the window.
        bound_to_sitting: True bounds the read by the sitting instead.
        match: how strictly the labels must agree, `POLICY_MATCHES`.
        latest_kwargs: `ingest_cost.latest_build_policy` knobs.

    Returns:
        ``(ok, line)``; ok is None when there is nothing to judge.
    """
    say = printer or (lambda _m: None)
    try:
        parsed = load_conditions_file(conditions_path, printer=None)
        expected = (parsed.expect_policy or {}).get(cell)
    except Exception as exc:
        line = f"  policy check skipped: {conditions_path} unreadable ({type(exc).__name__}: {exc})"
        say(line)
        return None, line
    if not expected:
        line = f"  policy check skipped: {cell} declares no expect_policy"
        say(line)
        return None, line
    label, note = latest_build_policy(
        since=since if bound_to_sitting else None,
        as_of=as_of if bound_to_sitting else None,
        session=session,
        **(latest_kwargs or {}),
    )
    if label is None:
        line = f"  policy check: {cell} expects {expected}, but {note}"
        say(line)
        return None, line
    ok, why = policies_agree(label, expected, match=match)
    if ok:
        line = f"  policy check: {cell} was served by {label} as declared ({note})"
        say(line)
        return True, line
    line = (
        f"  POLICY MISMATCH: {cell} expects {expected}, the corpus in the database was "
        f"chunked by {label} ({match}: {why}, {note}). These answers measure that "
        f"builder. Set CHUNKING_MODE for {cell}, recreate rag-service, re-ingest, and "
        f"ask the questions again before dumping this cell"
    )
    say(line)
    return False, line


def open_sittings(
    cells: Sequence[str],
    *,
    claim_conversations: bool = True,
    printer: Optional[Callable[[str], None]] = print,
    **kwargs: Any,
) -> List[SittingResult]:
    """`open_sitting` for several cells, one sitting each.

    Args:
        cells: cells to open, in the order they were asked.
        claim_conversations: a chat written into one cell is excluded from the
            next cell's discovery, so two cells cannot claim the same sitting.
        printer: forwarded; None silences the whole batch.
        **kwargs: forwarded to `open_sitting`.
    """
    claimed: List[str] = list(kwargs.pop("exclude_conversation_ids", ()) or [])
    out: List[SittingResult] = []
    for name in cells:
        res = open_sitting(
            name, exclude_conversation_ids=tuple(claimed), printer=printer, **kwargs
        )
        out.append(res)
        if claim_conversations and res.sitting:
            claimed += [c for c in res.sitting.conversation_ids if c not in claimed]
    return out
