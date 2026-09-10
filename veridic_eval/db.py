"""
Read-only database access to the app's Postgres.

We deliberately do NOT import the backend package; the eval is a separate repo
and only needs raw SQL over a handful of tables. A default read-only transaction
is used so an eval run can never mutate the app's data.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from .config import settings

# ---- SQL typing -----------------------------------------------------------
#: The app declares every id column (``messages.id``, ``conversations.id``,
#: ``message_evidence.message_id``, ``chunks.id``) as Postgres ``uuid``. psycopg2
#: binds a Python ``str`` as ``text`` and ``uuid = text`` has no operator, so every
#: id comparison in this package types its bound parameter through `bind_cast`,
#: and the rule is written here once.
ID_SQL_TYPE = "uuid"


def bind_cast(name: str, *, sql_type: str = ID_SQL_TYPE, array: bool = False) -> str:
    """``CAST(:name AS uuid)``: one bound parameter, typed like the column.

    Casting the parameter instead of the column keeps the comparison on the
    indexed side, accepts an id in whatever letter case the yaml carries, and
    fails loudly on a malformed id rather than matching no row.

    Args:
        name: parameter name as it appears in the params dict, without the colon.
        sql_type: type to cast to; ``text`` makes the cast a no-op for a schema
            that stores ids as strings.
        array: True casts to ``uuid[]`` for ``= ANY(...)`` against a list.
            psycopg2 renders a list of strings as ``ARRAY['a','b']``, whose
            elements are text, and that is the mismatch this cast removes. An
            empty list renders as ``'{}'`` and ``None`` as ``NULL``; the cast
            takes either, so a caller may pass whichever it already holds.
    """
    if not name.isidentifier():
        raise ValueError(f"bind parameter name must be an identifier, got {name!r}")
    return f"CAST(:{name} AS {sql_type}{'[]' if array else ''})"


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.postgres_url,
            poolclass=NullPool,
            future=True,
        )

        # Belt & braces: force every connection into read-only mode so the eval
        # physically cannot write to the app database (Postgres honours this).
        @event.listens_for(_engine, "connect")
        def _set_read_only(dbapi_conn, _rec):  # pragma: no cover - driver glue
            try:
                cur = dbapi_conn.cursor()
                cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
                cur.close()
            except Exception:
                # Non-Postgres backends (e.g. sqlite in tests) may not support it.
                pass

    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Read-only session; rolls back on exit (never commits)."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    sess = _SessionLocal()
    try:
        yield sess
    finally:
        sess.rollback()
        sess.close()


def ping() -> bool:
    """True if the database is reachable."""
    try:
        with session_scope() as s:
            s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
