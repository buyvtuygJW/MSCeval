"""
Table handling utilities.

Hard rules:
- Tables are atomic: never sentence-split and never merged with narrative.
- Tables may exceed 1000 words.
- Store tables as:
  - raw representation
  - serialised text using tabulate
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

# `tabulate(tablefmt="github")` pads every cell to the widest cell in its
# column, so one long OCR line pads every other row to the same width: a served
# chunk of this corpus measures 85,081 characters of which 775 are not
# whitespace, and the 500 that reach the prompt are almost all padding. Off by
# default because collapsing the padding changes the stored chunk text, which
# means a re-ingest: control is already dumped and keeps the padded bytes.
COLLAPSE_TABLE_PADDING_DEFAULT = False


def default_collapse_table_padding() -> bool:
    """Table padding collapse; override via COLLAPSE_TABLE_PADDING."""
    raw = os.getenv("COLLAPSE_TABLE_PADDING")
    if raw is None or not raw.strip():
        return COLLAPSE_TABLE_PADDING_DEFAULT
    return raw.strip().lower() in ("1", "true", "yes", "on")


def collapse_table_padding(
    text: str,
    *,
    min_run: int = 2,
    replacement: str = " ",
    keep_newlines: bool = True,
    max_blank_lines: int = 1,
    strip_lines: bool = True,
) -> str:
    """Drop a serialized table's column padding, keeping every word and row.

    Args:
        min_run: shortest run of spaces or tabs that counts as padding. 2 leaves
            single spaces alone; 1 would rewrite ordinary prose as well.
        replacement: what a padding run becomes. "" glues neighbouring cells
            into one word, so the default keeps a single space as the boundary.
        keep_newlines: True keeps the row breaks, the only structure the table
            has left once the padding is gone. False returns a single line.
        max_blank_lines: consecutive blank lines kept; 0 drops them all.
        strip_lines: trim each row's leading and trailing whitespace.

    A snippet with no padding run comes back byte for byte, so prose chunks and
    the `github` header rule are untouched apart from their trailing spaces.
    """
    if not text:
        return text
    pattern = r"[ \t]{%d,}" % max(1, min_run)
    if not keep_newlines:
        return re.sub(pattern, replacement, text.replace("\n", replacement)).strip()
    out: List[str] = []
    blanks = 0
    for line in text.splitlines():
        line = re.sub(pattern, replacement, line)
        if strip_lines:
            line = line.strip()
        if line.strip():
            blanks = 0
        else:
            blanks += 1
            if blanks > max_blank_lines:
                continue
        out.append(line)
    return "\n".join(out)


def table_to_serialized_text(
    raw_text: str,
    html: Optional[str] = None,
    *,
    collapse: Optional[bool] = None,
    collapse_kwargs: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    Return (raw_representation, serialized_text).

    - raw_representation: original raw text or HTML if available (preferred for fidelity)
    - serialized_text: tabulated representation for stable text embedding & display

    Args:
        collapse: run `collapse_table_padding` over the serialized text, which is
            what the chunk builders store as the chunk's text. None reads
            COLLAPSE_TABLE_PADDING, whose default keeps the padded bytes.
        collapse_kwargs: forwarded to `collapse_table_padding`, so a run can move
            `min_run` or keep the blank rows without editing this file.
    """
    raw_representation = (html or raw_text or "").strip()
    do_collapse = default_collapse_table_padding() if collapse is None else collapse

    def _out(serialized: str) -> Tuple[str, str]:
        if not do_collapse:
            return raw_representation, serialized
        return raw_representation, collapse_table_padding(serialized, **(collapse_kwargs or {}))

    # Best-effort: if HTML is present, use pandas.read_html -> tabulate for a clean table.
    if html:
        try:
            import pandas as pd  # type: ignore
            from tabulate import tabulate  # type: ignore

            dfs = pd.read_html(html)
            if dfs:
                df = dfs[0]
                serialized = tabulate(df, headers="keys", tablefmt="github", showindex=False)
                return _out(serialized)
        except Exception:
            pass

    # Fallback: tabulate a single-column table to preserve row breaks.
    try:
        import pandas as pd  # type: ignore
        from tabulate import tabulate  # type: ignore

        rows = [[line.strip()] for line in (raw_text or "").splitlines() if line.strip()]
        if rows:
            df = pd.DataFrame(rows, columns=["table"])
            serialized = tabulate(df, headers="keys", tablefmt="github", showindex=False)
            return _out(serialized)
    except Exception:
        pass

    # Last resort: return raw text as-is.
    return _out((raw_text or "").strip())

