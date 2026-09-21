"""
PDF structural parsing.

Structural parsing (titles, paragraphs, lists) runs on the extracted page text
held in `document_pages`; the PDF itself is parsed once by ingestion-service
(`services/unstructured_parser.py`).

Torch-free mode:
- Do NOT use `unstructured.partition.pdf` (it pulls `unstructured_inference` -> `cv2` -> system GL libs, and may pull torch).
- Chunking operates on existing `document_pages` (text already extracted by ingestion-service).
- The paragraph/heading split is `chunking/text_partition.py`, standard library
  plus the spaCy model already loaded for sentences. It replaced
  `unstructured.partition.text.partition_text`, whose entire contribution here
  was one title/not-title flag while it pinned the image to an unpinned
  dependency, poppler/tesseract/libmagic, and NLTK data fetched at first call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Iterable, Dict, Sequence, Tuple

from .text_partition import partition_page_text


@dataclass(frozen=True)
class ParsedElement:
    kind: str  # "heading" | "narrative" | "table"
    text: str
    page_number: int
    # For tables (when available)
    table_html: Optional[str] = None


def _coerce_page_number(value: Any, default: int = 1) -> int:
    try:
        n = int(value)
        return n if n >= 1 else default
    except Exception:
        return default


def looks_like_table(
    text: str,
    *,
    min_lines: int = 2,
    pipe_char: str = "|",
    min_pipe_lines: int = 2,
    space_run: str = "  ",
    min_spaced_lines: int = 2,
    spaced_line_fraction: float = 0.5,
) -> bool:
    """
    Best-effort table heuristic for extracted text.

    Hard rule in chunker: tables are atomic and must not be merged with narrative.
    If this detects a table-ish block, we emit it as a table element.

    Defaults are the shipped rule, kept so a re-ingest reproduces today's chunks.
    Raising `spaced_line_fraction` or `min_spaced_lines` narrows the second test,
    which currently claims any page where half the lines contain a double space,
    justified prose and dotted contents pages included.

    Args:
        min_lines: pages with fewer non-empty lines are never tables.
        pipe_char / min_pipe_lines: markdown-style table test.
        space_run / min_spaced_lines / spaced_line_fraction: column-gap test; a
            page qualifies at `max(min_spaced_lines, floor(lines * fraction))`.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < min_lines:
        return False

    # Markdown-like tables
    pipe_lines = sum(1 for ln in lines if pipe_char in ln)
    if pipe_lines >= min_pipe_lines:
        return True

    # Multi-column spacing heuristic: many lines with 2+ spaces gaps
    spaced = sum(1 for ln in lines if space_run in ln)
    if spaced >= max(min_spaced_lines, int(len(lines) * spaced_line_fraction)):
        return True

    return False


# Kept for callers that imported the private name.
_looks_like_table = looks_like_table


def parse_pages_to_elements(
    pages: Iterable[Dict[str, Any]],
    *,
    partitioner: Callable[[str], Sequence[Tuple[str, str]]] = partition_page_text,
    table_detector: Callable[[str], bool] = looks_like_table,
    detect_tables: bool = True,
    table_kind: str = "table",
    page_number_key: str = "page_number",
    text_key: str = "text",
    default_page_number: int = 1,
) -> List[ParsedElement]:
    """
    Parse extracted page texts into structured elements.

    Returns a flat list of ParsedElement with page numbers, preserving ordering.

    Args:
        partitioner: page text -> `[(kind, text), ...]`; defaults to
            `text_partition.partition_page_text`. Pass a partial of it to retune
            paragraph or heading rules without touching this function.
        table_detector: page text -> is this a table; runs before partitioning
            because a split table cannot be put back together.
        detect_tables: False sends every page through the partitioner, so table
            pages are chunked as prose.
        table_kind: kind written for a detected table page.
        page_number_key / text_key: column names in the `document_pages` rows.
        default_page_number: used when a row carries no usable page number.
    """
    out: List[ParsedElement] = []

    for page in pages:
        page_number = _coerce_page_number(page.get(page_number_key), default=default_page_number)
        page_text = (page.get(text_key) or "").strip()
        if not page_text:
            continue

        # Table detection is done before partitioning so we don't accidentally split it.
        if detect_tables and table_detector(page_text):
            out.append(ParsedElement(kind=table_kind, text=page_text, page_number=page_number))
            continue

        for kind, text in partitioner(page_text):
            text = (text or "").strip()
            if not text:
                continue
            out.append(ParsedElement(kind=kind, text=text, page_number=page_number))

    return out

