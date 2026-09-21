"""
Chunk building logic.

Constraints:
- Sentence-level atomicity (never split sentences; spaCy only)
- <= 1000 words per narrative chunk
- 200-word overlap, sentence-aligned
- Preserve headings / section hierarchy (section_path)
- Tables are atomic and never merged with narrative
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

from .parser import ParsedElement
from .sentence import split_sentences, word_count
from .table_utils import table_to_serialized_text


MAX_WORDS_DEFAULT = 1000
OVERLAP_WORDS_DEFAULT = 200


@dataclass
class BuiltChunk:
    document_id: str
    chunk_index: int
    text: str
    word_count: int
    sentence_count: int
    page_start: int
    page_end: int
    section_path: List[str]
    contains_table: bool
    overlap_with_previous: bool

    # For tables (optional but required to be stored somewhere)
    table_raw: Optional[str] = None
    table_text: Optional[str] = None

    # Set only by chunking/token_budget.py, when one atom (a sentence, a heading
    # or a serialized table) was larger than the token budget and had to be cut
    # into `sub_count` pieces. This builder never sets them.
    sub_index: Optional[int] = None
    sub_count: Optional[int] = None
    sub_of: Optional[str] = None  # "sentence" | "heading" | "table"


def _is_probable_top_level_heading(text: str) -> bool:
    # Heuristic: short + mostly uppercase => treat as top-level.
    words = [w for w in text.split() if w]
    if len(words) <= 8 and text.upper() == text:
        return True
    return False


def _update_section_path(current: List[str], heading: str) -> List[str]:
    h = heading.strip()
    if not h:
        return current
    if _is_probable_top_level_heading(h):
        return [h]
    # Default: append as a deeper section
    return [*current, h]


@dataclass(frozen=True)
class _Unit:
    kind: str  # "sentence" | "heading"
    text: str
    page_number: int


def _finalize_narrative_chunk(
    document_id: str,
    chunk_index: int,
    units: Sequence[_Unit],
    section_path: List[str],
    overlap_with_previous: bool,
) -> BuiltChunk:
    text = "\n".join([u.text for u in units]).strip()
    pages = [u.page_number for u in units] or [1]
    sentence_count = sum(1 for u in units if u.kind == "sentence")
    wc = word_count(text)
    return BuiltChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        word_count=wc,
        sentence_count=sentence_count,
        page_start=min(pages),
        page_end=max(pages),
        section_path=list(section_path),
        contains_table=False,
        overlap_with_previous=overlap_with_previous,
    )


def build_chunks(
    document_id: str,
    elements: Iterable[ParsedElement],
    *,
    max_words: int = MAX_WORDS_DEFAULT,
    overlap_words: int = OVERLAP_WORDS_DEFAULT,
) -> Iterator[BuiltChunk]:
    """
    Build chunks from parsed elements.

    Narrative chunks:
    - accumulate full sentences up to max_words
    - start next chunk with sentence-aligned overlap of ~overlap_words

    Table chunks:
    - one chunk per table element
    - never mixed with narrative
    - can exceed max_words
    """
    section_path: List[str] = []
    chunk_index = 0

    current_units: List[_Unit] = []
    current_words = 0
    overlap_units_next: List[_Unit] = []
    next_chunk_has_overlap = False

    def flush_current() -> Optional[BuiltChunk]:
        nonlocal chunk_index, current_units, current_words, overlap_units_next, next_chunk_has_overlap
        if not current_units:
            return None

        built = _finalize_narrative_chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            units=current_units,
            section_path=section_path,
            overlap_with_previous=next_chunk_has_overlap,
        )
        chunk_index += 1

        # Compute overlap (sentence-aligned) for next chunk.
        overlap_units_next = []
        overlap_wc = 0
        for u in reversed(current_units):
            if u.kind != "sentence":
                continue
            w = word_count(u.text)
            if overlap_wc + w > overlap_words and overlap_units_next:
                break
            overlap_units_next.append(u)
            overlap_wc += w
        overlap_units_next = list(reversed(overlap_units_next))

        # Reset current chunk, but seed the next chunk with overlap units.
        current_units = list(overlap_units_next)
        current_words = sum(word_count(u.text) for u in current_units if u.kind == "sentence")
        next_chunk_has_overlap = bool(current_units)
        return built

    for el in elements:
        if el.kind == "heading":
            # Heading changes section path, but also should be preserved in text.
            section_path = _update_section_path(section_path, el.text)
            # Put heading as its own unit; doesn't count toward sentence boundaries.
            heading_unit = _Unit(kind="heading", text=el.text.strip(), page_number=el.page_number)
            # If the current chunk is non-empty and we're close to the limit, flush first
            # so the heading can start a new section cleanly.
            if current_units and current_words >= max_words * 0.9:
                built = flush_current()
                if built:
                    yield built
            current_units.append(heading_unit)
            continue

        if el.kind == "table":
            # Table is atomic and must not be merged with narrative.
            built = flush_current()
            if built:
                yield built

            raw, serialized = table_to_serialized_text(el.text, html=el.table_html)
            t_text = serialized.strip()
            wc = word_count(t_text)
            yield BuiltChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=t_text,
                word_count=wc,
                sentence_count=0,
                page_start=el.page_number,
                page_end=el.page_number,
                section_path=list(section_path),
                contains_table=True,
                overlap_with_previous=False,
                table_raw=raw,
                table_text=t_text,
            )
            chunk_index += 1

            # After a table, reset overlap state (tables do not participate in overlap)
            current_units = []
            current_words = 0
            overlap_units_next = []
            next_chunk_has_overlap = False
            continue

        # Narrative: split into sentences and add.
        sentences = split_sentences(el.text)
        for sent in sentences:
            wc = word_count(sent)
            # If adding this sentence would exceed max_words, flush current.
            if current_units and (current_words + wc) > max_words:
                built = flush_current()
                if built:
                    yield built
            current_units.append(_Unit(kind="sentence", text=sent, page_number=el.page_number))
            current_words += wc

    # Final flush
    if current_units:
        # For the final chunk, do not seed a "next" chunk; just emit.
        built = _finalize_narrative_chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            units=current_units,
            section_path=section_path,
            overlap_with_previous=next_chunk_has_overlap,
        )
        yield built

