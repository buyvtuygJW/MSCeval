"""
Structure-aware chunking (strategy 3 of methodnow-rescoped.md:88-92).

Boundaries follow the element categories the parser emits (heading, narrative,
table). An element is atomic: consecutive whole elements pack into one chunk up
to the token budget, a chunk never starts mid-element, and a table is always
one chunk on its own. Same ParsedElement in, same BuiltChunk out as the two app
builders, so the winner pastes into rag-service/chunking/ unchanged.
"""

from __future__ import annotations

from typing import Callable, Iterator, List

from .chunk_builder import BuiltChunk, _update_section_path
from .parser import ParsedElement
from .sentence import split_sentences, word_count
from .table_utils import table_to_serialized_text


def build_chunks_structure_aware(
    document_id: str,
    elements: Iterator[ParsedElement],
    *,
    token_counter: Callable[[str], int],
    max_tokens: int = 256,
    reserve_special_tokens: int = 2,
    keep_tables_intact: bool = True,
    split_oversize_narrative: bool = True,
    heading_flush_ratio: float = 0.9,
    join_with: str = "\n",
    sentence_splitter: Callable[[str], List[str]] = split_sentences,
    word_counter: Callable[[str], int] = word_count,
    section_path_fn: Callable[[List[str], str], List[str]] = _update_section_path,
) -> Iterator[BuiltChunk]:
    """
    Args:
        token_counter: tokens per string, special tokens EXCLUDED.
        max_tokens: hard ceiling, normally the embedder's max_length.
        reserve_special_tokens: seats for [CLS]/[SEP]; 2 for BERT-family.
        keep_tables_intact: a table is one chunk even past the budget (the
            embedder then truncates it; that cost is part of the comparison).
            False splits it sentence-wise like an oversize narrative element.
        split_oversize_narrative: one narrative element alone over budget is
            cut on sentence boundaries into parts carrying sub_index/sub_count
            (sub_of="sentence"). False emits it whole, like a table.
        heading_flush_ratio: a heading arriving when the chunk is past this
            fraction of the budget flushes first, so the heading starts its
            section's chunk instead of dangling at a tail.
    """
    budget = max_tokens - reserve_special_tokens
    if budget <= 0:
        raise ValueError("max_tokens must exceed reserve_special_tokens")

    section_path: List[str] = []
    chunk_index = 0
    cur_texts: List[str] = []
    cur_tokens = 0
    cur_pages: List[int] = []
    cur_sentences = 0
    cur_path: List[str] = []

    def flush() -> Iterator[BuiltChunk]:
        nonlocal chunk_index, cur_texts, cur_tokens, cur_pages, cur_sentences
        if not cur_texts:
            return
        text = join_with.join(cur_texts).strip()
        yield BuiltChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            word_count=word_counter(text),
            sentence_count=cur_sentences,
            page_start=min(cur_pages),
            page_end=max(cur_pages),
            section_path=list(cur_path),
            contains_table=False,
            overlap_with_previous=False,  # element-bounded: no overlap by design
        )
        chunk_index += 1
        cur_texts, cur_tokens, cur_pages, cur_sentences = [], 0, [], 0

    def emit_parts(el: ParsedElement) -> Iterator[BuiltChunk]:
        """Sentence-wise cut of one oversize narrative element."""
        nonlocal chunk_index
        parts: List[List[str]] = []
        part: List[str] = []
        part_tokens = 0
        for sent in sentence_splitter(el.text):
            t = token_counter(sent)
            if part and part_tokens + t > budget:
                parts.append(part)
                part, part_tokens = [], 0
            part.append(sent)
            part_tokens += t
        if part:
            parts.append(part)
        for i, p in enumerate(parts):
            text = join_with.join(p).strip()
            yield BuiltChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                word_count=word_counter(text),
                sentence_count=len(p),
                page_start=el.page_number,
                page_end=el.page_number,
                section_path=list(cur_path),
                contains_table=False,
                overlap_with_previous=False,
                sub_index=i,
                sub_count=len(parts),
                sub_of="sentence",
            )
            chunk_index += 1

    for el in elements:
        if el.kind == "heading":
            section_path = section_path_fn(section_path, el.text)
            if cur_texts and cur_tokens >= budget * heading_flush_ratio:
                yield from flush()
            if not cur_texts:
                cur_path = list(section_path)
            heading_tokens = token_counter(el.text.strip())
            if cur_texts and cur_tokens + heading_tokens > budget:
                yield from flush()
                cur_path = list(section_path)
            cur_texts.append(el.text.strip())
            cur_tokens += heading_tokens
            cur_pages.append(el.page_number)
            continue

        if el.kind == "table":
            yield from flush()
            cur_path = list(section_path)
            raw, serialized = table_to_serialized_text(el.text, html=el.table_html)
            t_text = serialized.strip()
            if keep_tables_intact or token_counter(t_text) <= budget:
                yield BuiltChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=t_text,
                    word_count=word_counter(t_text),
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
            else:
                yield from emit_parts(
                    ParsedElement(kind="narrative", text=t_text, page_number=el.page_number)
                )
            continue

        # Narrative element: atomic unless alone it exceeds the budget.
        el_text = el.text.strip()
        el_tokens = token_counter(el_text)
        if el_tokens > budget:
            yield from flush()
            if split_oversize_narrative:
                cur_path = list(section_path)
                yield from emit_parts(el)
            else:
                yield BuiltChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=el_text,
                    word_count=word_counter(el_text),
                    sentence_count=len(sentence_splitter(el_text)),
                    page_start=el.page_number,
                    page_end=el.page_number,
                    section_path=list(section_path),
                    contains_table=False,
                    overlap_with_previous=False,
                )
                chunk_index += 1
            continue
        if cur_texts and cur_tokens + el_tokens > budget:
            yield from flush()
        if not cur_texts:
            cur_path = list(section_path)
        cur_texts.append(el_text)
        cur_tokens += el_tokens
        cur_pages.append(el.page_number)
        cur_sentences += len(sentence_splitter(el_text))

    yield from flush()
