"""
Token-budgeted chunk building. Opt-in with CHUNKING_MODE=token_budget; the word
builder in `chunk_builder.py` stays the default.

Why this module exists
----------------------
`chunk_builder.build_chunks` packs up to 1000 words per chunk, but the embedder
runs at `EMBEDDINGS_MAX_LENGTH` (256) with `truncation=True`
(`services/embeddings.py:121-122`), so a full 1000-word chunk is indexed in
Postgres while its vector only ever sees the first ~256 tokens. Retrieval
therefore cannot reach ~80% of the corpus text, and `chunks.metadata.token_overflow`
(`services/build_metrics.py:79-82`) counts exactly how much is lost.

This builder sizes every chunk with the embedder's own tokenizer instead of a
word count, so `token_overflow` is 0 by construction: nothing reaches the
embedder that the embedder would cut.

What it keeps from the original builder
---------------------------------------
- sentence atomicity (spaCy boundaries; a sentence is never split mid-sentence
  unless it alone exceeds the budget, see `split_oversize_sentences`)
- heading / section hierarchy in `section_path`
- tables never merged with narrative text
- sentence-aligned overlap between consecutive narrative chunks

What it changes
---------------
- the ceiling is tokens, not words, and headings count against it (the original
  appended headings without charging them to `current_words`)
- an atom that alone exceeds the ceiling (a very long sentence, a large
  serialized table) is split into sub-chunks that carry `sub_index` / `sub_count`
  / `sub_of` instead of being handed to the embedder to truncate

Token counting contract
-----------------------
`token_counter` must count WITHOUT special tokens, and `reserve_special_tokens`
covers the [CLS]/[SEP] pair the embedder adds. Per-unit counts are summed rather
than re-tokenizing the joined chunk on every append: wordpiece tokenizes per
whitespace-delimited word, so the sum of the parts equals the count of the
whitespace-joined whole, and the packing loop stays O(units) tokenizer calls.
`services/chunker.py` builds such a counter from the loaded embedder tokenizer
via `EmbeddingsService.get_token_counter(add_special_tokens=False)`.

Every tunable is an explicit keyword argument with a sane default. Override per
call; do not edit callers.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, List, Optional, Sequence

from .chunk_builder import BuiltChunk, _Unit, _update_section_path
from .parser import ParsedElement
from .sentence import split_sentences, word_count
from .table_utils import table_to_serialized_text


# Ceiling defaults. MAX_TOKENS_DEFAULT mirrors EMBEDDINGS_MAX_LENGTH (256); the
# caller in services/chunker.py resolves it from the same env var the embedder
# reads, so raising the embedder window raises the chunk window with it.
MAX_TOKENS_DEFAULT = 256
RESERVE_SPECIAL_TOKENS_DEFAULT = 2   # [CLS] ... [SEP]
OVERLAP_TOKENS_DEFAULT = 64          # 25% of 256, sentence-aligned

# `sub_of` values on split atoms.
SUB_OF_SENTENCE = "sentence"
SUB_OF_HEADING = "heading"
SUB_OF_TABLE = "table"


def split_text_by_tokens(
    text: str,
    *,
    token_counter: Callable[[str], int],
    max_tokens: int,
    join_with: str = " ",
) -> List[str]:
    """
    Split `text` at whitespace boundaries into parts of at most `max_tokens`.

    Used only for atoms that alone exceed the budget, so the per-word tokenizer
    calls stay off the common path.

    A single word whose own token count exceeds `max_tokens` (a run-on string, a
    long table cell) is emitted as its own part and will still truncate at embed
    time: splitting inside a word would put half a wordpiece in each vector.
    That part is the only way `token_overflow` can stay above 0 in this mode.

    Args:
        token_counter: tokens per string, special tokens excluded.
        max_tokens: ceiling per returned part.
        join_with: separator used to rebuild each part.
    """
    words = [w for w in (text or "").split() if w]
    if not words:
        return []

    parts: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for w in words:
        wt = token_counter(w)
        if current and current_tokens + wt > max_tokens:
            parts.append(join_with.join(current))
            current = []
            current_tokens = 0
        current.append(w)
        current_tokens += wt
    if current:
        parts.append(join_with.join(current))
    return parts


def split_serialized_table(
    serialized: str,
    *,
    token_counter: Callable[[str], int],
    max_tokens: int,
    header_lines: int = 2,
    repeat_header: bool = True,
    split_oversize_rows: bool = True,
    join_with: str = "\n",
) -> List[str]:
    """
    Split a serialized table on row boundaries, header repeated on each part.

    `table_utils.table_to_serialized_text` emits GitHub-markdown, so the first
    two lines are the header row and its `|---|` separator: repeating them keeps
    every part readable on its own and keeps the column names in every vector.

    Args:
        header_lines: how many leading lines form the header block (2 for the
            github tablefmt; 0 for a headerless serialization).
        repeat_header: header block on every part, or only on the first.
        split_oversize_rows: a single row wider than the budget is split at
            whitespace boundaries instead of being left to truncate.
        join_with: separator used to rebuild each part.
    """
    lines = [ln for ln in (serialized or "").splitlines()]
    if not lines:
        return []

    header = lines[:header_lines] if header_lines > 0 else []
    body = lines[header_lines:] if header_lines > 0 else list(lines)
    if not body:
        return [join_with.join(lines).strip()] if lines else []

    header_text = join_with.join(header)
    header_tokens = token_counter(header_text) if header else 0
    room = max_tokens - header_tokens
    if room <= 0:
        # Header alone fills the budget: no room to repeat it per part.
        header = []
        header_text = ""
        header_tokens = 0
        room = max_tokens

    parts: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        include_header = bool(header) and (repeat_header or not parts)
        block = ([header_text] if include_header else []) + current
        parts.append(join_with.join(block).strip())
        current = []
        current_tokens = 0

    for line in body:
        if not line.strip():
            continue
        lt = token_counter(line)
        if lt > room:
            flush()
            if split_oversize_rows:
                for frag in split_text_by_tokens(
                    line, token_counter=token_counter, max_tokens=room
                ):
                    current = [frag]
                    current_tokens = token_counter(frag)
                    flush()
            else:
                current = [line]
                current_tokens = lt
                flush()
            continue
        if current and current_tokens + lt > room:
            flush()
        current.append(line)
        current_tokens += lt
    flush()

    return [p for p in parts if p]


def _finalize(
    *,
    document_id: str,
    chunk_index: int,
    units: Sequence[_Unit],
    section_path: Sequence[str],
    overlap_with_previous: bool,
    join_with: str,
    word_counter: Callable[[str], int],
) -> BuiltChunk:
    text = join_with.join([u.text for u in units]).strip()
    pages = [u.page_number for u in units] or [1]
    return BuiltChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        word_count=word_counter(text),
        sentence_count=sum(1 for u in units if u.kind == "sentence"),
        page_start=min(pages),
        page_end=max(pages),
        section_path=list(section_path),
        contains_table=False,
        overlap_with_previous=overlap_with_previous,
    )


def build_chunks_token_budgeted(
    document_id: str,
    elements: Iterable[ParsedElement],
    *,
    token_counter: Callable[[str], int],
    max_tokens: int = MAX_TOKENS_DEFAULT,
    reserve_special_tokens: int = RESERVE_SPECIAL_TOKENS_DEFAULT,
    overlap_tokens: int = OVERLAP_TOKENS_DEFAULT,
    split_oversize_sentences: bool = True,
    split_oversize_headings: bool = True,
    split_tables: bool = True,
    table_header_lines: int = 2,
    repeat_table_header: bool = True,
    split_oversize_table_rows: bool = True,
    table_raw_on_every_part: bool = False,
    heading_flush_ratio: float = 0.9,
    join_with: str = "\n",
    sentence_splitter: Callable[[str], List[str]] = split_sentences,
    word_counter: Callable[[str], int] = word_count,
    section_path_fn: Callable[[List[str], str], List[str]] = _update_section_path,
) -> Iterator[BuiltChunk]:
    """
    Build chunks whose token count never exceeds what the embedder will read.

    Budget per chunk is `max_tokens - reserve_special_tokens`; the embedder then
    adds its [CLS]/[SEP] back and lands exactly on `max_tokens`.

    Args:
        token_counter: tokens per string, special tokens EXCLUDED (see the
            module docstring for why the counts are summed, not recomputed).
        max_tokens: hard ceiling, normally the embedder's max_length.
        reserve_special_tokens: seats kept for the embedder's own special
            tokens. 2 for a BERT-family tokenizer, 0 if the counter already
            includes them.
        overlap_tokens: sentence-aligned tail carried into the next chunk.
            Clamped to budget-1 so a seed can never fill a chunk on its own.
            0 disables overlap.
        split_oversize_sentences: split a single sentence that exceeds the
            budget. False emits it whole, which re-introduces truncation for
            that chunk and is recorded as `token_overflow` by the caller.
        split_oversize_headings: same decision for a heading line.
        split_tables: split a serialized table that exceeds the budget into
            row-boundary parts. False keeps the original atomic-table rule and
            lets the embedder truncate the tail.
        table_header_lines / repeat_table_header / split_oversize_table_rows:
            passed to `split_serialized_table`.
        table_raw_on_every_part: copy `table_raw` (the original HTML) into every
            part's metadata. False stores it on the first part only, so a large
            table does not multiply its raw payload across N rows.
        heading_flush_ratio: flush the open chunk when a heading arrives and the
            chunk already holds this fraction of the budget, so a section starts
            cleanly. 1.0 disables the early flush.
        join_with: separator between units inside a chunk.
        sentence_splitter / word_counter / section_path_fn: injected for tests
            and for callers that segment or count differently.

    Yields:
        BuiltChunk in document order. Split atoms carry `sub_index`, `sub_count`
        and `sub_of`; every other chunk leaves those None.
    """
    budget = max_tokens - reserve_special_tokens
    if budget <= 0:
        raise ValueError(
            f"max_tokens={max_tokens} minus reserve_special_tokens="
            f"{reserve_special_tokens} leaves no room for text"
        )
    effective_overlap = max(0, min(overlap_tokens, budget - 1))

    section_path: List[str] = []
    chunk_index = 0
    current_units: List[_Unit] = []
    current_tokens = 0
    seeded_only = False          # chunk holds nothing but carried-over overlap
    next_has_overlap = False

    def flush() -> Iterator[BuiltChunk]:
        """Emit the open chunk, then seed the next one with its sentence tail."""
        nonlocal chunk_index, current_units, current_tokens, seeded_only, next_has_overlap
        if not current_units:
            return
        built = _finalize(
            document_id=document_id,
            chunk_index=chunk_index,
            units=current_units,
            section_path=section_path,
            overlap_with_previous=next_has_overlap,
            join_with=join_with,
            word_counter=word_counter,
        )
        chunk_index += 1

        seed: List[_Unit] = []
        seed_tokens = 0
        if effective_overlap > 0:
            for u in reversed(current_units):
                if u.kind != "sentence":
                    continue
                ut = token_counter(u.text)
                if seed and seed_tokens + ut > effective_overlap:
                    break
                if ut > effective_overlap and seed:
                    break
                seed.append(u)
                seed_tokens += ut
                if seed_tokens >= effective_overlap:
                    break
            seed = list(reversed(seed))
            if seed_tokens > budget:      # never seed past the ceiling
                seed, seed_tokens = [], 0

        current_units = list(seed)
        current_tokens = seed_tokens
        seeded_only = bool(seed)
        next_has_overlap = bool(seed)
        yield built

    def emit_atom_parts(
        parts: Sequence[str],
        *,
        kind: str,
        page_number: int,
        sub_of: str,
    ) -> Iterator[BuiltChunk]:
        """Emit an oversize atom as numbered sub-chunks."""
        nonlocal chunk_index
        total = len(parts)
        for i, part in enumerate(parts):
            yield BuiltChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=part,
                word_count=word_counter(part),
                sentence_count=1 if kind == "sentence" else 0,
                page_start=page_number,
                page_end=page_number,
                section_path=list(section_path),
                contains_table=False,
                overlap_with_previous=False,
                sub_index=i,
                sub_count=total,
                sub_of=sub_of,
            )
            chunk_index += 1

    def add_unit(unit: _Unit, tokens: int) -> Iterator[BuiltChunk]:
        """Append a within-budget unit, flushing first when it would overflow."""
        nonlocal current_units, current_tokens, seeded_only, next_has_overlap
        if current_units and current_tokens + tokens > budget:
            if seeded_only:
                # The open chunk is nothing but the previous chunk's tail, so
                # flushing it would re-emit that tail as a duplicate chunk (and
                # re-seed the same units forever). Drop the seed instead.
                current_units = []
                current_tokens = 0
                seeded_only = False
                next_has_overlap = False
            else:
                for built in flush():
                    yield built
        current_units.append(unit)
        current_tokens += tokens
        seeded_only = False

    for el in elements:
        if el.kind == "heading":
            section_path = section_path_fn(section_path, el.text)
            head_text = el.text.strip()
            if not head_text:
                continue
            head_tokens = token_counter(head_text)

            if head_tokens > budget:
                for built in flush():
                    yield built
                parts = (
                    split_text_by_tokens(
                        head_text, token_counter=token_counter, max_tokens=budget
                    )
                    if split_oversize_headings
                    else [head_text]
                )
                for built in emit_atom_parts(
                    parts,
                    kind="heading",
                    page_number=el.page_number,
                    sub_of=SUB_OF_HEADING,
                ):
                    yield built
                current_units, current_tokens = [], 0
                seeded_only = False
                next_has_overlap = False
                continue

            if (
                current_units
                and not seeded_only
                and current_tokens >= budget * heading_flush_ratio
            ):
                for built in flush():
                    yield built
            for built in add_unit(
                _Unit(kind="heading", text=head_text, page_number=el.page_number),
                head_tokens,
            ):
                yield built
            continue

        if el.kind == "table":
            for built in flush():
                yield built
            # A table starts clean: it never inherits narrative overlap.
            current_units, current_tokens = [], 0
            seeded_only = False
            next_has_overlap = False

            raw, serialized = table_to_serialized_text(el.text, html=el.table_html)
            t_text = serialized.strip()
            if not t_text:
                continue
            t_tokens = token_counter(t_text)

            if t_tokens <= budget or not split_tables:
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
                continue

            parts = split_serialized_table(
                t_text,
                token_counter=token_counter,
                max_tokens=budget,
                header_lines=table_header_lines,
                repeat_header=repeat_table_header,
                split_oversize_rows=split_oversize_table_rows,
                join_with=join_with,
            )
            total = len(parts)
            for i, part in enumerate(parts):
                yield BuiltChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=part,
                    word_count=word_counter(part),
                    sentence_count=0,
                    page_start=el.page_number,
                    page_end=el.page_number,
                    section_path=list(section_path),
                    contains_table=True,
                    overlap_with_previous=False,
                    table_raw=raw if (table_raw_on_every_part or i == 0) else None,
                    table_text=part,
                    sub_index=i,
                    sub_count=total,
                    sub_of=SUB_OF_TABLE,
                )
                chunk_index += 1
            continue

        for sent in sentence_splitter(el.text):
            s = sent.strip()
            if not s:
                continue
            s_tokens = token_counter(s)

            if s_tokens > budget:
                for built in flush():
                    yield built
                current_units, current_tokens = [], 0
                seeded_only = False
                next_has_overlap = False
                parts = (
                    split_text_by_tokens(
                        s, token_counter=token_counter, max_tokens=budget
                    )
                    if split_oversize_sentences
                    else [s]
                )
                for built in emit_atom_parts(
                    parts,
                    kind="sentence",
                    page_number=el.page_number,
                    sub_of=SUB_OF_SENTENCE,
                ):
                    yield built
                continue

            for built in add_unit(
                _Unit(kind="sentence", text=s, page_number=el.page_number), s_tokens
            ):
                yield built

    if current_units and not seeded_only:
        yield _finalize(
            document_id=document_id,
            chunk_index=chunk_index,
            units=current_units,
            section_path=section_path,
            overlap_with_previous=next_has_overlap,
            join_with=join_with,
            word_counter=word_counter,
        )
