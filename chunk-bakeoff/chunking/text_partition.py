"""
Deterministic partitioning of already-extracted page text.

Replaces `unstructured.partition.text.partition_text`, which this service used
only to turn one page of extracted text into paragraphs plus a title/not-title
flag (chunking/parser.py). The rules below are the ones that library applies to
plain text - paragraph grouping for hard-wrapped lines, capitalisation ratio,
alphabetic ratio, sentence count, verb presence - rewritten on the standard
library plus the spaCy model this service already loads for sentence splitting.
Chunk boundaries therefore stop depending on an unpinned dependency and on NLTK
data fetched at first call.

Every threshold is an explicit argument with the shipped value as its default.
The orchestrator takes the splitter and the classifier as callables, so a caller
retunes any rule with `functools.partial` instead of editing this module.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from .sentence import contains_verb as _spacy_contains_verb
from .sentence import split_sentences

# --- splitting -------------------------------------------------------------

# A run of two or more newlines: the paragraph break in text that already has
# blank lines between paragraphs.
PARAGRAPH_BREAK_PATTERN = r"(?:[ \t]*\r?\n[ \t]*){2,}"
# Any single newline: the paragraph break in text that is one paragraph per line.
LINE_BREAK_PATTERN = r"[ \t]*\r?\n[ \t]*"
# A newline inside a paragraph of hard-wrapped text, to be replaced by a space.
HARD_WRAP_PATTERN = r"(?<=\S)[ \t]*\r?\n[ \t]*(?=\S)"
# Leading bullet or enumerator: "- ", "* ", "1. ", "(a) ", and unicode bullets.
BULLET_PATTERN = (
    r"^\s*(?:[\*\-·•‣⁃∙▪●◦▫]"
    r"|\(?[0-9]{1,2}[\.\)]|\(?[a-zA-Z][\.\)])\s+"
)

SPLIT_MODES = ("auto", "hard_wrapped", "one_per_line", "blank_line")


def blank_line_ratio(text: str, *, max_lines_sampled: int = 2000) -> float:
    """Fraction of blank lines in the first `max_lines_sampled` lines. 0.0 when empty."""
    lines = (text or "").splitlines()[:max_lines_sampled]
    if not lines:
        return 0.0
    blank = sum(1 for ln in lines if not ln.strip())
    return blank / len(lines)


def is_bulleted_block(
    text: str,
    *,
    bullet_pattern: str = BULLET_PATTERN,
    min_bulleted_lines: int = 1,
) -> bool:
    """True when the block carries list markers and its lines must stay separate."""
    rx = re.compile(bullet_pattern)
    hits = sum(1 for ln in (text or "").splitlines() if ln.strip() and rx.match(ln))
    return hits >= min_bulleted_lines


def split_into_blocks(
    text: str,
    *,
    mode: str = "auto",
    blank_line_ratio_threshold: float = 0.1,
    max_lines_sampled: int = 2000,
    paragraph_break_pattern: str = PARAGRAPH_BREAK_PATTERN,
    line_break_pattern: str = LINE_BREAK_PATTERN,
    hard_wrap_pattern: str = HARD_WRAP_PATTERN,
    hard_wrap_replacement: str = " ",
    bullet_pattern: str = BULLET_PATTERN,
    split_bulleted_lines: bool = True,
    strip_blocks: bool = True,
    drop_empty_blocks: bool = True,
) -> List[str]:
    """
    Cut one page of text into candidate blocks.

    Args:
        mode: `auto` picks between the two layouts by counting blank lines;
            `hard_wrapped` always rejoins wrapped lines inside a paragraph;
            `one_per_line` treats every newline as a break;
            `blank_line` splits on blank lines and keeps newlines inside a block.
        blank_line_ratio_threshold: in `auto`, text with fewer blank lines than
            this fraction is treated as hard-wrapped.
        max_lines_sampled: cap on the lines inspected for that ratio.
        paragraph_break_pattern / line_break_pattern: the two split regexes.
        hard_wrap_pattern / hard_wrap_replacement: newline inside a paragraph
            and what replaces it.
        bullet_pattern / split_bulleted_lines: keep list items as one block each
            instead of gluing them into a paragraph.
        strip_blocks / drop_empty_blocks: whitespace and empty-block policy.

    Returns:
        Blocks in page order.
    """
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES}")
    if not text or not text.strip():
        return []

    resolved = mode
    if mode == "auto":
        ratio = blank_line_ratio(text, max_lines_sampled=max_lines_sampled)
        resolved = "hard_wrapped" if ratio < blank_line_ratio_threshold else "blank_line"

    if resolved == "one_per_line":
        raw = re.split(line_break_pattern, text)
    else:
        raw = re.split(paragraph_break_pattern, text)

    out: List[str] = []
    for block in raw:
        if resolved == "hard_wrapped":
            if split_bulleted_lines and is_bulleted_block(block, bullet_pattern=bullet_pattern):
                pieces = re.split(line_break_pattern, block)
            else:
                pieces = [re.sub(hard_wrap_pattern, hard_wrap_replacement, block)]
        elif resolved == "blank_line" and split_bulleted_lines and is_bulleted_block(
            block, bullet_pattern=bullet_pattern
        ):
            pieces = re.split(line_break_pattern, block)
        else:
            pieces = [block]

        for piece in pieces:
            value = piece.strip() if strip_blocks else piece
            if drop_empty_blocks and not value.strip():
                continue
            out.append(value)
    return out


# --- classification --------------------------------------------------------

TITLE_TRAILING_PUNCTUATION = ".,:;"
HEADING_KIND = "heading"
NARRATIVE_KIND = "narrative"


def capitalised_word_ratio(text: str) -> float:
    """Fraction of words that are Title Case or ALL CAPS. 0.0 when there are no words."""
    words = [w for w in (text or "").split() if w]
    if not words:
        return 0.0
    capped = sum(1 for w in words if w.istitle() or w.isupper())
    return capped / len(words)


def alpha_ratio(text: str) -> float:
    """Fraction of non-space characters that are letters. 0.0 when there are none."""
    chars = [c for c in (text or "") if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isalpha()) / len(chars)


def count_sentences(
    text: str,
    *,
    min_words: int = 5,
    splitter: Optional[Callable[[str], Sequence[str]]] = None,
) -> int:
    """Sentences holding at least `min_words` words, spaCy boundaries by default."""
    split = splitter or split_sentences
    return sum(1 for s in split(text) if len([w for w in s.split() if w]) >= min_words)


def looks_like_narrative(
    text: str,
    *,
    cap_ratio_threshold: float = 0.5,
    min_alpha_ratio: float = 0.5,
    min_sentences: int = 2,
    sentence_min_words: int = 5,
    reject_all_caps: bool = True,
    require_verb_when_short: bool = True,
    sentence_counter: Optional[Callable[[str], int]] = None,
    verb_detector: Optional[Callable[[str], bool]] = None,
) -> bool:
    """
    Prose test: body text under a heading rather than the heading itself.

    Args:
        cap_ratio_threshold: above this share of capitalised words it is not prose.
        min_alpha_ratio: below this share of letters it is not prose (page
            furniture, numeric runs, dotted leaders).
        min_sentences / sentence_min_words: how many real sentences clear the
            test on length alone.
        reject_all_caps: an ALL CAPS block is never prose.
        require_verb_when_short: a single-sentence block still counts as prose
            when it contains a verb; set False to drop the spaCy tag lookup.
        sentence_counter / verb_detector: injectable, default to spaCy.
    """
    body = (text or "").strip()
    if not body:
        return False
    if reject_all_caps and body.isupper():
        return False
    if capitalised_word_ratio(body) > cap_ratio_threshold:
        return False
    if alpha_ratio(body) < min_alpha_ratio:
        return False

    counter = sentence_counter or (lambda t: count_sentences(t, min_words=sentence_min_words))
    if counter(body) >= min_sentences:
        return True
    if not require_verb_when_short:
        return False
    detect = verb_detector or _spacy_contains_verb
    return detect(body)


def looks_like_title(
    text: str,
    *,
    max_words: int = 12,
    max_sentences: int = 1,
    sentence_min_words: int = 5,
    trailing_punctuation: str = TITLE_TRAILING_PUNCTUATION,
    reject_numeric: bool = True,
    sentence_counter: Optional[Callable[[str], int]] = None,
) -> bool:
    """
    Heading test, applied only to blocks the prose test already rejected.

    Args:
        max_words: longer blocks are body text, not a heading.
        max_sentences / sentence_min_words: a heading is at most one sentence.
        trailing_punctuation: a block ending in one of these reads as a sentence.
        reject_numeric: a bare page number is not a heading.
        sentence_counter: injectable, defaults to spaCy.
    """
    body = (text or "").strip()
    if not body:
        return False
    if len([w for w in body.split() if w]) > max_words:
        return False
    if trailing_punctuation and body[-1] in trailing_punctuation:
        return False
    if reject_numeric and body.replace(".", "").replace(",", "").strip().isdigit():
        return False
    counter = sentence_counter or (lambda t: count_sentences(t, min_words=sentence_min_words))
    return counter(body) <= max_sentences


def classify_block(
    text: str,
    *,
    heading_kind: str = HEADING_KIND,
    narrative_kind: str = NARRATIVE_KIND,
    bullet_pattern: str = BULLET_PATTERN,
    bullets_are_narrative: bool = True,
    narrative_test: Optional[Callable[[str], bool]] = None,
    title_test: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Label one block.

    Order matches what the service shipped: list items and prose are narrative,
    a short caps-heavy leftover is a heading, anything else stays narrative.
    Only headings feed `section_path` in chunk_builder, so an unsure block is
    cheaper as narrative than as a fake section.
    """
    body = (text or "").strip()
    if not body:
        return narrative_kind
    if bullets_are_narrative and is_bulleted_block(body, bullet_pattern=bullet_pattern):
        return narrative_kind
    if (narrative_test or looks_like_narrative)(body):
        return narrative_kind
    if (title_test or looks_like_title)(body):
        return heading_kind
    return narrative_kind


# --- orchestration ---------------------------------------------------------


def partition_page_text(
    text: str,
    *,
    splitter: Callable[[str], Sequence[str]] = split_into_blocks,
    classifier: Callable[[str], str] = classify_block,
    strip_blocks: bool = True,
    drop_empty_blocks: bool = True,
) -> List[Tuple[str, str]]:
    """
    One page of extracted text -> `[(kind, block_text), ...]` in page order.

    Args:
        splitter: text -> blocks; pass a partial of `split_into_blocks` to
            retune the layout rules.
        classifier: block -> kind; pass a partial of `classify_block` to retune
            the heading rules.
        strip_blocks / drop_empty_blocks: applied again after the classifier so
            a custom splitter cannot leak blank elements into the chunker.
    """
    out: List[Tuple[str, str]] = []
    for block in splitter(text or ""):
        value = block.strip() if strip_blocks else block
        if drop_empty_blocks and not value.strip():
            continue
        out.append((classifier(value), value))
    return out
