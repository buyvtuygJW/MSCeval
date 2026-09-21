"""
Sentence segmentation utilities.

Hard rules:
- Sentences must never be split.
- Use spaCy sentence boundaries only (en_core_web_sm).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence


@lru_cache(maxsize=1)
def _nlp():
    import spacy  # type: ignore

    # Keep parser so sentence boundaries match the model defaults.
    # Disable heavy components we don't need.
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy boundaries (never split sentences)."""
    if not text or not text.strip():
        return []

    doc = _nlp()(text)
    sents: List[str] = []
    for sent in doc.sents:
        s = sent.text.strip()
        if s:
            sents.append(s)
    return sents


def contains_verb(text: str, *, verb_pos: Sequence[str] = ("VERB", "AUX")) -> bool:
    """
    True when spaCy tags at least one token with one of `verb_pos`.

    Used by the heading test in chunking/text_partition.py: a short block with a
    verb is a sentence, a short block without one is a heading candidate. The
    tagger is already loaded for sentence boundaries, so this adds no model.

    Args:
        verb_pos: coarse tags that count as a verb; ("VERB",) alone drops
            copulas such as "is" and "are".
    """
    if not text or not text.strip():
        return False

    wanted = set(verb_pos)
    return any(token.pos_ in wanted for token in _nlp()(text))


def word_count(text: str) -> int:
    if not text:
        return 0
    # Simple whitespace split is fine for word counts; not token-based.
    return len([w for w in text.strip().split() if w])

