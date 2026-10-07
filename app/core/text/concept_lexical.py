"""Generic deterministic concept lexical matching.

Modes mirror common Match-rule intent (exact / prefix / token / phrase /
accent_sensitive). No SEO dictionaries, Industry terms, or CTA phrases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from app.core.text.normalization import casefold_preserve_accents, fold_accents, normalize_text

MatchMode = Literal["exact", "prefix", "token", "phrase", "accent_sensitive", "semantic"]

_DETERMINISTIC_MODES: frozenset[str] = frozenset(
    {"exact", "prefix", "token", "phrase", "accent_sensitive"}
)


def is_deterministic_mode(mode: str) -> bool:
    return mode in _DETERMINISTIC_MODES


def _collapse_ws(value: str) -> str:
    return normalize_text(value)


def _is_boundary_char(ch: str) -> bool:
    """True for letter / number / underscore (Unicode-aware, like PHP \\p{L}\\p{N}_)."""
    return ch == "_" or ch.isalnum()


def _token_boundary_match(haystack: str, needle: str) -> bool:
    """Whole-needle match with Unicode letter/number/underscore boundaries."""
    needle = needle.strip()
    if needle == "":
        return False
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return False
        before_ok = idx == 0 or not _is_boundary_char(haystack[idx - 1])
        end = idx + len(needle)
        after_ok = end >= len(haystack) or not _is_boundary_char(haystack[end])
        if before_ok and after_ok:
            return True
        start = idx + 1


def matches_value(haystack: str, needle: str, mode: MatchMode) -> bool:
    """Return True when ``needle`` matches ``haystack`` under ``mode``.

    Semantics (intentionally documented):

    - ``exact``: full-string equality after casefold (accents preserved)
    - ``prefix``: haystack starts with needle (casefold, accents preserved)
    - ``token`` / ``accent_sensitive``: needle appears as a bounded token/phrase
      (casefold, accents preserved — accents matter)
    - ``phrase``: whitespace-normalized contiguous substring with padding
      (casefold, accents preserved)
    - ``semantic``: never matches lexically (embeddings own this path)

    Unknown modes fall back to accent-folded phrase containment.
    """
    if mode == "semantic":
        return False

    if mode in {"exact", "prefix", "token", "phrase", "accent_sensitive"}:
        h = casefold_preserve_accents(haystack)
        n = casefold_preserve_accents(needle)
    else:
        h = fold_accents(haystack)
        n = fold_accents(needle)

    h = _collapse_ws(h)
    n = _collapse_ws(n)
    if n == "":
        return False

    if mode == "exact":
        return h == n
    if mode == "prefix":
        return h.startswith(n)
    if mode in {"token", "accent_sensitive"}:
        return _token_boundary_match(h, n)
    if mode == "phrase":
        return f" {h} ".find(f" {n} ") >= 0

    # Fallback: folded phrase
    return f" {h} ".find(f" {n} ") >= 0


@dataclass(frozen=True, slots=True)
class ConceptLexicalMatch:
    matched: bool
    match_mode: str
    matched_examples: tuple[str, ...]
    best_match: str | None
    negative_matched: bool
    negative_matched_examples: tuple[str, ...]


def match_concept_lexically(
    *,
    text: str,
    positive_examples: Sequence[str],
    negative_examples: Sequence[str] = (),
    match_mode: MatchMode = "phrase",
) -> ConceptLexicalMatch:
    """Evaluate deterministic lexical evidence for one entity × concept."""
    if not is_deterministic_mode(match_mode):
        return ConceptLexicalMatch(
            matched=False,
            match_mode=match_mode,
            matched_examples=(),
            best_match=None,
            negative_matched=False,
            negative_matched_examples=(),
        )

    positives: list[str] = []
    for example in positive_examples:
        if matches_value(text, example, match_mode):
            positives.append(example)

    negatives: list[str] = []
    for example in negative_examples:
        if matches_value(text, example, match_mode):
            negatives.append(example)

    return ConceptLexicalMatch(
        matched=len(positives) > 0,
        match_mode=match_mode,
        matched_examples=tuple(positives),
        best_match=positives[0] if positives else None,
        negative_matched=len(negatives) > 0,
        negative_matched_examples=tuple(negatives),
    )
