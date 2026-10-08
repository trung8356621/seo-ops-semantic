from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.text.normalization import normalize_text

# Small generic function-word set only — not an SEO/domain dictionary.
_FUNCTION_WORDS = frozenset(
    {
        # Vietnamese (small set)
        "cho",
        "cua",
        "của",
        "va",
        "và",
        "voi",
        "với",
        "tai",
        "tại",
        "o",
        "ở",
        "cac",
        "các",
        "mot",
        "một",
        "nhung",
        "những",
        "la",
        "là",
        "thi",
        "thì",
        "ma",
        "mà",
        "neu",
        "nếu",
        "de",
        "để",
        "tu",
        "từ",
        "tren",
        "trên",
        "trong",
        "ngoai",
        "ngoài",
        "vao",
        "vào",
        "ra",
        "len",
        "lên",
        "xuong",
        "xuống",
        "bi",
        "bị",
        "duoc",
        "được",
        "nhu",
        "như",
        "hay",
        "hoac",
        "hoặc",
        "này",
        "nay",
        "kia",
        "do",
        "đó",
        "ay",
        "ấy",
        # English generics (not domain/SEO terms)
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "for",
        "with",
        "at",
        "in",
        "on",
        "to",
        "from",
    }
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _is_contiguous_span(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def content_tokens(text: str) -> tuple[str, ...]:
    """Normalize → lowercase → word tokens → drop function words."""
    normalized = normalize_text(text).casefold()
    raw = _TOKEN_RE.findall(normalized)
    return tuple(tok for tok in raw if tok not in _FUNCTION_WORDS and tok)


def informative_ngrams(tokens: tuple[str, ...] | list[str], *, min_n: int = 2) -> frozenset[str]:
    """Contiguous content-token n-grams with length >= min_n."""
    seq = tuple(tokens)
    if len(seq) < min_n:
        return frozenset()
    out: set[str] = set()
    for n in range(min_n, len(seq) + 1):
        for i in range(0, len(seq) - n + 1):
            out.add(" ".join(seq[i : i + n]))
    return frozenset(out)


@dataclass(frozen=True, slots=True)
class LexicalPairEvidence:
    containment: float
    shared_ngrams: frozenset[str]
    exclusive_ngrams_a: frozenset[str]
    exclusive_ngrams_b: frozenset[str]
    compatible: bool
    conflict: bool


def pair_evidence(
    text_a: str,
    text_b: str,
    *,
    containment_min: float = 0.67,
    frequent_ngrams: frozenset[str] | None = None,
) -> LexicalPairEvidence:
    """Deterministic lexical pair evidence (no domain phrase hard-coding)."""
    tokens_a = content_tokens(text_a)
    tokens_b = content_tokens(text_b)
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return LexicalPairEvidence(
            containment=0.0,
            shared_ngrams=frozenset(),
            exclusive_ngrams_a=frozenset(),
            exclusive_ngrams_b=frozenset(),
            compatible=False,
            conflict=True,
        )

    inter = set_a & set_b
    containment = len(inter) / float(min(len(set_a), len(set_b)))
    ngrams_a = informative_ngrams(tokens_a)
    ngrams_b = informative_ngrams(tokens_b)
    shared = ngrams_a & ngrams_b
    exclusive_a = ngrams_a - ngrams_b
    exclusive_b = ngrams_b - ngrams_a

    # Corpus frequency must not change this label. The argument stays for callers.
    _ = frequent_ngrams
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    embedded = _is_contiguous_span(longer, shorter)
    suffix_len = 0
    for left, right in zip(reversed(tokens_a), reversed(tokens_b)):
        if left != right:
            break
        suffix_len += 1
    head_swap = suffix_len >= 2 and len(tokens_a) - suffix_len == 1 and len(tokens_b) - suffix_len == 1
    prefix_len = 0
    for left, right in zip(tokens_a, tokens_b):
        if left != right:
            break
        prefix_len += 1
    tail_a = set(tokens_a[prefix_len:])
    tail_b = set(tokens_b[prefix_len:])
    short_substitution = (
        prefix_len >= 2
        and 1 <= len(tail_a) <= 3
        and 1 <= len(tail_b) <= 3
        and not (tail_a & tail_b)
    )
    near_duplicate = abs(len(tokens_a) - len(tokens_b)) <= 2 and containment >= containment_min
    if short_substitution:
        compatible = False
        conflict = True
    elif embedded or head_swap or near_duplicate:
        compatible = True
        conflict = False
    elif exclusive_a and exclusive_b and not shared:
        compatible = False
        conflict = True
    else:
        compatible = False
        conflict = False

    return LexicalPairEvidence(
        containment=float(containment),
        shared_ngrams=frozenset(shared),
        exclusive_ngrams_a=frozenset(exclusive_a),
        exclusive_ngrams_b=frozenset(exclusive_b),
        compatible=compatible,
        conflict=conflict,
    )
