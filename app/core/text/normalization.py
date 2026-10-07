from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_COMBINING_RE = re.compile(r"[\u0300-\u036f]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Conservative embedding prep.

    - Unicode NFC
    - trim
    - collapse repeated whitespace

    Does NOT strip Vietnamese accents, lowercase, or apply SEO phrase identity rules.
    """
    if text is None:
        raise TypeError("text must be a string")

    value = unicodedata.normalize("NFC", str(text))
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def casefold_preserve_accents(text: str) -> str:
    """Lowercase / casefold while keeping diacritics (NFC + whitespace normalize)."""
    return normalize_text(text).casefold()


def fold_accents(text: str) -> str:
    """Accent-insensitive fold for generic matching (not SEO dictionaries).

    NFC → Vietnamese đ/Đ → d → NFD → strip combining marks → casefold → whitespace.
    """
    value = normalize_text(text)
    value = value.replace("đ", "d").replace("Đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = _COMBINING_RE.sub("", value)
    return value.casefold()
