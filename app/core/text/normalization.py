from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


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
