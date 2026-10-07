from __future__ import annotations

import pytest

from app.core.text.normalization import normalize_text


def test_vietnamese_accents_survive() -> None:
    assert normalize_text("  balo học sinh  ") == "balo học sinh"
    assert "ọ" in normalize_text("học")
    assert normalize_text("Balo Anh Văn") == "Balo Anh Văn"


def test_whitespace_collapsed() -> None:
    assert normalize_text("a\t\nb   c") == "a b c"


def test_unicode_nfc() -> None:
    composed = normalize_text("e\u0301")
    assert composed == "é"


def test_none_rejected() -> None:
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]
