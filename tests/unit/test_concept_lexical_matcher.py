from __future__ import annotations

from app.core.text.concept_lexical import match_concept_lexically, matches_value


def test_exact_mode() -> None:
    assert matches_value("Balo", "balo", "exact")
    assert not matches_value("balo học sinh", "balo", "exact")


def test_prefix_mode() -> None:
    assert matches_value("balo học sinh", "balo", "prefix")
    assert not matches_value("xưởng balo", "balo", "prefix")


def test_token_mode_positive_and_zalo_negative() -> None:
    assert matches_value("balo học sinh cấp 1", "balo", "token")
    assert matches_value("xưởng sản xuất balo theo yêu cầu", "balo", "token")
    assert not matches_value("Zalo 0909983833", "balo", "token")


def test_phrase_mode_material() -> None:
    assert matches_value("vải canvas chống thấm", "vải canvas", "phrase")
    assert matches_value("vải canvas chống thấm", "canvas", "phrase")
    assert not matches_value("vải bố mềm", "vải canvas", "phrase")


def test_accent_sensitive_preserves_diacritics() -> None:
    # Accents matter: "balo" does not equal "ba lô" as token needle.
    assert matches_value("ba lô học sinh", "ba lô", "accent_sensitive")
    assert not matches_value("ba lô học sinh", "balo", "accent_sensitive")
    assert matches_value("balo học sinh", "balo", "token")


def test_positive_and_negative_lexical_match() -> None:
    result = match_concept_lexically(
        text="balo đẹp giá rẻ",
        positive_examples=["balo", "backpack"],
        negative_examples=["balo đẹp", "giá balo"],
        match_mode="phrase",
    )
    assert result.matched
    assert "balo" in result.matched_examples
    assert result.negative_matched
    assert "balo đẹp" in result.negative_matched_examples


def test_semantic_mode_never_matches_lexically() -> None:
    result = match_concept_lexically(
        text="balo học sinh",
        positive_examples=["balo"],
        match_mode="semantic",
    )
    assert not result.matched
    assert result.match_mode == "semantic"
