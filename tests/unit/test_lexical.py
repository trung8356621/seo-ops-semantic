from __future__ import annotations

from app.core.text.lexical import content_tokens, informative_ngrams, pair_evidence


def test_content_tokens_drop_function_words() -> None:
    assert content_tokens("balo cho học sinh") == ("balo", "học", "sinh")
    assert content_tokens("balo cho sinh viên") == ("balo", "sinh", "viên")


def test_informative_ngrams_ignore_function_word_bigrams() -> None:
    # "balo cho" must not become an informative anchor.
    tokens = content_tokens("balo cho học sinh")
    ngrams = informative_ngrams(tokens)
    assert "học sinh" in ngrams
    assert "balo cho" not in ngrams


def test_pair_compatible_same_modifier() -> None:
    ev = pair_evidence("balo học sinh", "balo cho học sinh")
    assert ev.compatible
    assert not ev.conflict
    assert "học sinh" in ev.shared_ngrams


def test_pair_compatible_shared_modifier_different_product() -> None:
    ev = pair_evidence("cặp học sinh", "balo học sinh")
    assert ev.compatible
    assert "học sinh" in ev.shared_ngrams


def test_pair_conflict_distinct_modifiers() -> None:
    ev = pair_evidence("balo học sinh", "balo sinh viên")
    assert not ev.compatible
    assert ev.conflict
    assert not ev.shared_ngrams


def test_pair_conflict_despite_shared_function_word_pattern() -> None:
    ev = pair_evidence("balo cho học sinh", "balo cho sinh viên")
    assert not ev.compatible
    assert ev.conflict
