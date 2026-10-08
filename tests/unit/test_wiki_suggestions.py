from __future__ import annotations

from app.modules.wiki_suggestions.contracts import (
    CanonicalConceptIn,
    WikiSuggestionPolicyIn,
    WikiSuggestionRequest,
)
from app.modules.wiki_suggestions.suggester import suggest_wiki_links


def _catalog() -> list[CanonicalConceptIn]:
    return [
        CanonicalConceptIn(
            ref="wiki:roi",
            label="ROI",
            url="https://en.wikipedia.org/wiki/Return_on_investment",
            aliases=["Return on investment"],
            semantic_score=0.91,
        ),
        CanonicalConceptIn(
            ref="wiki:polyester",
            label="Polyester",
            url="https://en.wikipedia.org/wiki/Polyester",
            aliases=["polyester"],
            semantic_score=0.84,
        ),
        CanonicalConceptIn(
            ref="wiki:rfid",
            label="RFID",
            url="https://en.wikipedia.org/wiki/Radio-frequency_identification",
            semantic_score=0.88,
        ),
        CanonicalConceptIn(
            ref="wiki:oem",
            label="OEM",
            url="https://en.wikipedia.org/wiki/Original_equipment_manufacturer",
            semantic_score=0.8,
        ),
        CanonicalConceptIn(
            ref="wiki:moq",
            label="MOQ",
            url="https://en.wikipedia.org/wiki/Minimum_order_quantity",
            semantic_score=0.2,
        ),
        CanonicalConceptIn(
            ref="wiki:generic",
            label="khách hàng",
            url="https://en.wikipedia.org/wiki/Customer",
            semantic_score=0.99,
        ),
    ]


def test_wiki_accepts_technical_terms_rejects_generic_and_keeps_quota() -> None:
    result = suggest_wiki_links(
        WikiSuggestionRequest(
            article_ref="article:1",
            content="Nhà máy dùng RFID và Polyester cho OEM. ROI và MOQ cần giải thích cho khách hàng.",
            catalog=_catalog(),
            policy=WikiSuggestionPolicyIn(max_suggestions=2, min_semantic_score=0.5),
        )
    )
    assert len(result.suggestions) == 2
    assert {item.ref for item in result.suggestions} <= {"wiki:roi", "wiki:polyester", "wiki:rfid", "wiki:oem"}
    assert "wiki:moq" not in {item.ref for item in result.suggestions}
    assert "wiki:generic" not in {item.ref for item in result.suggestions}
    assert any("khách hàng" in term or term == "khach hang" for term in result.rejected_terms)


def test_unknown_acronym_is_not_invented() -> None:
    result = suggest_wiki_links(
        WikiSuggestionRequest(
            article_ref="article:1",
            content="Thiết bị dùng XYZ và ROI.",
            catalog=_catalog(),
            policy=WikiSuggestionPolicyIn(max_suggestions=2, min_semantic_score=0.5),
        )
    )
    assert "XYZ" in result.rejected_terms
    assert any(item.ref == "wiki:roi" for item in result.suggestions)


def test_verified_cache_supplies_a_real_wikipedia_url() -> None:
    result = suggest_wiki_links(
        WikiSuggestionRequest(
            article_ref="article:9",
            content="Dây chuyền dùng RFID cho kho.",
        )
    )
    assert result.suggestions
    assert result.suggestions[0].url == "https://en.wikipedia.org/wiki/Radio-frequency_identification"
    assert len(result.suggestions) <= 2


def test_ambiguous_wikipedia_lookup_is_rejected() -> None:
    def lookup(term: str, language: str) -> str | None:
        del language
        if term == "CAD":
            return None
        return "https://en.wikipedia.org/wiki/Polyester"

    result = suggest_wiki_links(
        WikiSuggestionRequest(
            article_ref="article:9",
            content="CAD và khách hàng.",
            policy=WikiSuggestionPolicyIn(max_suggestions=2, lookup="wikipedia"),
            catalog=[],
        ),
        lookup,
    )
    assert "CAD" in result.rejected_terms
    assert all(item.term != "CAD" for item in result.suggestions)
    assert all("khách hàng" not in item.term.casefold() for item in result.suggestions)
