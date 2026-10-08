"""Verified Wikipedia targets. URLs are canonical pages, not guessed paths."""

from __future__ import annotations

from app.modules.wiki_suggestions.contracts import CanonicalConceptIn

VERIFIED_CONCEPTS: list[CanonicalConceptIn] = [
    CanonicalConceptIn(
        ref="wiki:roi",
        label="ROI",
        url="https://en.wikipedia.org/wiki/Return_on_investment",
        aliases=["Return on investment"],
    ),
    CanonicalConceptIn(
        ref="wiki:rfid",
        label="RFID",
        url="https://en.wikipedia.org/wiki/Radio-frequency_identification",
        aliases=["Radio-frequency identification"],
    ),
    CanonicalConceptIn(
        ref="wiki:polyester",
        label="Polyester",
        url="https://en.wikipedia.org/wiki/Polyester",
        aliases=["polyester"],
    ),
    CanonicalConceptIn(
        ref="wiki:oem",
        label="OEM",
        url="https://en.wikipedia.org/wiki/Original_equipment_manufacturer",
        aliases=["Original equipment manufacturer"],
    ),
    CanonicalConceptIn(
        ref="wiki:moq",
        label="MOQ",
        url="https://en.wikipedia.org/wiki/Minimum_order_quantity",
        aliases=["minimum order quantity"],
    ),
]


def verified_for_language(language: str | None) -> list[CanonicalConceptIn]:
    del language
    return list(VERIFIED_CONCEPTS)
