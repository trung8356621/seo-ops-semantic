from __future__ import annotations

from app.modules.wiki_suggestions.wikipedia_lookup import wikipedia_opensearch


def test_exact_title_is_accepted_when_related_hits_exist() -> None:
    def fetch(url: str) -> object:
        if "opensearch" in url:
            return [
                "USB",
                ["USB", "USB-C"],
                ["", ""],
                ["https://en.wikipedia.org/wiki/USB", "https://en.wikipedia.org/wiki/USB-C"],
            ]
        return {"query": {"pages": {"1": {"title": "USB"}}}}

    assert wikipedia_opensearch("USB", "en", fetch) == "https://en.wikipedia.org/wiki/USB"


def test_redirect_and_non_exact_titles_are_rejected() -> None:
    def fetch(url: str) -> object:
        if "opensearch" in url and "CAD" in url:
            return [
                "CAD",
                ["CAD", "Cadbury"],
                ["", ""],
                ["https://en.wikipedia.org/wiki/CAD", "https://en.wikipedia.org/wiki/Cadbury"],
            ]
        if "titles=CAD" in url:
            return {"query": {"pages": {"1": {"title": "CAD", "redirect": ""}}}}
        return [
            "XYZ",
            ["Something else"],
            [""],
            ["https://en.wikipedia.org/wiki/Something_else"],
        ]

    assert wikipedia_opensearch("CAD", "en", fetch) is None
    assert wikipedia_opensearch("XYZ", "en", fetch) is None
