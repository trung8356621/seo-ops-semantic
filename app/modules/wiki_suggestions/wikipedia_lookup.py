"""Single-term Wikipedia opensearch lookup. Does not crawl or embed pages."""

from __future__ import annotations

from collections.abc import Callable

import httpx

Lookup = Callable[[str, str], str | None]


def wikipedia_opensearch(term: str, language: str, transport: httpx.Client | None = None) -> str | None:
    lang = "vi" if language.lower().startswith("vi") else "en"
    params = {
        "action": "opensearch",
        "search": term,
        "limit": "3",
        "namespace": "0",
        "format": "json",
    }
    url = f"https://{lang}.wikipedia.org/w/api.php"
    client = transport or httpx.Client(timeout=5.0)
    close = transport is None
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if close:
            client.close()

    if not isinstance(payload, list) or len(payload) < 4:
        return None
    titles = payload[1] if isinstance(payload[1], list) else []
    descriptions = payload[2] if isinstance(payload[2], list) else []
    urls = payload[3] if isinstance(payload[3], list) else []
    if len(urls) != 1 or len(titles) != 1:
        return None
    title = str(titles[0]).casefold()
    description = str(descriptions[0]).casefold() if descriptions else ""
    if "disambiguation" in title or "định hướng" in title or "disambiguation" in description:
        return None
    found = str(urls[0])
    if not found.startswith(f"https://{lang}.wikipedia.org/wiki/"):
        return None
    return found
