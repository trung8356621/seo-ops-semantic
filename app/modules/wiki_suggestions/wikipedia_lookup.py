"""Single-term Wikipedia opensearch lookup. Does not crawl or embed pages."""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

Lookup = Callable[[str, str], str | None]
Fetch = Callable[[str], object]
USER_AGENT = "seo-ops-semantic/0.1 (wiki suggestion lookup)"


def wikipedia_opensearch(term: str, language: str, fetch: Fetch | None = None) -> str | None:
    lang = "vi" if language.lower().startswith("vi") else "en"
    loader = fetch or _fetch_json
    payload = _load(
        loader,
        _api(lang, {"action": "opensearch", "search": term, "limit": "5", "namespace": "0", "format": "json"}),
    )
    if not isinstance(payload, list) or len(payload) < 4:
        return None
    titles = payload[1] if isinstance(payload[1], list) else []
    descriptions = payload[2] if isinstance(payload[2], list) else []
    urls = payload[3] if isinstance(payload[3], list) else []
    found: str | None = None
    for index, title in enumerate(titles):
        if str(title).casefold() != term.casefold():
            continue
        description = str(descriptions[index]).casefold() if index < len(descriptions) else ""
        if "disambiguation" in str(title).casefold() or "định hướng" in description or "disambiguation" in description:
            return None
        if index >= len(urls):
            return None
        found = str(urls[index])
        break
    if found is None or not found.startswith(f"https://{lang}.wikipedia.org/wiki/"):
        return None
    info = _load(
        loader,
        _api(lang, {"action": "query", "titles": term, "prop": "info", "format": "json"}),
    )
    if _page_is_redirect_or_missing(info):
        return None
    return found


def _api(language: str, params: dict[str, str]) -> str:
    return f"https://{language}.wikipedia.org/w/api.php?{urlencode(params)}"


def _load(fetch: Fetch, url: str) -> object:
    try:
        return fetch(url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _page_is_redirect_or_missing(payload: object) -> bool:
    if not isinstance(payload, dict):
        return True
    pages = payload.get("query", {}).get("pages", {}) if isinstance(payload.get("query"), dict) else {}
    if not isinstance(pages, dict) or pages == {}:
        return True
    page = next(iter(pages.values()))
    if not isinstance(page, dict):
        return True
    return "missing" in page or "redirect" in page
