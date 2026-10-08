from __future__ import annotations

from app.core.text.concept_lexical import matches_value
from app.core.text.normalization import casefold_preserve_accents, fold_accents
from app.modules.wiki_suggestions.contracts import (
    GENERIC_TERMS,
    CanonicalConceptIn,
    WikiSuggestionOut,
    WikiSuggestionRequest,
    WikiSuggestionResponse,
    _ACRONYM,
)
from app.modules.wiki_suggestions.verified_cache import verified_for_language
from app.modules.wiki_suggestions.wikipedia_lookup import Lookup


def suggest_wiki_links(
    request: WikiSuggestionRequest,
    lookup: Lookup | None = None,
) -> WikiSuggestionResponse:
    catalog = list(request.catalog) if request.catalog else verified_for_language(request.language)
    rejected: list[str] = []
    folded_content = fold_accents(request.content)
    for generic in GENERIC_TERMS:
        if fold_accents(generic) in folded_content:
            rejected.append(generic)

    found: list[WikiSuggestionOut] = []
    seen_refs: set[str] = set()
    for concept in catalog:
        term = _verified_term(request.content, concept, request.policy.min_semantic_score)
        if term is None:
            continue
        if _is_generic(term) or _is_generic(concept.label):
            rejected.append(concept.label)
            continue
        if concept.ref in seen_refs:
            continue
        seen_refs.add(concept.ref)
        found.append(
            WikiSuggestionOut(
                ref=concept.ref,
                term=term,
                url=concept.url,
                score=1.0 if term.isupper() else 0.9,
                evidence="canonical_alias" if concept.semantic_score is None else "canonical_alias+semantic",
            )
        )

    for acronym in _ACRONYM.findall(request.content):
        if _is_generic(acronym):
            continue
        if any(casefold_preserve_accents(item.term) == casefold_preserve_accents(acronym) for item in found):
            continue
        if not any(_name_hit(acronym, concept) for concept in catalog):
            resolved = _lookup_url(acronym, request, lookup)
            if resolved is None:
                rejected.append(acronym)
                continue
            found.append(
                WikiSuggestionOut(
                    ref="wiki:"+acronym.casefold(),
                    term=acronym,
                    url=resolved,
                    score=0.8,
                    evidence="wikipedia_opensearch",
                )
            )

    found.sort(key=lambda item: item.score, reverse=True)
    return WikiSuggestionResponse(
        article_ref=request.article_ref,
        suggestions=found[: request.policy.max_suggestions],
        rejected_terms=_unique(rejected),
    )


def _lookup_url(term: str, request: WikiSuggestionRequest, lookup: Lookup | None) -> str | None:
    if request.policy.lookup != "wikipedia" or lookup is None or _is_generic(term):
        return None
    found = lookup(term, request.language or "en")
    if found is None:
        return None
    if not (found.startswith("https://en.wikipedia.org/wiki/") or found.startswith("https://vi.wikipedia.org/wiki/")):
        return None
    return found


def _verified_term(content: str, concept: CanonicalConceptIn, min_semantic: float | None) -> str | None:
    if not (concept.url.startswith("https://") or concept.url.startswith("http://")):
        return None
    if min_semantic is not None and (
        concept.semantic_score is None or concept.semantic_score < min_semantic
    ):
        return None
    for name in [concept.label, *concept.aliases]:
        if _is_generic(name):
            continue
        if matches_value(content, name, "token"):
            return name
    return None


def _name_hit(term: str, concept: CanonicalConceptIn) -> bool:
    return any(
        casefold_preserve_accents(name) == casefold_preserve_accents(term)
        for name in [concept.label, *concept.aliases]
    )


def _is_generic(value: str) -> bool:
    folded = fold_accents(value)
    return folded in {fold_accents(item) for item in GENERIC_TERMS}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
