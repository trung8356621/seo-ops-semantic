from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.modules.wiki_suggestions.contracts import WikiSuggestionRequest, WikiSuggestionResponse
from app.modules.wiki_suggestions.suggester import suggest_wiki_links

router = APIRouter(prefix="/v1/wiki-suggestions", tags=["wiki-suggestions"])


@router.post("", response_model=WikiSuggestionResponse)
def create_wiki_suggestions(body: WikiSuggestionRequest) -> WikiSuggestionResponse:
    try:
        return suggest_wiki_links(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
