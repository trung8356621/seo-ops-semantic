from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from app.core.vector.postgres import PostgresVectorStore
from app.modules.internal_link_v2.contracts import InternalLinkRankRequest, InternalLinkRankResponse
from app.modules.internal_link_v2.ranker import rank_internal_links
from app.modules.internal_link_v2.relevance import apply_text_relevance

router = APIRouter(prefix="/v1/internal-links", tags=["internal-links-v2"])


@router.post("/v2/rank", response_model=InternalLinkRankResponse)
def rank_internal_link_v2(body: InternalLinkRankRequest, request: Request) -> InternalLinkRankResponse:
    provider = request.app.state.embedding_provider
    database = request.app.state.database
    try:
        if body.source_text.strip() != "":
            if not provider.is_loaded:
                provider.load()
            with database.connection() as conn:
                body = apply_text_relevance(body, provider, PostgresVectorStore(conn))
        return rank_internal_links(body)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
