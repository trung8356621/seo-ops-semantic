from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from app.modules.internal_link_v2.contracts import InternalLinkRankRequest, InternalLinkRankResponse
from app.modules.internal_link_v2.ranker import rank_internal_links

router = APIRouter(prefix="/v1/internal-links", tags=["internal-links-v2"])


@router.post("/v2/rank", response_model=InternalLinkRankResponse)
def rank_internal_link_v2(body: InternalLinkRankRequest, request: Request) -> InternalLinkRankResponse:
    del request
    try:
        return rank_internal_links(body)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
