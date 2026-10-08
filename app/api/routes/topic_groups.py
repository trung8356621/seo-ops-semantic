from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.modules.topic_group_retrieval.contracts import (
    TopicGroupMatchRequest,
    TopicGroupMatchResponse,
)
from app.modules.topic_group_retrieval.matcher import TopicGroupMatcher

router = APIRouter(prefix="/v1/topic-groups", tags=["topic-groups"])


@router.post("/matches", response_model=TopicGroupMatchResponse)
def match_topic_groups(body: TopicGroupMatchRequest, request: Request) -> TopicGroupMatchResponse:
    provider = request.app.state.embedding_provider
    try:
        if not provider.is_loaded:
            provider.load()
        return TopicGroupMatcher(provider).match(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
