from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.modules.tool_intent.contracts import ToolIntentMatchRequest, ToolIntentMatchResponse
from app.modules.tool_intent.router import ToolIntentRouter

router = APIRouter(prefix="/v1/tool-intents", tags=["tool-intents"])


@router.post("/match", response_model=ToolIntentMatchResponse)
def match_tool_intent(body: ToolIntentMatchRequest, request: Request) -> ToolIntentMatchResponse:
    provider = request.app.state.embedding_provider
    try:
        if not provider.is_loaded:
            provider.load()
        return ToolIntentRouter(provider).match(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
