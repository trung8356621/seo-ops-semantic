from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.modules.cta_planner.contracts import CtaPlanRequest, CtaPlanResponse
from app.modules.cta_planner.planner import CtaPlanner

router = APIRouter(prefix="/v1/cta", tags=["cta"])


@router.post("/plan", response_model=CtaPlanResponse)
def plan_ctas(body: CtaPlanRequest, request: Request) -> CtaPlanResponse:
    provider = request.app.state.embedding_provider
    try:
        if not provider.is_loaded:
            provider.load()
        return CtaPlanner(provider).plan(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
