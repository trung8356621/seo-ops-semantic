from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app.diagnostics.health import build_health, build_ready, check_live

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    database = request.app.state.database
    provider = request.app.state.embedding_provider
    return build_health(settings, database, provider, force_embedding_load=False)


@router.get("/health/live")
def health_live(request: Request) -> dict:
    return check_live(request.app.state.settings)


@router.get("/health/ready")
def health_ready(request: Request, response: Response) -> dict:
    payload, ready = build_ready(
        request.app.state.settings,
        request.app.state.database,
        request.app.state.embedding_provider,
    )
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload
