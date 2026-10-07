from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.modules.concept_matching.analyzer import ConceptMatchAnalyzer
from app.modules.concept_matching.contracts import (
    ConceptMatchAnalysisRequest,
    ConceptMatchAnalysisResponse,
)

router = APIRouter(prefix="/v1/concept-matches", tags=["concept-matching"])


@router.post(
    "/analyses",
    response_model=ConceptMatchAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
def create_concept_match_analysis(
    body: ConceptMatchAnalysisRequest,
    request: Request,
) -> ConceptMatchAnalysisResponse:
    provider = request.app.state.embedding_provider
    try:
        if not provider.is_loaded:
            provider.load()
        return ConceptMatchAnalyzer(provider).analyze(body)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
