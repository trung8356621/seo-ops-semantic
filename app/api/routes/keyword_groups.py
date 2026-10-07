from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from app.core.vector.postgres import PostgresVectorStore
from app.modules.keyword_grouping.analyzer import KeywordGroupAnalyzer
from app.modules.keyword_grouping.contracts import (
    KeywordGroupAnalysisRequest,
    KeywordGroupAnalysisResponse,
    KeywordGroupSearchRequest,
    KeywordGroupSearchResponse,
)
from app.modules.keyword_grouping.repository import KeywordGroupAnalysisRepository

router = APIRouter(prefix="/v1/keyword-groups", tags=["keyword-grouping"])


@router.post("/search", response_model=KeywordGroupSearchResponse, status_code=status.HTTP_200_OK)
def search_keywords(body: KeywordGroupSearchRequest, request: Request) -> KeywordGroupSearchResponse:
    settings = request.app.state.settings
    database = request.app.state.database
    provider = request.app.state.embedding_provider

    try:
        with database.connection() as conn:
            analyzer = KeywordGroupAnalyzer(
                settings=settings,
                embedding=provider,
                vectors=PostgresVectorStore(conn),
                repository=None,
            )
            return analyzer.search(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/analyses", response_model=KeywordGroupAnalysisResponse, status_code=status.HTTP_200_OK)
def create_analysis(body: KeywordGroupAnalysisRequest, request: Request) -> KeywordGroupAnalysisResponse:
    settings = request.app.state.settings
    database = request.app.state.database
    provider = request.app.state.embedding_provider

    try:
        with database.connection() as conn:
            analyzer = KeywordGroupAnalyzer(
                settings=settings,
                embedding=provider,
                vectors=PostgresVectorStore(conn),
                repository=KeywordGroupAnalysisRepository(conn),
            )
            return analyzer.analyze(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/analyses/{analysis_id}", response_model=KeywordGroupAnalysisResponse)
def get_analysis(analysis_id: str, request: Request) -> KeywordGroupAnalysisResponse:
    database = request.app.state.database
    with database.connection() as conn:
        row = KeywordGroupAnalysisRepository(conn).get(analysis_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="analysis_not_found")
    return row


@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: str, request: Request) -> Response:
    database = request.app.state.database
    with database.connection() as conn:
        deleted = KeywordGroupAnalysisRepository(conn).delete(analysis_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="analysis_not_found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
