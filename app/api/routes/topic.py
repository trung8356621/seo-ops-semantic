from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from app.core.vector.postgres import PostgresVectorStore
from app.modules.topic.analyzer import TopicAnalyzer
from app.modules.topic.contracts import TopicAnalysisRequest, TopicAnalysisResponse
from app.modules.topic.repository import TopicAnalysisRepository

router = APIRouter(prefix="/v1/topic", tags=["topic-analysis"])


@router.post("/analyses", response_model=TopicAnalysisResponse, status_code=status.HTTP_200_OK)
def create_analysis(body: TopicAnalysisRequest, request: Request) -> TopicAnalysisResponse:
    settings = request.app.state.settings
    database = request.app.state.database
    provider = request.app.state.embedding_provider

    try:
        with database.connection() as conn:
            analyzer = TopicAnalyzer(
                settings=settings,
                embedding=provider,
                vectors=PostgresVectorStore(conn),
                repository=TopicAnalysisRepository(conn),
            )
            return analyzer.analyze(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/analyses/{analysis_id}", response_model=TopicAnalysisResponse)
def get_analysis(analysis_id: str, request: Request) -> TopicAnalysisResponse:
    database = request.app.state.database
    with database.connection() as conn:
        row = TopicAnalysisRepository(conn).get(analysis_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="analysis_not_found")
    return row


@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: str, request: Request) -> Response:
    database = request.app.state.database
    with database.connection() as conn:
        deleted = TopicAnalysisRepository(conn).delete(analysis_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="analysis_not_found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
