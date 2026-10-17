from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from app.core.vector.postgres import PostgresVectorStore
from app.modules.keyword_grouping.analyzer import KeywordGroupAnalyzer
from app.modules.keyword_grouping.contracts import (
    KeywordGroupAnalysisRequest,
    KeywordGroupAnalysisResponse,
    KeywordGroupEmbeddingBatchRequest,
    KeywordGroupEmbeddingBatchResponse,
    KeywordGroupSearchRequest,
    KeywordGroupSearchResponse,
)
from app.modules.keyword_grouping.dataset_contracts import (
    DatasetBatchRequest,
    DatasetBatchResponse,
    DatasetCommitRequest,
    DatasetCommitResponse,
    DatasetCurrentResponse,
    DatasetSessionRequest,
    DatasetSessionResponse,
)
from app.modules.keyword_grouping.dataset_repository import (
    DatasetConflict,
    DatasetNotFound,
    DatasetRejected,
    KeywordGroupDatasetRepository,
)
from app.modules.keyword_grouping.run_contracts import RunCreateRequest, RunGroupPageResponse, RunStatusResponse
from app.modules.keyword_grouping.run_repository import KeywordGroupRunRepository, RunConflict, RunNotFound, RunRejected
from app.modules.keyword_grouping.fingerprint import build_fingerprint

router = APIRouter(prefix="/v1/keyword-groups", tags=["keyword-grouping"])


@router.post(
    "/datasets/sessions",
    response_model=DatasetSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def begin_dataset_session(body: DatasetSessionRequest, request: Request) -> DatasetSessionResponse:
    with request.app.state.database.connection() as conn:
        repository = KeywordGroupDatasetRepository(conn, request.app.state.settings)
        try:
            payload = repository.begin_session(
                scope_ref=body.scope_ref,
                language=body.language,
                expected_keyword_count=body.expected_keyword_count,
                expected_batch_count=body.expected_batch_count,
            )
        except DatasetRejected as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.detail) from exc
    return DatasetSessionResponse.model_validate(payload)


@router.put(
    "/datasets/sessions/{session_id}/batches/{batch_index}",
    response_model=DatasetBatchResponse,
)
def upload_dataset_batch(
    session_id: str,
    batch_index: int,
    body: DatasetBatchRequest,
    request: Request,
) -> DatasetBatchResponse:
    with request.app.state.database.connection() as conn:
        repository = KeywordGroupDatasetRepository(conn, request.app.state.settings)
        try:
            payload = repository.put_batch(
                session_id,
                batch_index,
                [item.model_dump() for item in body.keywords],
            )
        except DatasetNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.detail) from exc
        except DatasetConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.detail) from exc
        except DatasetRejected as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.detail) from exc
    return DatasetBatchResponse.model_validate(payload)


@router.post(
    "/datasets/sessions/{session_id}/commit",
    response_model=DatasetCommitResponse,
)
def commit_dataset_session(
    session_id: str,
    body: DatasetCommitRequest,
    request: Request,
) -> DatasetCommitResponse:
    with request.app.state.database.connection() as conn:
        repository = KeywordGroupDatasetRepository(conn, request.app.state.settings)
        try:
            payload = repository.commit_session(session_id, body.dataset_hash)
        except DatasetNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.detail) from exc
        except DatasetConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.detail) from exc
        except DatasetRejected as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.detail) from exc
    return DatasetCommitResponse.model_validate(payload)


@router.post("/datasets/sessions/{session_id}/abort", status_code=status.HTTP_204_NO_CONTENT)
def abort_dataset_session(session_id: str, request: Request) -> Response:
    with request.app.state.database.connection() as conn:
        repository = KeywordGroupDatasetRepository(conn, request.app.state.settings)
        try:
            repository.abort_session(session_id)
        except DatasetNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.detail) from exc
        except DatasetRejected as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.detail) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/datasets/{scope_ref}", response_model=DatasetCurrentResponse)
def get_dataset(
    scope_ref: str,
    request: Request,
    include_refs: bool = False,
    language: str | None = None,
) -> DatasetCurrentResponse:
    with request.app.state.database.connection() as conn:
        row = KeywordGroupDatasetRepository(conn, request.app.state.settings).current(
            scope_ref,
            include_refs=include_refs,
            language=language,
        )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="dataset_not_found")
    return DatasetCurrentResponse.model_validate(row)


@router.post("/runs", response_model=RunStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def create_run(body: RunCreateRequest, request: Request) -> RunStatusResponse:
    with request.app.state.database.connection() as conn:
        with conn.cursor() as cur:
            language_clause = ""
            params: tuple[object, ...]
            if body.language is None:
                params = (body.scope_ref,) if body.revision is None else (body.scope_ref, body.revision)
            else:
                language_clause = " AND language IS NOT DISTINCT FROM %s"
                params = (
                    (body.scope_ref, body.language)
                    if body.revision is None
                    else (body.scope_ref, body.revision, body.language)
                )
            revision_clause = "" if body.revision is None else " AND revision = %s"
            if body.revision is not None and body.language is not None:
                params = (body.scope_ref, body.revision, body.language)
            cur.execute(
                f"""
                SELECT id, revision, dataset_hash, keyword_count, language
                FROM keyword_group_dataset_revisions
                WHERE scope_ref = %s AND status = 'committed'{revision_clause}{language_clause}
                """,
                params,
            )
            dataset = cur.fetchone()
        if dataset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="committed_dataset_not_found")
        fingerprint, payload = build_fingerprint(
            request.app.state.settings,
            str(dataset["dataset_hash"]),
            request.app.state.embedding_provider,
            scope_ref=body.scope_ref,
            language=dataset["language"],
        )
        try:
            row = KeywordGroupRunRepository(conn).create(
                scope_ref=body.scope_ref,
                revision_id=int(dataset["id"]),
                dataset_revision=int(dataset["revision"]),
                dataset_hash=str(dataset["dataset_hash"]),
                fingerprint=fingerprint,
                fingerprint_payload=payload,
                algorithm=str(payload["algorithm"]),
                keyword_count=int(dataset["keyword_count"]),
                language=dataset["language"],
            )
        except RunConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.detail) from exc
    return RunStatusResponse.model_validate(row)


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run(run_id: str, request: Request) -> RunStatusResponse:
    with request.app.state.database.connection() as conn:
        row = KeywordGroupRunRepository(conn).get(run_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_not_found")
    return RunStatusResponse.model_validate(row)


@router.get("/runs/{run_id}/groups", response_model=RunGroupPageResponse)
def get_run_groups(
    run_id: str,
    request: Request,
    cursor: int = 0,
    limit: int = 100,
) -> RunGroupPageResponse:
    page_limit = max(1, min(limit, 500))
    with request.app.state.database.connection() as conn:
        try:
            page = KeywordGroupRunRepository(conn).groups_page(run_id, cursor=max(0, cursor), limit=page_limit)
        except RunNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.detail) from exc
        except RunRejected as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.detail) from exc
    return RunGroupPageResponse.model_validate(page)


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


@router.post(
    "/embedding-batches",
    response_model=KeywordGroupEmbeddingBatchResponse,
    status_code=status.HTTP_200_OK,
)
def create_embedding_batch(
    body: KeywordGroupEmbeddingBatchRequest,
    request: Request,
) -> KeywordGroupEmbeddingBatchResponse:
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
            return analyzer.embed_batch(body)
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


@router.post(
    "/analyses/from-cache",
    response_model=KeywordGroupAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
def create_analysis_from_cache(
    body: KeywordGroupAnalysisRequest,
    request: Request,
) -> KeywordGroupAnalysisResponse:
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
            return analyzer.analyze_from_cache(body)
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
