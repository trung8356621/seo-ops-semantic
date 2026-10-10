from __future__ import annotations

import secrets
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/internal/agent-routing-feedback", tags=["internal"])


class FeedbackItem(BaseModel):
    review_id: UUID
    client_id: str = Field(min_length=1, max_length=160)
    agent_app: str = Field(min_length=1, max_length=120)
    service_id: str = Field(min_length=1, max_length=120)
    module_id: str | None = Field(default=None, max_length=160)
    operation_id: str | None = Field(default=None, max_length=160)
    routing_group_id: str | None = Field(default=None, max_length=160)
    routing_version: str | None = Field(default=None, max_length=160)
    routing_outcome: str = Field(min_length=1, max_length=80)
    rating: bool


class FeedbackBatch(BaseModel):
    items: list[FeedbackItem] = Field(min_length=1, max_length=100)


class FeedbackAck(BaseModel):
    acknowledged: list[UUID]


@router.post("/batch", response_model=FeedbackAck)
def ingest_feedback(body: FeedbackBatch, request: Request) -> FeedbackAck:
    configured = request.app.state.settings.internal_api_token
    supplied = request.headers.get("X-Internal-Token", "")
    if not configured or not secrets.compare_digest(configured, supplied):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal token.")

    database = request.app.state.database
    acknowledged: list[UUID] = []
    with database.connection() as conn:
        with conn.cursor() as cur:
            for item in body.items:
                cur.execute(
                    """
                    INSERT INTO agent_routing_feedback (
                        id, client_id, agent_app, service_id, module_id,
                        operation_id, routing_group_id, routing_version,
                        routing_outcome, rating
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        rating = EXCLUDED.rating,
                        updated_at = NOW()
                    WHERE agent_routing_feedback.client_id = EXCLUDED.client_id
                    """,
                    (
                        item.review_id, item.client_id, item.agent_app,
                        item.service_id, item.module_id, item.operation_id,
                        item.routing_group_id, item.routing_version,
                        item.routing_outcome, item.rating,
                    ),
                )
                if cur.rowcount == 1:
                    acknowledged.append(item.review_id)
        conn.commit()
    return FeedbackAck(acknowledged=acknowledged)
