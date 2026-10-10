from __future__ import annotations

import secrets
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, model_validator

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
    review_kind: str = Field(pattern="^(answer_rating_legacy|candidate_selection)$")
    rating: bool | None = None
    selected_candidate_id: str | None = Field(default=None, max_length=360)
    preferred_candidate_id: str | None = Field(default=None, max_length=360)
    none_of_above: bool = False

    @model_validator(mode="after")
    def valid_choice(self) -> "FeedbackItem":
        if self.review_kind == "answer_rating_legacy":
            if self.rating is None:
                raise ValueError("legacy answer rating requires rating")
            return self
        if self.rating is not None:
            raise ValueError("candidate review cannot include rating")
        if self.none_of_above == (self.preferred_candidate_id is not None):
            raise ValueError("candidate review requires exactly one preference")
        return self


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
                        routing_outcome, review_kind, rating,
                        selected_candidate_id, preferred_candidate_id, none_of_above
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        review_kind = EXCLUDED.review_kind,
                        rating = EXCLUDED.rating,
                        selected_candidate_id = EXCLUDED.selected_candidate_id,
                        preferred_candidate_id = EXCLUDED.preferred_candidate_id,
                        none_of_above = EXCLUDED.none_of_above,
                        updated_at = NOW()
                    WHERE agent_routing_feedback.client_id = EXCLUDED.client_id
                    """,
                    (
                        item.review_id, item.client_id, item.agent_app,
                        item.service_id, item.module_id, item.operation_id,
                        item.routing_group_id, item.routing_version,
                        item.routing_outcome, item.review_kind, item.rating,
                        item.selected_candidate_id, item.preferred_candidate_id,
                        item.none_of_above,
                    ),
                )
                if cur.rowcount == 1:
                    acknowledged.append(item.review_id)
        conn.commit()
    return FeedbackAck(acknowledged=acknowledged)
