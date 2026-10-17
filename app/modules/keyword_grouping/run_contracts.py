from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RunCreateRequest(BaseModel):
    scope_ref: str = Field(min_length=1, max_length=128)
    revision: int | None = Field(default=None, ge=1)
    language: str | None = Field(default=None, max_length=32)

    @field_validator("scope_ref")
    @classmethod
    def strip_scope(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("scope_ref must not be blank")
        return stripped


class RunStatusResponse(BaseModel):
    run_id: str
    scope_ref: str
    dataset_revision: int
    dataset_hash: str
    fingerprint: str
    status: str
    progress_phase: str
    progress_completed: int
    progress_total: int
    group_count: int
    member_count: int
    unassigned_count: int
    algorithm: str
    input_hash: str
    error: str | None
    started_at: str | None
    finished_at: str | None
    heartbeat_at: str | None
    created_at: str | None


class RunGroupPageResponse(BaseModel):
    run_id: str
    status: str
    scope_ref: str
    input_hash: str
    algorithm: str
    groups: list[dict[str, Any]]
    next_cursor: int | None
