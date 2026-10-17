from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.modules.keyword_grouping.contracts import IndustryMembershipIn


class DatasetSessionRequest(BaseModel):
    scope_ref: str = Field(min_length=1, max_length=128)
    language: str | None = Field(default=None, max_length=32)
    expected_keyword_count: int = Field(ge=1)
    expected_batch_count: int = Field(ge=1)

    @field_validator("scope_ref")
    @classmethod
    def strip_scope(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("scope_ref must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def strip_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class DatasetKeywordIn(BaseModel):
    ref: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2000)
    language: str | None = Field(default=None, max_length=32)
    industry_evidence: list[IndustryMembershipIn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ref", "text")
    @classmethod
    def strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def strip_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class DatasetBatchRequest(BaseModel):
    keywords: list[DatasetKeywordIn] = Field(min_length=1)


class DatasetCommitRequest(BaseModel):
    dataset_hash: str = Field(min_length=64, max_length=64)


class DatasetSessionResponse(BaseModel):
    session_id: str
    scope_ref: str
    revision: int
    expected_keyword_count: int
    expected_batch_count: int
    expires_at: str


class DatasetBatchResponse(BaseModel):
    session_id: str
    batch_index: int
    keyword_count: int
    batch_hash: str
    idempotent: bool


class DatasetCommitResponse(BaseModel):
    scope_ref: str
    session_id: str
    revision: int
    dataset_hash: str
    keyword_count: int
    status: str
    unchanged: bool


class DatasetCurrentResponse(BaseModel):
    scope_ref: str
    revision: int
    dataset_hash: str
    language: str | None
    keyword_count: int
    committed_at: str
    refs: list[str] | None = None
