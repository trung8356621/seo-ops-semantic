from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class KeywordIn(BaseModel):
    ref: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("ref", "text")
    @classmethod
    def strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must not be blank")
        return stripped


class KeywordGroupAnalysisRequest(BaseModel):
    scope_ref: str = Field(min_length=1, max_length=128)
    language: str | None = Field(default=None, max_length=32)
    keywords: list[KeywordIn] = Field(min_length=1)
    input_hash: str | None = None
    request_id: str | None = None

    @field_validator("scope_ref")
    @classmethod
    def scope_ref_strip(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("scope_ref must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def language_strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_unique_refs(self) -> KeywordGroupAnalysisRequest:
        refs = [item.ref for item in self.keywords]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate keyword refs are not allowed")
        return self


class GroupMemberOut(BaseModel):
    ref: str
    text: str
    similarity_score: float = Field(
        description="Cosine similarity to the group representative (not a probability).",
    )
    is_representative: bool = False


class GroupOut(BaseModel):
    group_ref: str
    representative_ref: str
    representative_text: str
    member_count: int
    mean_similarity: float
    min_similarity: float
    cohesion: float
    members: list[GroupMemberOut]


class UnassignedOut(BaseModel):
    ref: str
    text: str
    reason: str


class KeywordGroupDiagnostics(BaseModel):
    keyword_count: int
    group_count: int
    unassigned_count: int
    algorithm: str
    algorithm_config: dict[str, Any]
    timings_ms: dict[str, int]
    embedding_cache: dict[str, int]
    strategy: str | None = None
    lexical_reject_count: int | None = None
    rescue_assignment_count: int | None = None
    ambiguous_count: int | None = None


class KeywordGroupAnalysisResponse(BaseModel):
    analysis_id: str
    status: Literal["completed", "failed"]
    scope_ref: str
    language: str | None
    input_hash: str
    request_id: str | None = None
    groups: list[GroupOut]
    unassigned: list[UnassignedOut]
    diagnostics: KeywordGroupDiagnostics
    error: str | None = None


class KeywordGroupSearchRequest(BaseModel):
    """Rank supplied candidate keywords by cosine similarity to a query text."""

    scope_ref: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)
    keywords: list[KeywordIn] = Field(min_length=1)
    language: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=20, ge=1, le=50)
    request_id: str | None = None

    @field_validator("scope_ref", "query")
    @classmethod
    def strip_required_search(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def language_strip_search(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_unique_search_refs(self) -> KeywordGroupSearchRequest:
        refs = [item.ref for item in self.keywords]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate keyword refs are not allowed")
        return self


class KeywordGroupSearchHit(BaseModel):
    ref: str
    text: str
    similarity_score: float
    # True when similarity meets topic_assignment_min_score (Python-owned gate).
    accepted: bool


class KeywordGroupSearchResponse(BaseModel):
    scope_ref: str
    query: str
    language: str | None
    hits: list[KeywordGroupSearchHit]
    candidate_count: int
    embedding_cache: dict[str, int]
    acceptance_min_score: float | None = None
    request_id: str | None = None
