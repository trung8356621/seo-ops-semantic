from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TopicKeywordIn(BaseModel):
    ref: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("ref", "text")
    @classmethod
    def strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must not be blank")
        return stripped


class TopicAnalysisRequest(BaseModel):
    site_ref: str = Field(min_length=1, max_length=128)
    language: str | None = Field(default=None, max_length=32)
    keywords: list[TopicKeywordIn] = Field(min_length=1)
    input_hash: str | None = None
    request_id: str | None = None

    @field_validator("site_ref")
    @classmethod
    def site_ref_strip(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("site_ref must not be blank")
        return stripped

    @field_validator("language")
    @classmethod
    def language_strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_unique_refs(self) -> TopicAnalysisRequest:
        refs = [item.ref for item in self.keywords]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate keyword refs are not allowed")
        return self


class TopicModelInfo(BaseModel):
    provider: str
    name: str
    version: str
    dimensions: int


class TopicGroupMemberOut(BaseModel):
    keyword_ref: str
    text: str
    similarity_score: float = Field(
        description="Cosine similarity to the group representative (not a probability).",
    )
    confidence: float = Field(
        description=(
            "Heuristic confidence in [0,1]: similarity_score scaled relative to "
            "assignment floor. Not a calibrated probability."
        ),
    )
    is_representative: bool = False


class TopicGroupOut(BaseModel):
    group_ref: str
    suggested_label: str
    member_count: int
    mean_similarity: float
    min_similarity: float
    cohesion: float
    members: list[TopicGroupMemberOut]


class TopicUnassignedOut(BaseModel):
    keyword_ref: str
    text: str
    reason: str


class TopicAnalysisDiagnostics(BaseModel):
    keyword_count: int
    group_count: int
    unassigned_count: int
    singleton_count: int
    low_confidence_member_count: int
    group_size_histogram: dict[str, int]
    timings_ms: dict[str, int]
    embedding_cache: dict[str, int]
    algorithm: str
    algorithm_config: dict[str, Any]
    cluster_diagnostics: dict[str, Any] = Field(default_factory=dict)


class TopicAnalysisResponse(BaseModel):
    analysis_id: str
    status: Literal["completed", "failed"]
    site_ref: str
    language: str | None
    input_hash: str
    request_id: str | None = None
    model: TopicModelInfo
    groups: list[TopicGroupOut]
    unassigned: list[TopicUnassignedOut]
    diagnostics: TopicAnalysisDiagnostics
    started_at: str
    finished_at: str
    duration_ms: int
    error: str | None = None
