from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DistributionPolicyIn(BaseModel):
    """Deterministic weights for this use case. Components stay visible in evidence."""

    overuse_weight: float = Field(default=0.45, ge=0.0, le=5.0)
    underlinked_bonus: float = Field(default=0.18, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=0.35, ge=0.0, le=5.0)


class InternalLinkCandidateIn(BaseModel):
    ref: str = Field(min_length=1, max_length=191)
    topic_group_ref: str = Field(min_length=1, max_length=191)
    url: str = ""
    eligible: bool = False
    inbound_count: int = Field(default=0, ge=0)
    outbound_count: int = Field(default=0, ge=0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    representation: str = ""
    same_as_source: bool = False
    already_linked_from_source: bool = False

    @field_validator("ref", "topic_group_ref")
    @classmethod
    def strip_ref(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("ref must not be blank")
        return stripped


class InternalLinkRankRequest(BaseModel):
    source_ref: str = Field(min_length=1, max_length=191)
    source_text: str = ""
    candidate_boundary: Literal["topic_group"]
    candidates: list[InternalLinkCandidateIn]
    limit: int = Field(default=5, ge=1, le=20)
    policy: DistributionPolicyIn = Field(default_factory=DistributionPolicyIn)

    @field_validator("source_ref")
    @classmethod
    def strip_source(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("source_ref must not be blank")
        return stripped


class DistributionMetricsOut(BaseModel):
    articles_with_zero_inbound: int
    inbound_counts: dict[str, int]
    median_inbound: float
    p90_inbound: float
    max_inbound: int
    outbound_counts: dict[str, int]
    repeated_target_refs: list[str]
    top_link_concentration: float


class RankComponentOut(BaseModel):
    relevance: float
    overuse_penalty: float
    underlinked_bonus: float
    repetition_penalty: float
    final_score: float


class InternalLinkSuggestionOut(BaseModel):
    ref: str
    topic_group_ref: str
    url: str
    score: float
    components: RankComponentOut


class RejectedCandidateOut(BaseModel):
    ref: str
    reason: str


class InternalLinkRankResponse(BaseModel):
    source_ref: str
    candidate_boundary: Literal["topic_group"]
    suggestions: list[InternalLinkSuggestionOut]
    rejected: list[RejectedCandidateOut]
    metrics: DistributionMetricsOut
