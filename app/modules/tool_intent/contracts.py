from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.modules.tool_intent.catalog import (
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_POSITIVE_SCORE,
    TOOL_INTENT_NAMESPACE,
)

ToolIntentStatus = Literal["confident", "ambiguous", "none"]


class ToolIntentDefinitionIn(BaseModel):
    key: str = Field(min_length=1, max_length=191)
    positive_examples: list[str] = Field(min_length=1)
    negative_examples: list[str] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def key_strip(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("key must not be blank")
        return stripped


class ToolIntentPolicyIn(BaseModel):
    """Tool-intent gates only. Not a universal semantic threshold."""

    min_positive_score: float = Field(default=DEFAULT_MIN_POSITIVE_SCORE, ge=-1.0, le=1.0)
    min_margin: float = Field(default=DEFAULT_MIN_MARGIN, ge=0.0, le=2.0)


class ToolIntentMatchRequest(BaseModel):
    scope_ref: str = Field(default=TOOL_INTENT_NAMESPACE, min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(default=None, max_length=32)
    intents: list[ToolIntentDefinitionIn] = Field(default_factory=list)
    allowed_keys: list[str] | None = None
    policy: ToolIntentPolicyIn = Field(default_factory=ToolIntentPolicyIn)

    @field_validator("query")
    @classmethod
    def query_strip(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("query must not be blank")
        return stripped

    @model_validator(mode="after")
    def unique_keys(self) -> ToolIntentMatchRequest:
        keys = [item.key for item in self.intents]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate intent keys are not allowed")
        return self


class ToolIntentEvidenceOut(BaseModel):
    lexical_matched: bool
    semantic_score: float | None = Field(
        default=None,
        description="Max cosine similarity to positive examples. Not a probability.",
    )
    runner_up_margin: float | None = None
    best_positive_example: str | None = None


class ToolIntentMatchOut(BaseModel):
    ref: str
    score: float = Field(description="Ranking evidence for this use case. Not a probability.")
    evidence: ToolIntentEvidenceOut


class ToolIntentMatchResponse(BaseModel):
    namespace: str = TOOL_INTENT_NAMESPACE
    status: ToolIntentStatus
    scope_ref: str
    matches: list[ToolIntentMatchOut]
    policy: ToolIntentPolicyIn


class WeightedTargetIn(BaseModel):
    ref: str = Field(min_length=1, max_length=191)
    weight: float = Field(gt=0, le=100)


class WeightedGroupIn(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    examples: list[str] = Field(min_length=1)
    targets: list[WeightedTargetIn] = Field(min_length=1)
    enabled: bool = True

    @field_validator("examples")
    @classmethod
    def examples_strip(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("examples must not be blank")
        return cleaned[:12]


class WeightedMatchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    groups: list[WeightedGroupIn] = Field(min_length=1)
    policy: ToolIntentPolicyIn = Field(default_factory=ToolIntentPolicyIn)

    @field_validator("query")
    @classmethod
    def query_strip(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("query must not be blank")
        return stripped


class WeightedCandidateOut(BaseModel):
    ref: str
    semantic_relevance: float
    weight: float
    score: float
    group_id: str
    example: str


class WeightedMatchResponse(BaseModel):
    status: ToolIntentStatus
    winner: str | None = None
    candidates: list[WeightedCandidateOut]
    policy: ToolIntentPolicyIn
