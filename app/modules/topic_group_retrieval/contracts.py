from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class TopicGroupCandidateIn(BaseModel):
    ref: str = Field(min_length=1, max_length=191)
    label: str = ""
    examples: list[str] = Field(min_length=1)

    @field_validator("ref")
    @classmethod
    def ref_strip(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("ref must not be blank")
        return stripped


class TopicGroupMatchPolicyIn(BaseModel):
    """Topic-group gates. Independent from tool-intent thresholds."""

    min_score: float = Field(default=0.55, ge=-1.0, le=1.0)
    limit: int = Field(default=3, ge=1, le=20)


class TopicGroupMatchRequest(BaseModel):
    scope_ref: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    language: str | None = None
    groups: list[TopicGroupCandidateIn] = Field(min_length=1)
    policy: TopicGroupMatchPolicyIn = Field(default_factory=TopicGroupMatchPolicyIn)

    @field_validator("query", "scope_ref")
    @classmethod
    def strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must not be blank")
        return stripped

    @model_validator(mode="after")
    def unique_refs(self) -> TopicGroupMatchRequest:
        refs = [item.ref for item in self.groups]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate topic group refs are not allowed")
        return self


class TopicGroupEvidenceOut(BaseModel):
    lexical_matched: bool
    semantic_score: float | None = None
    best_example: str | None = None


class TopicGroupMatchOut(BaseModel):
    ref: str
    score: float
    evidence: TopicGroupEvidenceOut


class TopicGroupMatchResponse(BaseModel):
    """Ranked external refs only. No keyword, article, or permission mutation."""

    scope_ref: str
    matches: list[TopicGroupMatchOut]
