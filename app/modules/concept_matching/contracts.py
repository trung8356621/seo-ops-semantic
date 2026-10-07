from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.text.normalization import normalize_text


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        normalized = normalize_text(raw)
        if normalized == "" or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


class ConceptEntityIn(BaseModel):
    ref: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("ref")
    @classmethod
    def ref_strip(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("ref must not be blank")
        return stripped

    @field_validator("text")
    @classmethod
    def text_normalize(cls, value: str) -> str:
        normalized = normalize_text(value)
        if normalized == "":
            raise ValueError("text must not be empty after normalization")
        return normalized


class ConceptDefinitionIn(BaseModel):
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

    @field_validator("positive_examples", "negative_examples", mode="before")
    @classmethod
    def coerce_list(cls, value: object) -> object:
        if value is None:
            return []
        return value

    @model_validator(mode="after")
    def normalize_examples(self) -> ConceptDefinitionIn:
        positives = _dedupe_preserve(list(self.positive_examples))
        if not positives:
            raise ValueError("positive_examples must contain at least one non-empty entry")
        negatives = _dedupe_preserve(list(self.negative_examples))
        self.positive_examples = positives
        self.negative_examples = negatives
        return self


class ConceptDecisionPolicy(BaseModel):
    """Optional caller-supplied gates. Scores remain cosine similarities, not probabilities."""

    min_positive_score: float = Field(ge=-1.0, le=1.0)
    min_margin: float = Field(default=0.0, ge=-2.0, le=2.0)


class ConceptMatchAnalysisRequest(BaseModel):
    scope_ref: str = Field(min_length=1, max_length=128)
    language: str | None = Field(default=None, max_length=32)
    entities: list[ConceptEntityIn] = Field(min_length=1)
    concepts: list[ConceptDefinitionIn] = Field(min_length=1)
    decision_policy: ConceptDecisionPolicy | None = None

    @field_validator("scope_ref")
    @classmethod
    def scope_strip(cls, value: str) -> str:
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
    def unique_refs_and_keys(self) -> ConceptMatchAnalysisRequest:
        refs = [item.ref for item in self.entities]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate entity refs are not allowed")
        keys = [item.key for item in self.concepts]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate concept keys are not allowed")
        return self


class ConceptScoreOut(BaseModel):
    key: str
    positive_max: float = Field(description="Max cosine similarity to any positive example.")
    positive_top_k_mean: float = Field(
        description="Mean of top-K positive cosine similarities (K=min(3, n_positives))."
    )
    negative_max: float | None = Field(
        default=None,
        description="Max cosine similarity to any negative example, or null when none.",
    )
    margin: float | None = Field(
        default=None,
        description="positive_max - negative_max when negatives exist; otherwise null.",
    )
    best_positive_example: str
    best_positive_similarity: float
    best_negative_example: str | None = None
    best_negative_similarity: float | None = None
    suggested_match: bool | None = Field(
        default=None,
        description="Set only when decision_policy is supplied; otherwise null.",
    )


class ConceptEntityOut(BaseModel):
    ref: str
    text: str
    concepts: list[ConceptScoreOut]


class ConceptMatchDiagnostics(BaseModel):
    entity_count: int
    concept_count: int
    unique_text_count: int
    embed_ms: int
    score_ms: int
    total_ms: int
    model: str
    provider: str
    dimensions: int
    cache: str = Field(description="V1 uses direct embed_batch; no persistent text cache.")


class ConceptMatchAnalysisResponse(BaseModel):
    analysis_id: str
    scope_ref: str
    language: str | None
    entities: list[ConceptEntityOut]
    diagnostics: ConceptMatchDiagnostics
