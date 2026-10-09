from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CtaIntent = Literal[
    "product_discovery",
    "service_discovery",
    "comparison",
    "consultation",
    "conversion",
]
LegacyClass = Literal["promotional", "uncertain", "editorial"]
PlacementPosition = Literal["section_end"]


class CtaSectionIn(BaseModel):
    section_id: str = Field(min_length=1, max_length=128)
    heading: str = Field(default="", max_length=500)
    position_ratio: float = Field(ge=0.0, le=1.0)
    start_word: int = Field(default=0, ge=0)
    word_count: int = Field(ge=0)
    content: str = Field(default="", max_length=8000)

    @field_validator("section_id")
    @classmethod
    def strip_id(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("section_id must not be blank")
        return stripped


class LegacyCandidateIn(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    section_id: str = Field(default="", max_length=128)
    text: str = Field(default="", max_length=2000)
    structural_signal: str = Field(default="unknown", max_length=64)


class CtaPlanRequest(BaseModel):
    language: str = Field(default="en", max_length=32)
    article_word_count: int = Field(default=0, ge=0)
    article_type: str = Field(default="article", max_length=64)
    sections: list[CtaSectionIn] = Field(default_factory=list)
    legacy_candidates: list[LegacyCandidateIn] = Field(default_factory=list)


class CtaPlacementOut(BaseModel):
    placement_id: str
    section_id: str
    position: PlacementPosition = "section_end"
    intent: CtaIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SkippedSectionOut(BaseModel):
    section_id: str
    reason: str
    best_intent: str | None = None
    score: float | None = None


class LegacyClassificationOut(BaseModel):
    candidate_id: str
    section_id: str
    classification: LegacyClass
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class CtaPlanResponse(BaseModel):
    placements: list[CtaPlacementOut]
    skipped: list[SkippedSectionOut]
    legacy: list[LegacyClassificationOut]
    debug: dict[str, str | int | float]
