from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

GENERIC_TERMS = frozenset(
    {
        "sản phẩm",
        "san pham",
        "khách hàng",
        "khach hang",
        "công ty",
        "cong ty",
        "product",
        "customer",
        "company",
        "website",
        "dịch vụ",
        "dich vu",
    }
)

_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")


class CanonicalConceptIn(BaseModel):
    ref: str = Field(min_length=1, max_length=191)
    label: str = Field(min_length=1, max_length=191)
    url: str = Field(min_length=1, max_length=500)
    aliases: list[str] = Field(default_factory=list)
    semantic_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional caller-supplied verification cosine. Not a probability.",
    )

    @field_validator("ref", "label", "url")
    @classmethod
    def strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must not be blank")
        return stripped


class WikiSuggestionPolicyIn(BaseModel):
    max_suggestions: int = Field(default=2, ge=1, le=2)
    min_semantic_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class WikiSuggestionRequest(BaseModel):
    article_ref: str = Field(min_length=1, max_length=191)
    content: str = Field(min_length=1, max_length=20000)
    catalog: list[CanonicalConceptIn] = Field(min_length=1)
    policy: WikiSuggestionPolicyIn = Field(default_factory=WikiSuggestionPolicyIn)

    @field_validator("article_ref", "content")
    @classmethod
    def strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            raise ValueError("must not be blank")
        return stripped


class WikiSuggestionOut(BaseModel):
    ref: str
    term: str
    url: str
    score: float
    evidence: str


class WikiSuggestionResponse(BaseModel):
    article_ref: str
    suggestions: list[WikiSuggestionOut]
    rejected_terms: list[str]
