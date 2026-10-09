from __future__ import annotations

from typing import Sequence

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.cta_planner.contracts import CtaPlanRequest, CtaSectionIn, LegacyCandidateIn
from app.modules.cta_planner.planner import (
    INFO_PROTOTYPES,
    INTENT_PROTOTYPES,
    PROMO_PROTOTYPES,
    CtaPlanner,
)


class MappingEmbeddingProvider:
    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self._mapping = {normalize_text(key): value for key, value in mapping.items()}

    @property
    def provider_key(self) -> str:
        return "fake"

    @property
    def model_key(self) -> str:
        return "fake-cta"

    @property
    def model_version(self) -> str:
        return "v0"

    @property
    def dimensions(self) -> int | None:
        return 2

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        out: list[EmbeddingResult] = []
        for raw in texts:
            vector = self._mapping.get(normalize_text(raw), (0.0, 0.0))
            out.append(
                EmbeddingResult(
                    vector=vector,
                    model_key=self.model_key,
                    model_version=self.model_version,
                    dimensions=2,
                    provider=self.provider_key,
                )
            )
        return out


def _anchors() -> dict[str, tuple[float, ...]]:
    mapping: dict[str, tuple[float, ...]] = {}
    intent_vectors = {
        "product_discovery": (1.0, 0.0),
        "service_discovery": (0.8, 0.2),
        "comparison": (0.6, 0.4),
        "consultation": (0.0, 1.0),
        "conversion": (0.2, 0.9),
    }
    for intent, examples in INTENT_PROTOTYPES.items():
        for example in examples:
            mapping[example] = intent_vectors[intent]
    for example in PROMO_PROTOTYPES:
        mapping[example] = (0.95, 0.1)
    for example in INFO_PROTOTYPES:
        mapping[example] = (-1.0, 0.0)
    return mapping


def _section(section_id: str, content: str, start: int, words: int, ratio: float) -> CtaSectionIn:
    return CtaSectionIn(
        section_id=section_id,
        heading=section_id,
        position_ratio=ratio,
        start_word=start,
        word_count=words,
        content=content,
    )


def test_distribution_skips_clustered_and_unsuitable_sections() -> None:
    mapping = _anchors()
    mapping["product section alpha materials"] = (1.0, 0.0)
    mapping["product section beta adjacent"] = (0.99, 0.01)
    mapping["educational history of nylon"] = (-1.0, 0.0)
    mapping["consultation on size and fit"] = (0.0, 1.0)
    mapping["product section later in article"] = (0.98, 0.02)
    planner = CtaPlanner(MappingEmbeddingProvider(mapping))
    result = planner.plan(
        CtaPlanRequest(
            language="en",
            article_word_count=1600,
            sections=[
                _section("section_1", "product section alpha materials", 0, 300, 0.1),
                _section("section_2", "product section beta adjacent", 40, 280, 0.2),
                _section("section_3", "educational history of nylon", 400, 320, 0.45),
                _section("section_4", "consultation on size and fit", 800, 300, 0.7),
                _section("section_5", "product section later in article", 1200, 300, 0.9),
            ],
        )
    )
    ids = [row.section_id for row in result.placements]
    assert "section_2" not in ids
    assert "section_3" not in ids
    assert "section_1" in ids
    assert "section_4" in ids
    assert len(result.placements) >= 2
    intents = {row.section_id: row.intent for row in result.placements}
    assert intents["section_1"] == "product_discovery"
    assert intents["section_4"] == "consultation"


def test_no_suitable_section_returns_zero_placements() -> None:
    mapping = _anchors()
    mapping["pure educational paragraph one"] = (-1.0, 0.0)
    mapping["pure educational paragraph two"] = (-0.9, 0.05)
    planner = CtaPlanner(MappingEmbeddingProvider(mapping))
    result = planner.plan(
        CtaPlanRequest(
            language="vi",
            article_word_count=800,
            sections=[
                _section("section_1", "pure educational paragraph one", 0, 200, 0.2),
                _section("section_2", "pure educational paragraph two", 400, 200, 0.7),
            ],
        )
    )
    assert result.placements == []


def test_mixed_legacy_candidate_stays_uncertain() -> None:
    mapping = _anchors()
    mapping["useful fabric facts and please contact us"] = (0.95, 0.1)
    planner = CtaPlanner(MappingEmbeddingProvider(mapping))
    result = planner.plan(
        CtaPlanRequest(
            language="en",
            article_word_count=400,
            sections=[],
            legacy_candidates=[
                LegacyCandidateIn(
                    candidate_id="legacy_1",
                    section_id="section_1",
                    text="useful fabric facts and please contact us",
                    structural_signal="mixed_paragraph",
                )
            ],
        )
    )
    assert result.legacy[0].classification == "uncertain"


def test_standalone_promo_is_promotional() -> None:
    mapping = _anchors()
    mapping["contact us via zalo facebook email or visit the showroom"] = (0.95, 0.1)
    planner = CtaPlanner(MappingEmbeddingProvider(mapping))
    result = planner.plan(
        CtaPlanRequest(
            language="vi",
            article_word_count=400,
            legacy_candidates=[
                LegacyCandidateIn(
                    candidate_id="legacy_2",
                    section_id="section_1",
                    text="contact us via zalo facebook email or visit the showroom",
                    structural_signal="standalone_blockquote",
                )
            ],
        )
    )
    assert result.legacy[0].classification == "promotional"
    assert result.legacy[0].confidence > 0
