from __future__ import annotations

from typing import Sequence

import pytest

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.concept_matching.analyzer import ConceptMatchAnalyzer
from app.modules.concept_matching.contracts import ConceptMatchAnalysisRequest


class MappingEmbeddingProvider:
    PROVIDER_KEY = "fake"
    MODEL_KEY = "fake-hybrid"
    MODEL_VERSION = "v0"
    DIMENSIONS = 3

    def __init__(self, mapping: dict[str, tuple[float, ...]] | None = None) -> None:
        self._mapping = {normalize_text(k): v for k, v in (mapping or {}).items()}
        self._loaded = False
        self.batch_calls = 0
        self.texts_seen: list[list[str]] = []

    @property
    def provider_key(self) -> str:
        return self.PROVIDER_KEY

    @property
    def model_key(self) -> str:
        return self.MODEL_KEY

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def dimensions(self) -> int | None:
        return self.DIMENSIONS if self._loaded else self.DIMENSIONS

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        self.load()
        self.batch_calls += 1
        self.texts_seen.append(list(texts))
        out: list[EmbeddingResult] = []
        for raw in texts:
            key = normalize_text(str(raw))
            vector = self._mapping.get(key)
            if vector is None:
                raise KeyError(f"missing vector for {key!r}")
            out.append(
                EmbeddingResult(
                    vector=vector,
                    model_key=self.MODEL_KEY,
                    model_version=self.MODEL_VERSION,
                    dimensions=self.DIMENSIONS,
                    provider=self.PROVIDER_KEY,
                )
            )
        return out


def test_semantic_only_backward_compatibility() -> None:
    provider = MappingEmbeddingProvider(
        {
            "entity": (1.0, 0.0, 0.0),
            "pos": (1.0, 0.0, 0.0),
        }
    )
    response = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                "scope_ref": "site:1",
                "entities": [{"ref": "e1", "text": "entity"}],
                "concepts": [{"key": "c1", "positive_examples": ["pos"]}],
            }
        )
    )
    score = response.entities[0].concepts[0]
    assert score.matching_strategy == "semantic"
    assert score.lexical.match_mode == "phrase"
    assert score.positive_max == pytest.approx(1.0)
    assert score.suggested_match is None
    assert provider.batch_calls == 1


def test_lexical_only_skips_embedding() -> None:
    provider = MappingEmbeddingProvider()
    response = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                "scope_ref": "site:1",
                "entities": [
                    {"ref": "e1", "text": "balo học sinh cấp 1"},
                    {"ref": "e2", "text": "Zalo 0909983833"},
                ],
                "concepts": [
                    {
                        "key": "industry.products.balo.test",
                        "positive_examples": ["balo", "ba lô", "backpack"],
                        "match_mode": "token",
                        "matching_strategy": "lexical",
                    }
                ],
                "decision_policy": {"min_positive_score": 0.99},
            }
        )
    )
    assert provider.batch_calls == 0
    assert response.diagnostics.unique_text_count == 0
    assert response.diagnostics.lexical_only_concepts == 1
    assert response.diagnostics.semantic_concepts == 0

    a, b = response.entities[0].concepts[0], response.entities[1].concepts[0]
    assert a.lexical.matched is True
    assert a.positive_max is None
    assert a.suggested_match is True
    assert b.lexical.matched is False
    assert b.suggested_match is False


def test_hybrid_lexical_wins_without_semantic_fallback() -> None:
    # Semantic would be weak/orthogonal; lexical still decides match.
    provider = MappingEmbeddingProvider(
        {
            "balo học sinh cấp 1": (0.0, 1.0, 0.0),
            "Zalo 0909983833": (0.0, 1.0, 0.0),
            "balo": (1.0, 0.0, 0.0),
            "ba lô": (1.0, 0.0, 0.0),
            "backpack": (1.0, 0.0, 0.0),
        }
    )
    response = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                "scope_ref": "site:1",
                "entities": [
                    {"ref": "e1", "text": "balo học sinh cấp 1"},
                    {"ref": "e2", "text": "Zalo 0909983833"},
                ],
                "concepts": [
                    {
                        "key": "industry.products.balo.test",
                        "positive_examples": ["balo", "ba lô", "backpack"],
                        "match_mode": "token",
                        "matching_strategy": "hybrid",
                    }
                ],
                "decision_policy": {
                    "min_positive_score": 0.9,
                    "semantic_fallback": False,
                },
            }
        )
    )
    assert provider.batch_calls == 1
    a, b = response.entities[0].concepts[0], response.entities[1].concepts[0]
    assert a.lexical.matched is True
    assert a.suggested_match is True
    assert a.positive_max is not None  # semantic still returned for calibration
    assert b.lexical.matched is False
    assert b.suggested_match is False


def test_hybrid_semantic_fallback_off_and_on() -> None:
    provider = MappingEmbeddingProvider(
        {
            "synonym only phrase": (1.0, 0.0, 0.0),
            "canonical": (0.95, 0.05, 0.0),
        }
    )
    base = {
        "scope_ref": "site:1",
        "entities": [{"ref": "e1", "text": "synonym only phrase"}],
        "concepts": [
            {
                "key": "c1",
                "positive_examples": ["canonical"],
                "match_mode": "token",
                "matching_strategy": "hybrid",
            }
        ],
    }
    off = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                **base,
                "decision_policy": {
                    "min_positive_score": 0.5,
                    "semantic_fallback": False,
                },
            }
        )
    )
    assert off.entities[0].concepts[0].lexical.matched is False
    assert off.entities[0].concepts[0].suggested_match is False

    on = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                **base,
                "decision_policy": {
                    "min_positive_score": 0.5,
                    "semantic_fallback": True,
                },
            }
        )
    )
    assert on.entities[0].concepts[0].suggested_match is True


def test_no_hard_coded_global_threshold_in_module() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "app" / "modules" / "concept_matching"
    forbidden = ("0.7", "0.8", "0.5", "0.48", "0.49")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            # Allow comments mentioning calibration history, but not assignments as defaults.
            assert f"= {token}" not in text
            assert f"={token}" not in text


def test_industry_product_calibration_lexical() -> None:
    provider = MappingEmbeddingProvider(
        {
            "balo học sinh cấp 1": (0.5, 0.5, 0.0),
            "xưởng sản xuất balo theo yêu cầu": (0.5, 0.5, 0.0),
            "Zalo 0909983833": (0.5, 0.5, 0.0),
            "balo": (1.0, 0.0, 0.0),
            "ba lô": (1.0, 0.0, 0.0),
            "backpack": (1.0, 0.0, 0.0),
        }
    )
    response = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                "scope_ref": "calib:industry-product",
                "entities": [
                    {"ref": "p1", "text": "balo học sinh cấp 1"},
                    {"ref": "p2", "text": "xưởng sản xuất balo theo yêu cầu"},
                    {"ref": "p3", "text": "Zalo 0909983833"},
                ],
                "concepts": [
                    {
                        "key": "industry.products.balo.test",
                        "positive_examples": ["balo", "ba lô", "backpack"],
                        "match_mode": "token",
                        "matching_strategy": "hybrid",
                    }
                ],
            }
        )
    )
    matched = [e.concepts[0].lexical.matched for e in response.entities]
    assert matched == [True, True, False]


def test_industry_material_hybrid_returns_both_evidences() -> None:
    provider = MappingEmbeddingProvider(
        {
            "vải canvas chống thấm": (1.0, 0.0, 0.0),
            "vải canvas": (0.95, 0.05, 0.0),
            "canvas": (0.9, 0.1, 0.0),
            "vải bố": (0.2, 0.8, 0.0),
        }
    )
    score = (
        ConceptMatchAnalyzer(provider)
        .analyze(
            ConceptMatchAnalysisRequest.model_validate(
                {
                    "scope_ref": "calib:material",
                    "entities": [{"ref": "m1", "text": "vải canvas chống thấm"}],
                    "concepts": [
                        {
                            "key": "industry.materials.canvas.test",
                            "positive_examples": ["vải canvas", "canvas", "vải bố"],
                            "match_mode": "phrase",
                            "matching_strategy": "hybrid",
                        }
                    ],
                }
            )
        )
        .entities[0]
        .concepts[0]
    )
    assert score.lexical.matched is True
    assert score.positive_max is not None
    assert score.best_positive_example is not None


def test_batch_embedding_preserved_for_hybrid() -> None:
    provider = MappingEmbeddingProvider(
        {
            "a": (1.0, 0.0, 0.0),
            "b": (0.0, 1.0, 0.0),
            "pos": (1.0, 0.0, 0.0),
        }
    )
    ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                "scope_ref": "site:1",
                "entities": [{"ref": "e1", "text": "a"}, {"ref": "e2", "text": "b"}],
                "concepts": [
                    {
                        "key": "c1",
                        "positive_examples": ["pos"],
                        "matching_strategy": "hybrid",
                        "match_mode": "token",
                    }
                ],
            }
        )
    )
    assert provider.batch_calls == 1
    assert len(provider.texts_seen[0]) == 3
