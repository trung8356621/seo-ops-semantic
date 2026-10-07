from __future__ import annotations

from typing import Sequence

import pytest

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.concept_matching.analyzer import ConceptMatchAnalyzer
from app.modules.concept_matching.contracts import ConceptMatchAnalysisRequest


class MappingEmbeddingProvider:
    """Deterministic vectors keyed by normalized text. Counts embed_batch calls."""

    PROVIDER_KEY = "fake"
    MODEL_KEY = "fake-concept"
    MODEL_VERSION = "v0"
    DIMENSIONS = 3

    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self._mapping = {normalize_text(k): v for k, v in mapping.items()}
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
        return self.DIMENSIONS if self._loaded else None

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


def test_batch_embedding_and_response_shape() -> None:
    provider = MappingEmbeddingProvider(
        {
            "tuyển nhân viên may balo": (1.0, 0.0, 0.0),
            "xưởng sản xuất balo": (0.0, 1.0, 0.0),
            "tuyển thợ may": (0.95, 0.05, 0.0),
            "việc làm xưởng may": (0.9, 0.1, 0.0),
            "xưởng may balo": (0.1, 0.9, 0.0),
        }
    )
    request = ConceptMatchAnalysisRequest.model_validate(
        {
            "scope_ref": "site:4",
            "language": "vi",
            "entities": [
                {"ref": "a", "text": "tuyển nhân viên may balo"},
                {"ref": "b", "text": "xưởng sản xuất balo"},
            ],
            "concepts": [
                {
                    "key": "custom.recruitment",
                    "positive_examples": ["tuyển thợ may", "việc làm xưởng may"],
                    "negative_examples": ["xưởng may balo"],
                }
            ],
        }
    )
    response = ConceptMatchAnalyzer(provider).analyze(request)

    assert provider.batch_calls == 1
    assert response.scope_ref == "site:4"
    assert response.language == "vi"
    assert response.analysis_id
    assert len(response.entities) == 2
    assert response.diagnostics.cache == "direct_embed_batch"
    assert response.diagnostics.unique_text_count == 5
    assert response.diagnostics.model == "fake-concept"

    a = response.entities[0].concepts[0]
    b = response.entities[1].concepts[0]
    assert a.key == "custom.recruitment"
    assert a.suggested_match is None
    assert a.positive_max > b.positive_max
    assert a.margin is not None
    assert b.margin is not None


def test_decision_policy_null_vs_set() -> None:
    provider = MappingEmbeddingProvider(
        {
            "entity": (1.0, 0.0, 0.0),
            "pos": (1.0, 0.0, 0.0),
        }
    )
    base = {
        "scope_ref": "site:1",
        "entities": [{"ref": "e1", "text": "entity"}],
        "concepts": [{"key": "c1", "positive_examples": ["pos"]}],
    }
    without = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(base)
    )
    assert without.entities[0].concepts[0].suggested_match is None

    with_policy = ConceptMatchAnalyzer(provider).analyze(
        ConceptMatchAnalysisRequest.model_validate(
            {
                **base,
                "decision_policy": {"min_positive_score": 0.7, "min_margin": 0.05},
            }
        )
    )
    assert with_policy.entities[0].concepts[0].suggested_match is True


def test_duplicate_texts_embedded_once() -> None:
    provider = MappingEmbeddingProvider(
        {
            "same text": (1.0, 0.0, 0.0),
            "pos": (1.0, 0.0, 0.0),
        }
    )
    request = ConceptMatchAnalysisRequest.model_validate(
        {
            "scope_ref": "site:1",
            "entities": [
                {"ref": "e1", "text": "same text"},
                {"ref": "e2", "text": "same text"},
            ],
            "concepts": [{"key": "c1", "positive_examples": ["pos", "same text"]}],
        }
    )
    ConceptMatchAnalyzer(provider).analyze(request)
    assert provider.batch_calls == 1
    assert len(provider.texts_seen[0]) == 2  # "same text", "pos"
