from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from app.config import Settings
from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.core.vector.contracts import VectorRecord
from app.modules.topic.analyzer import TopicAnalyzer, content_hash, model_cache_key
from app.modules.topic.contracts import TopicAnalysisRequest, TopicKeywordIn


class FakeEmbedding:
    PROVIDER_KEY = "fake"
    MODEL_KEY = "fake-model"
    MODEL_VERSION = "v1"
    DIMENSIONS = 8

    def __init__(self) -> None:
        self._loaded = True
        self.calls = 0

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
        return self.DIMENSIONS

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        self.calls += 1
        out: list[EmbeddingResult] = []
        for text in texts:
            normalized = normalize_text(text).casefold()
            # Family centroids — orthogonal enough for threshold tests.
            if "balo" in normalized:
                seed = np.array([1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if "học" in normalized or "sinh" in normalized:
                    seed[1] += 0.15
                if "quà" in normalized or "tặng" in normalized:
                    seed[2] += 0.15
            elif "túi" in normalized or "tui" in normalized:
                seed = np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif "giặt" in normalized or "giat" in normalized:
                seed = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            else:
                seed = np.zeros(self.DIMENSIONS, dtype=np.float64)
                seed[hash(normalized) % self.DIMENSIONS] = 1.0
            norm = float(np.linalg.norm(seed)) or 1.0
            vector = tuple(float(x / norm) for x in seed)
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


class MemoryVectorStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], VectorRecord] = {}

    def upsert(self, record: VectorRecord) -> None:
        self.rows[(record.namespace, record.entity_ref, record.model_key)] = record

    def get(self, namespace: str, entity_ref: str, model_key: str) -> VectorRecord | None:
        return self.rows.get((namespace, entity_ref, model_key))

    def delete(self, namespace: str, entity_ref: str, model_key: str | None = None) -> int:
        return 0

    def nearest(self, **kwargs):  # noqa: ANN003
        return []


def _settings(**overrides: object) -> Settings:
    base = {
        "TOPIC_CLUSTER_SIMILARITY_THRESHOLD": 0.55,
        "TOPIC_MIN_MEMBER_SIMILARITY": 0.50,
        "TOPIC_ASSIGNMENT_MIN_SCORE": 0.50,
        "TOPIC_MIN_GROUP_SIZE": 2,
        "TOPIC_EMBEDDING_CACHE_ENABLED": True,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_analyzer_groups_similar_phrases_and_keeps_outlier() -> None:
    analyzer = TopicAnalyzer(settings=_settings(), embedding=FakeEmbedding())
    result = analyzer.analyze(
        TopicAnalysisRequest(
            site_ref="6",
            language="vi",
            keywords=[
                TopicKeywordIn(ref="1", text="balo học sinh"),
                TopicKeywordIn(ref="2", text="balo sinh viên"),
                TopicKeywordIn(ref="3", text="cách giặt áo"),
            ],
        )
    )
    assert result.status == "completed"
    member_refs = {m.keyword_ref for g in result.groups for m in g.members}
    assert "1" in member_refs and "2" in member_refs
    assert any(u.keyword_ref == "3" for u in result.unassigned) or "3" not in member_refs
    # labels are deterministic medoid texts
    for group in result.groups:
        assert group.suggested_label in {m.text for m in group.members}
        assert all(0.0 <= m.similarity_score <= 1.0 for m in group.members)
        assert all(0.0 <= m.confidence <= 1.0 for m in group.members)


def test_analyzer_deterministic() -> None:
    analyzer = TopicAnalyzer(settings=_settings(), embedding=FakeEmbedding())
    req = TopicAnalysisRequest(
        site_ref="6",
        keywords=[
            TopicKeywordIn(ref="b", text="balo quà tặng"),
            TopicKeywordIn(ref="a", text="balo quà tặng doanh nghiệp"),
            TopicKeywordIn(ref="c", text="túi xách nữ"),
        ],
    )
    first = analyzer.analyze(req)
    second = analyzer.analyze(req)
    assert [g.suggested_label for g in first.groups] == [g.suggested_label for g in second.groups]
    assert [tuple(m.keyword_ref for m in g.members) for g in first.groups] == [
        tuple(m.keyword_ref for m in g.members) for g in second.groups
    ]


def test_embedding_cache_hit_and_invalidate_on_text_change() -> None:
    store = MemoryVectorStore()
    embedding = FakeEmbedding()
    analyzer = TopicAnalyzer(settings=_settings(), embedding=embedding, vectors=store)
    req = TopicAnalysisRequest(
        site_ref="9",
        keywords=[TopicKeywordIn(ref="kw-1", text="balo học sinh")],
    )
    analyzer.analyze(req)
    assert embedding.calls == 1
    analyzer.analyze(req)
    assert embedding.calls == 1  # cached

    key = model_cache_key(embedding)
    cached = store.get("topic:9", "kw-1", key)
    assert cached is not None
    assert cached.content_hash == content_hash("balo học sinh")

    analyzer.analyze(
        TopicAnalysisRequest(
            site_ref="9",
            keywords=[TopicKeywordIn(ref="kw-1", text="balo học sinh cao cấp")],
        )
    )
    assert embedding.calls == 2


def test_empty_keyword_list_rejected() -> None:
    with pytest.raises(Exception):
        TopicAnalysisRequest(site_ref="6", keywords=[])
