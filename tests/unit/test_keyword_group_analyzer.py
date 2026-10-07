from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from app.config import Settings
from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.core.vector.contracts import VectorRecord
from app.modules.keyword_grouping.analyzer import (
    KeywordGroupAnalyzer,
    content_hash,
    compute_input_hash,
    grouping_namespace,
    model_cache_key,
)
from app.modules.keyword_grouping.contracts import KeywordGroupAnalysisRequest, KeywordIn


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
            if "balo" in normalized:
                seed = np.array([1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                if "học" in normalized or "sinh" in normalized:
                    seed[1] += 0.15
            elif "túi" in normalized or "tui" in normalized:
                seed = np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif "giặt" in normalized or "giat" in normalized:
                seed = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            elif "vali" in normalized:
                seed = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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


def test_post_style_analyze_returns_representative_and_no_suggested_label() -> None:
    analyzer = KeywordGroupAnalyzer(settings=_settings(), embedding=FakeEmbedding())
    result = analyzer.analyze(
        KeywordGroupAnalysisRequest(
            scope_ref="site-4",
            language="vi",
            keywords=[
                KeywordIn(ref="1", text="balo học sinh"),
                KeywordIn(ref="2", text="balo cho học sinh"),
                KeywordIn(ref="99", text="vali kéo du lịch"),
            ],
        )
    )
    assert result.status == "completed"
    assert result.scope_ref == "site-4"
    assert result.language == "vi"
    assert result.diagnostics.algorithm == "hybrid_semantic_lexical_v1"
    dumped = result.model_dump()
    assert "suggested_label" not in dumped
    assert "site_ref" not in dumped
    assert "topic_id" not in dumped

    assert len(result.groups) >= 1
    group = result.groups[0]
    assert {m.ref for m in group.members} == {"1", "2"}
    assert group.representative_ref
    assert group.representative_text
    assert group.representative_text in {m.text for m in group.members}
    assert any(m.is_representative for m in group.members)
    for member in group.members:
        assert member.ref
        assert member.text
        assert 0.0 <= member.similarity_score <= 1.0
        assert "keyword_ref" not in member.model_dump()
        assert "confidence" not in member.model_dump()

    assert any(u.ref == "99" for u in result.unassigned)
    assert all(u.reason for u in result.unassigned)


def test_duplicate_refs_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        KeywordGroupAnalysisRequest(
            scope_ref="site-4",
            keywords=[
                KeywordIn(ref="1", text="balo"),
                KeywordIn(ref="1", text="túi"),
            ],
        )


def test_bad_input_hash_rejected() -> None:
    analyzer = KeywordGroupAnalyzer(
        settings=_settings(TOPIC_EMBEDDING_CACHE_ENABLED=False),
        embedding=FakeEmbedding(),
    )
    with pytest.raises(ValueError, match="input_hash mismatch"):
        analyzer.analyze(
            KeywordGroupAnalysisRequest(
                scope_ref="site-4",
                keywords=[KeywordIn(ref="1", text="balo học sinh")],
                input_hash="deadbeef",
            )
        )


def test_correct_input_hash_accepted() -> None:
    keywords = [
        KeywordIn(ref="1", text="balo học sinh"),
        KeywordIn(ref="2", text="balo sinh viên"),
    ]
    prepared_hash = compute_input_hash("site-4", "vi", keywords)
    analyzer = KeywordGroupAnalyzer(
        settings=_settings(TOPIC_EMBEDDING_CACHE_ENABLED=False),
        embedding=FakeEmbedding(),
    )
    result = analyzer.analyze(
        KeywordGroupAnalysisRequest(
            scope_ref="site-4",
            language="vi",
            keywords=keywords,
            input_hash=prepared_hash,
        )
    )
    assert result.input_hash == prepared_hash


def test_embedding_cache_reused_under_keyword_group_namespace() -> None:
    store = MemoryVectorStore()
    embedding = FakeEmbedding()
    analyzer = KeywordGroupAnalyzer(settings=_settings(), embedding=embedding, vectors=store)
    req = KeywordGroupAnalysisRequest(
        scope_ref="site-9",
        keywords=[KeywordIn(ref="kw-1", text="balo học sinh")],
    )
    first = analyzer.analyze(req)
    assert embedding.calls == 1
    assert first.diagnostics.embedding_cache["misses"] == 1

    second = analyzer.analyze(req)
    assert embedding.calls == 1
    assert second.diagnostics.embedding_cache["hits"] == 1
    assert second.diagnostics.embedding_cache["misses"] == 0

    key = model_cache_key(embedding)
    cached = store.get(grouping_namespace("site-9"), "kw-1", key)
    assert cached is not None
    assert cached.content_hash == content_hash("balo học sinh")
    assert "topic:" not in grouping_namespace("site-9")


def test_topic_endpoint_compat_analyzer_still_works() -> None:
    """Old Topic analyzer path remains functional (suggested_label intact)."""
    from app.modules.topic.analyzer import TopicAnalyzer
    from app.modules.topic.contracts import TopicAnalysisRequest, TopicKeywordIn

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
    assert result.site_ref == "6"
    for group in result.groups:
        assert group.suggested_label
        assert "representative_text" not in group.model_dump()
