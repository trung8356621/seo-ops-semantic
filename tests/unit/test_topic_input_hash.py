from __future__ import annotations

import pytest

from app.config import Settings
from app.core.clustering import ClusterPoint
from app.modules.topic.analyzer import TopicAnalyzer, compute_input_hash
from app.modules.topic.contracts import TopicAnalysisRequest, TopicKeywordIn


class _FakeEmbedding:
    provider_key = "fake"
    model_key = "fake-model"
    model_version = "v0"
    dimensions = 3

    def embed_batch(self, texts):  # noqa: ANN001
        from app.core.embedding.contracts import EmbeddingResult

        out = []
        for text in texts:
            # Deterministic tiny vectors from hash of text.
            seed = sum(ord(ch) for ch in text) % 97
            vec = (1.0, float(seed) / 100.0, 0.1)
            out.append(
                EmbeddingResult(
                    vector=vec,
                    dimensions=3,
                    model_key=self.model_key,
                    model_version=self.model_version,
                    provider=self.provider_key,
                )
            )
        return out

    def embed(self, text: str):  # noqa: ANN001
        return self.embed_batch([text])[0]

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None


def test_compute_hash_ignores_keyword_order() -> None:
    a = [
        TopicKeywordIn(ref="2", text="balo học sinh"),
        TopicKeywordIn(ref="1", text="túi xách"),
    ]
    b = list(reversed(a))
    assert compute_input_hash("4", None, a) == compute_input_hash("4", None, b)


def test_missing_hash_uses_computed() -> None:
    settings = Settings(
        topic_cluster_algorithm="average_linkage",
        topic_cluster_similarity_threshold=0.9,
        topic_assignment_min_score=0.9,
        topic_embedding_cache_enabled=False,
    )
    analyzer = TopicAnalyzer(settings=settings, embedding=_FakeEmbedding(), vectors=None, repository=None)
    req = TopicAnalysisRequest(
        site_ref="4",
        language=None,
        keywords=[
            TopicKeywordIn(ref="1", text="balo A"),
            TopicKeywordIn(ref="2", text="balo B"),
        ],
    )
    response = analyzer.analyze(req)
    expected = compute_input_hash("4", None, req.keywords)
    # analyzer normalizes first; recompute from normalized texts
    prepared = [
        TopicKeywordIn(ref="1", text="balo A"),
        TopicKeywordIn(ref="2", text="balo B"),
    ]
    expected = compute_input_hash("4", None, prepared)
    assert response.input_hash == expected


def test_wrong_hash_rejected() -> None:
    settings = Settings(
        topic_cluster_algorithm="average_linkage",
        topic_cluster_similarity_threshold=0.9,
        topic_assignment_min_score=0.9,
        topic_embedding_cache_enabled=False,
    )
    analyzer = TopicAnalyzer(settings=settings, embedding=_FakeEmbedding(), vectors=None, repository=None)
    req = TopicAnalysisRequest(
        site_ref="4",
        language=None,
        keywords=[TopicKeywordIn(ref="1", text="balo A")],
        input_hash="deadbeef",
    )
    with pytest.raises(ValueError, match="input_hash mismatch"):
        analyzer.analyze(req)


def test_correct_hash_accepted() -> None:
    settings = Settings(
        topic_cluster_algorithm="average_linkage",
        topic_cluster_similarity_threshold=0.9,
        topic_assignment_min_score=0.9,
        topic_embedding_cache_enabled=False,
    )
    analyzer = TopicAnalyzer(settings=settings, embedding=_FakeEmbedding(), vectors=None, repository=None)
    keywords = [TopicKeywordIn(ref="1", text="balo A"), TopicKeywordIn(ref="2", text="túi B")]
    prepared_hash = compute_input_hash("4", "vi", keywords)
    req = TopicAnalysisRequest(
        site_ref="4",
        language="vi",
        keywords=keywords,
        input_hash=prepared_hash,
    )
    response = analyzer.analyze(req)
    assert response.input_hash == prepared_hash
