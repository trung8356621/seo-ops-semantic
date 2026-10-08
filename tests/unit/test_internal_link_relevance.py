from __future__ import annotations

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.core.vector.contracts import VectorRecord
from app.modules.internal_link_v2.contracts import InternalLinkCandidateIn, InternalLinkRankRequest
from app.modules.internal_link_v2.relevance import apply_text_relevance


class FakeEmbed:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def provider_key(self) -> str:
        return "fake"

    @property
    def model_key(self) -> str:
        return "fake-article"

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
        self.calls.append(normalize_text(text))
        vector = (1.0, 0.0) if "cotton" in text.casefold() else (0.0, 1.0)
        return EmbeddingResult(vector=vector, model_key=self.model_key, model_version="v0", dimensions=2, provider="fake")

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


class MemoryStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], VectorRecord] = {}

    def upsert(self, record: VectorRecord) -> None:
        self.rows[(record.namespace, record.entity_ref, record.model_key)] = record

    def get(self, namespace: str, entity_ref: str, model_key: str) -> VectorRecord | None:
        return self.rows.get((namespace, entity_ref, model_key))

    def delete(self, namespace: str, entity_ref: str, model_key: str | None = None) -> int:
        return 0

    def nearest(self, **kwargs):
        return []


def test_published_target_is_not_reembedded_when_hash_matches() -> None:
    embed = FakeEmbed()
    store = MemoryStore()
    request = InternalLinkRankRequest(
        source_ref="article:1",
        source_text="áo thun cotton nam",
        candidate_boundary="topic_group",
        candidates=[
            InternalLinkCandidateIn(
                ref="article:2",
                topic_group_ref="tg:cotton",
                url="https://example.test/cotton",
                eligible=True,
                representation="áo thun cotton",
            )
        ],
    )
    first = apply_text_relevance(request, embed, store)
    second = apply_text_relevance(request, embed, store)
    assert first.candidates[0].relevance > 0
    assert second.candidates[0].relevance == first.candidates[0].relevance
    target_calls = [text for text in embed.calls if text == normalize_text("áo thun cotton")]
    assert target_calls == [normalize_text("áo thun cotton")]
    assert ("published_articles", "article:1", "fake-article") not in store.rows
