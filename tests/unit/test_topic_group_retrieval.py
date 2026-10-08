from __future__ import annotations

from typing import Sequence

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.topic_group_retrieval.contracts import (
    TopicGroupCandidateIn,
    TopicGroupMatchPolicyIn,
    TopicGroupMatchRequest,
)
from app.modules.topic_group_retrieval.matcher import TopicGroupMatcher


class MappingEmbeddingProvider:
    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self._mapping = {normalize_text(key): value for key, value in mapping.items()}

    @property
    def provider_key(self) -> str:
        return "fake"

    @property
    def model_key(self) -> str:
        return "fake-group"

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
            vector = self._mapping.get(normalize_text(raw), (0.0, 1.0))
            out.append(
                EmbeddingResult(vector=vector, model_key="fake-group", model_version="v0", dimensions=2, provider="fake")
            )
        return out


def test_topic_group_ranking_preserves_external_refs() -> None:
    matcher = TopicGroupMatcher(
        MappingEmbeddingProvider(
            {
                "áo thun cotton": (1.0, 0.0),
                "cotton t-shirt": (0.95, 0.05),
                "áo thun cotton nam": (0.9, 0.1),
                "máy lọc nước": (0.0, 1.0),
            }
        )
    )
    result = matcher.match(
        TopicGroupMatchRequest(
            scope_ref="site:ext-1",
            query="áo thun cotton nam",
            groups=[
                TopicGroupCandidateIn(ref="tg:cotton", label="Cotton", examples=["áo thun cotton", "cotton t-shirt"]),
                TopicGroupCandidateIn(ref="tg:water", label="Water", examples=["máy lọc nước"]),
            ],
            policy=TopicGroupMatchPolicyIn(min_score=0.55, limit=2),
        )
    )
    assert result.matches[0].ref == "tg:cotton"
    assert result.scope_ref == "site:ext-1"
    assert all(item.ref.startswith("tg:") for item in result.matches)
    assert "tg:water" not in [item.ref for item in result.matches]
