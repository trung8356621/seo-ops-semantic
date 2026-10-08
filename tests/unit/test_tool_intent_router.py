from __future__ import annotations

from typing import Sequence

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text
from app.modules.tool_intent.contracts import ToolIntentDefinitionIn, ToolIntentMatchRequest, ToolIntentPolicyIn
from app.modules.tool_intent.router import ToolIntentRouter


class MappingEmbeddingProvider:
    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self._mapping = {normalize_text(key): value for key, value in mapping.items()}
        self._loaded = True

    @property
    def provider_key(self) -> str:
        return "fake"

    @property
    def model_key(self) -> str:
        return "fake-tool"

    @property
    def model_version(self) -> str:
        return "v0"

    @property
    def dimensions(self) -> int | None:
        return 2

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        out: list[EmbeddingResult] = []
        for raw in texts:
            vector = self._mapping.get(normalize_text(raw), (0.0, 1.0))
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


def _request(query: str, intents: list[tuple[str, list[str]]]) -> ToolIntentMatchRequest:
    return ToolIntentMatchRequest(
        query=query,
        intents=[
            ToolIntentDefinitionIn(key=key, positive_examples=examples)
            for key, examples in intents
        ],
        policy=ToolIntentPolicyIn(min_positive_score=0.62, min_margin=0.08),
    )


def test_obvious_lexical_intent_is_confident() -> None:
    router = ToolIntentRouter(MappingEmbeddingProvider({}))
    result = router.match(
        _request(
            "tìm bài SEO kém",
            [
                ("seo_audit.worst_articles", ["tìm bài seo kém", "bài nào cần tối ưu"]),
                ("gsc.performance", ["traffic tháng này", "click gsc"]),
            ],
        )
    )
    assert result.status == "confident"
    assert result.matches[0].ref == "seo_audit.worst_articles"
    assert result.matches[0].evidence.lexical_matched is True
    assert result.namespace == "agent_tool_intents"


def test_close_semantic_scores_are_ambiguous() -> None:
    router = ToolIntentRouter(
        MappingEmbeddingProvider(
            {
                "câu hỏi mơ hồ": (1.0, 0.0),
                "ví dụ audit": (0.9, 0.1),
                "ví dụ gsc": (0.88, 0.12),
            }
        )
    )
    result = router.match(
        _request(
            "câu hỏi mơ hồ",
            [
                ("seo_audit.worst_articles", ["ví dụ audit"]),
                ("gsc.performance", ["ví dụ gsc"]),
            ],
        )
    )
    assert result.status == "ambiguous"
    assert {item.ref for item in result.matches[:2]} == {
        "seo_audit.worst_articles",
        "gsc.performance",
    }


def test_low_scores_are_no_match() -> None:
    router = ToolIntentRouter(
        MappingEmbeddingProvider(
            {
                "thời tiết hôm nay": (1.0, 0.0),
                "tìm bài seo kém": (0.0, 1.0),
                "traffic tháng này": (0.0, 1.0),
            }
        )
    )
    result = router.match(
        _request(
            "thời tiết hôm nay",
            [
                ("seo_audit.worst_articles", ["tìm bài seo kém"]),
                ("gsc.performance", ["traffic tháng này"]),
            ],
        )
    )
    assert result.status == "none"
