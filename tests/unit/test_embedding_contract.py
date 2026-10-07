from __future__ import annotations

import math
from typing import Sequence

import pytest

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text


class FakeEmbeddingProvider:
    PROVIDER_KEY = "fake"
    MODEL_KEY = "fake-model"
    MODEL_VERSION = "v0"
    DIMENSIONS = 4

    def __init__(self) -> None:
        self._loaded = False

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
        if len(texts) == 0:
            return []
        self.load()
        out: list[EmbeddingResult] = []
        for index, raw in enumerate(texts):
            normalized = normalize_text(str(raw))
            if normalized == "":
                raise ValueError(f"texts[{index}] is empty after normalization")
            seed = float(len(normalized) % 7)
            vector = (seed, seed + 1.0, seed + 2.0, seed + 3.0)
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


def test_embed_shape() -> None:
    provider = FakeEmbeddingProvider()
    result = provider.embed("balo học sinh")
    assert result.dimensions == 4
    assert len(result.vector) == 4
    assert result.model_key == "fake-model"
    assert result.model_version == "v0"
    assert result.provider == "fake"
    assert all(math.isfinite(v) for v in result.vector)


def test_batch_order_preserved() -> None:
    provider = FakeEmbeddingProvider()
    texts = ["một", "hai", "ba"]
    results = provider.embed_batch(texts)
    assert [r.vector[0] for r in results] == [
        provider.embed(t).vector[0] for t in texts
    ]


def test_empty_batch_returns_empty() -> None:
    assert FakeEmbeddingProvider().embed_batch([]) == []


def test_empty_text_rejected() -> None:
    with pytest.raises(ValueError):
        FakeEmbeddingProvider().embed("   ")
