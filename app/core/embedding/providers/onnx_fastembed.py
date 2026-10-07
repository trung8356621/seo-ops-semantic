from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

from app.core.embedding.contracts import EmbeddingResult
from app.core.text.normalization import normalize_text


class OnnxFastEmbedProvider:
    """ONNX Runtime via FastEmbed — CPU-only, no PyTorch stack."""

    PROVIDER_KEY = "onnx_fastembed"
    # FastEmbed pin of this model family; kept explicit for diagnostics.
    MODEL_VERSION = "fastembed-onnx-q"

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        *,
        lazy_load: bool = True,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._lazy_load = lazy_load
        self._model = None
        self._dimensions: int | None = None

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path(cache_dir) / "fastembed"))
        os.environ.setdefault("HF_HOME", str(Path(cache_dir) / "hf"))

        if not lazy_load:
            self.load()

    @property
    def provider_key(self) -> str:
        return self.PROVIDER_KEY

    @property
    def model_key(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from fastembed import TextEmbedding

        cache = str(Path(self._cache_dir) / "fastembed")
        Path(cache).mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=self._model_name, cache_dir=cache)
        # Probe dimensions once without logging the vector.
        probe = list(self._model.embed(["dimension probe"]))
        if not probe:
            raise RuntimeError("embedding model returned empty probe")
        self._dimensions = int(len(probe[0]))

    def embed(self, text: str) -> EmbeddingResult:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        if texts is None:
            raise TypeError("texts must be a sequence")
        if len(texts) == 0:
            return []

        prepared: list[str] = []
        for index, raw in enumerate(texts):
            if raw is None:
                raise ValueError(f"texts[{index}] is None")
            normalized = normalize_text(str(raw))
            if normalized == "":
                raise ValueError(f"texts[{index}] is empty after normalization")
            prepared.append(normalized)

        self.load()
        assert self._model is not None
        assert self._dimensions is not None

        vectors = list(self._model.embed(prepared))
        if len(vectors) != len(prepared):
            raise RuntimeError("embedding batch size mismatch")

        out: list[EmbeddingResult] = []
        for vector in vectors:
            floats = tuple(float(x) for x in vector)
            if len(floats) != self._dimensions:
                raise RuntimeError(
                    f"embedding dimension mismatch: got {len(floats)}, expected {self._dimensions}"
                )
            if any(not math.isfinite(v) for v in floats):
                raise RuntimeError("embedding contains non-finite values")
            out.append(
                EmbeddingResult(
                    vector=floats,
                    model_key=self._model_name,
                    model_version=self.MODEL_VERSION,
                    dimensions=self._dimensions,
                    provider=self.PROVIDER_KEY,
                )
            )
        return out
