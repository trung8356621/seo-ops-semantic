from __future__ import annotations

import hashlib
import math
import os
import threading
from array import array
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
        self._embed_lock = threading.Lock()
        self._memory: dict[str, tuple[float, ...]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

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

    def release(self) -> None:
        self._model = None

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

        with self._embed_lock:
            ordered: list[tuple[float, ...] | None] = []
            missing: list[str] = []
            missing_at: list[int] = []
            for normalized in prepared:
                cached = self._cached_vector(normalized)
                if cached is None:
                    missing.append(normalized)
                    missing_at.append(len(ordered))
                    ordered.append(None)
                    continue
                self.cache_hits += 1
                ordered.append(cached)

            if missing:
                self.cache_misses += len(missing)
                vectors = list(self._model.embed(missing))
                if len(vectors) != len(missing):
                    raise RuntimeError("embedding batch size mismatch")
                for slot, normalized, vector in zip(missing_at, missing, vectors, strict=True):
                    floats = self._finite_vector(vector)
                    self._store_vector(normalized, floats)
                    ordered[slot] = floats

        out: list[EmbeddingResult] = []
        for vector in ordered:
            assert vector is not None
            out.append(
                EmbeddingResult(
                    vector=vector,
                    model_key=self._model_name,
                    model_version=self.MODEL_VERSION,
                    dimensions=self._dimensions,
                    provider=self.PROVIDER_KEY,
                )
            )
        return out

    def _cache_digest(self, normalized: str) -> str:
        material = f"{self._model_name}\0{self.MODEL_VERSION}\0{self._dimensions}\0{normalized}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _cache_file(self, digest: str) -> Path:
        directory = Path(self._cache_dir) / "text-embeddings" / self.MODEL_VERSION
        directory.mkdir(parents=True, exist_ok=True)
        return directory / digest

    def _cached_vector(self, normalized: str) -> tuple[float, ...] | None:
        digest = self._cache_digest(normalized)
        memory = self._memory.get(digest)
        if memory is not None:
            return memory
        path = self._cache_file(digest)
        if not path.is_file():
            return None
        raw = path.read_bytes()
        width = (self._dimensions or 0) * 8
        if width <= 0 or len(raw) != width:
            return None
        values = array("d")
        values.frombytes(raw)
        vector = tuple(float(v) for v in values)
        self._memory[digest] = vector
        return vector

    def _store_vector(self, normalized: str, vector: tuple[float, ...]) -> None:
        digest = self._cache_digest(normalized)
        self._memory[digest] = vector
        path = self._cache_file(digest)
        temporary = path.with_suffix(".tmp")
        payload = array("d", vector)
        temporary.write_bytes(payload.tobytes())
        os.replace(temporary, path)

    def _finite_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        floats = tuple(float(x) for x in vector)
        if len(floats) != self._dimensions:
            raise RuntimeError(
                f"embedding dimension mismatch: got {len(floats)}, expected {self._dimensions}"
            )
        if any(not math.isfinite(v) for v in floats):
            raise RuntimeError("embedding contains non-finite values")
        return floats
