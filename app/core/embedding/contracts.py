from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Runtime-neutral embedding payload. Never expose ONNX/library objects here."""

    vector: tuple[float, ...]
    model_key: str
    model_version: str
    dimensions: int
    provider: str


class EmbeddingProvider(Protocol):
    @property
    def provider_key(self) -> str: ...

    @property
    def model_key(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def dimensions(self) -> int | None: ...

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def embed(self, text: str) -> EmbeddingResult: ...

    def embed_batch(self, texts: Sequence[str]) -> list[EmbeddingResult]: ...
