from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class VectorRecord:
    namespace: str
    entity_ref: str
    content_hash: str
    model_key: str
    embedding: Sequence[float]
    metadata: Mapping[str, Any]
    dimensions: int


@dataclass(frozen=True, slots=True)
class NearestNeighbor:
    namespace: str
    entity_ref: str
    model_key: str
    distance: float
    metadata: Mapping[str, Any]


class VectorStore(Protocol):
    def upsert(self, record: VectorRecord) -> None: ...

    def get(self, namespace: str, entity_ref: str, model_key: str) -> VectorRecord | None: ...

    def delete(self, namespace: str, entity_ref: str, model_key: str | None = None) -> int: ...

    def nearest(
        self,
        *,
        namespace: str,
        model_key: str,
        query: Sequence[float],
        limit: int = 10,
    ) -> list[NearestNeighbor]: ...
