from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pytest

from app.core.similarity import cosine_similarity
from app.core.vector.contracts import NearestNeighbor, VectorRecord


@dataclass
class InMemoryVectorStore:
    """Unit-test double for the VectorStore contract."""

    _rows: dict[tuple[str, str, str], VectorRecord] = field(default_factory=dict)

    def upsert(self, record: VectorRecord) -> None:
        key = (record.namespace, record.entity_ref, record.model_key)
        self._rows[key] = record

    def get(self, namespace: str, entity_ref: str, model_key: str) -> VectorRecord | None:
        return self._rows.get((namespace, entity_ref, model_key))

    def delete(self, namespace: str, entity_ref: str, model_key: str | None = None) -> int:
        if model_key is not None:
            key = (namespace, entity_ref, model_key)
            if key in self._rows:
                del self._rows[key]
                return 1
            return 0
        deleted = 0
        for key in list(self._rows):
            if key[0] == namespace and key[1] == entity_ref:
                del self._rows[key]
                deleted += 1
        return deleted

    def nearest(
        self,
        *,
        namespace: str,
        model_key: str,
        query: Sequence[float],
        limit: int = 10,
    ) -> list[NearestNeighbor]:
        scored: list[NearestNeighbor] = []
        for record in self._rows.values():
            if record.namespace != namespace or record.model_key != model_key:
                continue
            # Convert cosine similarity to a distance-like score for ordering.
            sim = cosine_similarity(query, record.embedding)
            scored.append(
                NearestNeighbor(
                    namespace=record.namespace,
                    entity_ref=record.entity_ref,
                    model_key=record.model_key,
                    distance=1.0 - sim,
                    metadata=dict(record.metadata),
                )
            )
        scored.sort(key=lambda row: row.distance)
        return scored[:limit]


def _record(ref: str, embedding: Sequence[float], metadata: Mapping[str, Any] | None = None) -> VectorRecord:
    return VectorRecord(
        namespace="unit",
        entity_ref=ref,
        content_hash=ref,
        model_key="fake",
        embedding=tuple(float(x) for x in embedding),
        metadata=dict(metadata or {}),
        dimensions=len(embedding),
    )


def test_upsert_get_delete() -> None:
    store = InMemoryVectorStore()
    store.upsert(_record("a", [1.0, 0.0], {"k": 1}))
    got = store.get("unit", "a", "fake")
    assert got is not None
    assert got.metadata["k"] == 1
    assert store.delete("unit", "a", "fake") == 1
    assert store.get("unit", "a", "fake") is None


def test_nearest_order() -> None:
    store = InMemoryVectorStore()
    store.upsert(_record("near", [1.0, 0.0]))
    store.upsert(_record("far", [0.0, 1.0]))
    hits = store.nearest(namespace="unit", model_key="fake", query=[1.0, 0.0], limit=2)
    assert [h.entity_ref for h in hits] == ["near", "far"]
