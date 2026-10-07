from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.vector.contracts import NearestNeighbor, VectorRecord


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


class PostgresVectorStore:
    """pgvector-backed VectorStore. Feature modules must not open SQL elsewhere."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def upsert(self, record: VectorRecord) -> None:
        if record.dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        if len(record.embedding) != record.dimensions:
            raise ValueError("embedding length must equal dimensions")

        sql = """
            INSERT INTO semantic_embeddings (
                namespace, entity_ref, content_hash, model_key, dimensions, embedding, metadata
            ) VALUES (
                %(namespace)s, %(entity_ref)s, %(content_hash)s, %(model_key)s,
                %(dimensions)s, %(embedding)s::vector, %(metadata)s
            )
            ON CONFLICT (namespace, entity_ref, model_key)
            DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                dimensions = EXCLUDED.dimensions,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """
        with self._conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "namespace": record.namespace,
                    "entity_ref": record.entity_ref,
                    "content_hash": record.content_hash,
                    "model_key": record.model_key,
                    "dimensions": record.dimensions,
                    "embedding": _vector_literal(record.embedding),
                    "metadata": Jsonb(dict(record.metadata)),
                },
            )
        self._conn.commit()

    def get(self, namespace: str, entity_ref: str, model_key: str) -> VectorRecord | None:
        sql = """
            SELECT namespace, entity_ref, content_hash, model_key, dimensions,
                   embedding::text AS embedding_text, metadata
            FROM semantic_embeddings
            WHERE namespace = %s AND entity_ref = %s AND model_key = %s
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (namespace, entity_ref, model_key))
            row = cur.fetchone()
        if row is None:
            return None
        return self._to_record(row)

    def delete(self, namespace: str, entity_ref: str, model_key: str | None = None) -> int:
        if model_key is None:
            sql = "DELETE FROM semantic_embeddings WHERE namespace = %s AND entity_ref = %s"
            params: tuple[Any, ...] = (namespace, entity_ref)
        else:
            sql = (
                "DELETE FROM semantic_embeddings "
                "WHERE namespace = %s AND entity_ref = %s AND model_key = %s"
            )
            params = (namespace, entity_ref, model_key)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            deleted = cur.rowcount
        self._conn.commit()
        return int(deleted)

    def nearest(
        self,
        *,
        namespace: str,
        model_key: str,
        query: Sequence[float],
        limit: int = 10,
    ) -> list[NearestNeighbor]:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        sql = """
            SELECT namespace, entity_ref, model_key, metadata,
                   (embedding <=> %(query)s::vector) AS distance
            FROM semantic_embeddings
            WHERE namespace = %(namespace)s AND model_key = %(model_key)s
            ORDER BY embedding <=> %(query)s::vector
            LIMIT %(limit)s
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql,
                {
                    "namespace": namespace,
                    "model_key": model_key,
                    "query": _vector_literal(query),
                    "limit": limit,
                },
            )
            rows = cur.fetchall()
        return [
            NearestNeighbor(
                namespace=row["namespace"],
                entity_ref=row["entity_ref"],
                model_key=row["model_key"],
                distance=float(row["distance"]),
                metadata=self._metadata(row["metadata"]),
            )
            for row in rows
        ]

    def _to_record(self, row: Mapping[str, Any]) -> VectorRecord:
        text = str(row["embedding_text"]).strip("[]")
        embedding = tuple(float(part) for part in text.split(",") if part.strip() != "")
        return VectorRecord(
            namespace=row["namespace"],
            entity_ref=row["entity_ref"],
            content_hash=row["content_hash"],
            model_key=row["model_key"],
            embedding=embedding,
            metadata=self._metadata(row["metadata"]),
            dimensions=int(row["dimensions"]),
        )

    @staticmethod
    def _metadata(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, dict) else {}
        return dict(value)
