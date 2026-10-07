from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.core.vector.contracts import VectorRecord
from app.core.vector.postgres import PostgresVectorStore
from app.storage.database import Database

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    return Settings(
        POSTGRES_HOST=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        POSTGRES_PORT=int(os.getenv("POSTGRES_PORT", "5433")),
        POSTGRES_DB=os.getenv("POSTGRES_DB", "semantic"),
        POSTGRES_USER=os.getenv("POSTGRES_USER", "semantic"),
        POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD", "semantic_local_change_me"),
    )


@pytest.fixture(scope="module")
def database() -> Database:
    if os.getenv("SEMANTIC_INTEGRATION") != "1" and not os.getenv("POSTGRES_HOST"):
        pytest.skip("integration DB not configured; set SEMANTIC_INTEGRATION=1")
    db = Database(_settings())
    try:
        db.wait_until_ready()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"postgres unavailable: {exc}")
    db.migrate()
    return db


def test_pgvector_extension(database: Database) -> None:
    with database.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
    assert row is not None
    assert row["extversion"]


def test_upsert_get_nearest(database: Database) -> None:
    with database.connection() as conn:
        store = PostgresVectorStore(conn)
        namespace = "integration-smoke"
        model_key = "test-model"
        store.delete(namespace, "alpha", model_key)
        store.delete(namespace, "beta", model_key)

        store.upsert(
            VectorRecord(
                namespace=namespace,
                entity_ref="alpha",
                content_hash="alpha",
                model_key=model_key,
                embedding=(1.0, 0.0, 0.0),
                metadata={"label": "alpha"},
                dimensions=3,
            )
        )
        store.upsert(
            VectorRecord(
                namespace=namespace,
                entity_ref="beta",
                content_hash="beta",
                model_key=model_key,
                embedding=(0.0, 1.0, 0.0),
                metadata={"label": "beta"},
                dimensions=3,
            )
        )

        got = store.get(namespace, "alpha", model_key)
        assert got is not None
        assert got.entity_ref == "alpha"
        assert len(got.embedding) == 3

        hits = store.nearest(
            namespace=namespace,
            model_key=model_key,
            query=(1.0, 0.0, 0.0),
            limit=2,
        )
        assert hits[0].entity_ref == "alpha"
        assert hits[0].distance <= hits[1].distance
