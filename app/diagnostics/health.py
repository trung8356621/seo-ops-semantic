from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.embedding.contracts import EmbeddingProvider
from app.storage.database import Database


def check_live(settings: Settings) -> dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.app_name,
    }


def check_postgres(database: Database) -> dict[str, Any]:
    try:
        with database.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok, current_setting('server_version') AS version")
                row = cur.fetchone()
        return {
            "status": "ok",
            "version": row["version"] if row else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
        }


def check_pgvector(database: Database) -> dict[str, Any]:
    try:
        with database.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                    """
                )
                row = cur.fetchone()
        if row is None:
            return {"status": "error", "error": "pgvector extension not installed"}
        return {"status": "ok", "version": row["extversion"]}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
        }


def check_embedding(provider: EmbeddingProvider, *, force_load: bool = False) -> dict[str, Any]:
    try:
        if not provider.is_loaded and not force_load:
            return {
                "status": "not_loaded",
                "provider": provider.provider_key,
                "model": provider.model_key,
                "model_version": provider.model_version,
                "dimensions": provider.dimensions,
            }
        if force_load or not provider.is_loaded:
            provider.load()
        return {
            "status": "ok",
            "provider": provider.provider_key,
            "model": provider.model_key,
            "model_version": provider.model_version,
            "dimensions": provider.dimensions,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "provider": provider.provider_key,
            "model": provider.model_key,
            "error": str(exc),
        }


def build_health(
    settings: Settings,
    database: Database,
    provider: EmbeddingProvider,
    *,
    force_embedding_load: bool = False,
) -> dict[str, Any]:
    postgres = check_postgres(database)
    pgvector = check_pgvector(database)
    embedding = check_embedding(provider, force_load=force_embedding_load)

    component_statuses = [postgres["status"], pgvector["status"], embedding["status"]]
    if any(status == "error" for status in component_statuses):
        overall = "error"
    elif embedding["status"] == "not_loaded":
        overall = "degraded"
    else:
        overall = "ok"

    return {
        "status": overall,
        "service": settings.app_name,
        "postgres": postgres,
        "pgvector": pgvector,
        "embedding": embedding,
    }


def build_ready(
    settings: Settings,
    database: Database,
    provider: EmbeddingProvider,
) -> tuple[dict[str, Any], bool]:
    payload = build_health(settings, database, provider, force_embedding_load=True)
    ready = (
        payload["postgres"]["status"] == "ok"
        and payload["pgvector"]["status"] == "ok"
        and payload["embedding"]["status"] == "ok"
    )
    payload["status"] = "ok" if ready else "error"
    payload["ready"] = ready
    return payload, ready
