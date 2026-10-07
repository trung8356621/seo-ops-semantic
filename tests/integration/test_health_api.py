from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.config import Settings

pytestmark = pytest.mark.integration


def _settings(**overrides: object) -> Settings:
    base = {
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "127.0.0.1"),
        "POSTGRES_PORT": int(os.getenv("POSTGRES_PORT", "5433")),
        "POSTGRES_DB": os.getenv("POSTGRES_DB", "semantic"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER", "semantic"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "semantic_local_change_me"),
        "EMBEDDING_LAZY_LOAD": True,
        "DB_CONNECT_RETRIES": 5,
        "DB_CONNECT_RETRY_SECONDS": 1,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def client() -> TestClient:
    if os.getenv("SEMANTIC_INTEGRATION") != "1" and not os.getenv("POSTGRES_HOST"):
        pytest.skip("integration DB not configured; set SEMANTIC_INTEGRATION=1")
    app = create_app(_settings())
    with TestClient(app) as test_client:
        yield test_client


def test_live_and_health(client: TestClient) -> None:
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["postgres"]["status"] == "ok"
    assert body["pgvector"]["status"] == "ok"
    assert body["embedding"]["status"] in {"ok", "not_loaded"}


def test_ready_loads_embedding(client: TestClient) -> None:
    ready = client.get("/health/ready")
    # First ready may download the model; allow long timeout via TestClient default.
    assert ready.status_code == 200
    body = ready.json()
    assert body["ready"] is True
    assert body["embedding"]["status"] == "ok"
    assert body["embedding"]["dimensions"] > 0


def test_health_reports_bad_db() -> None:
    app = create_app(
        _settings(
            POSTGRES_HOST="127.0.0.1",
            POSTGRES_PORT=1,
            DB_CONNECT_RETRIES=1,
            DB_CONNECT_RETRY_SECONDS=0.1,
        )
    )
    # Bypass lifespan wait by calling health helpers through a short client that
    # will fail startup — instead exercise diagnostics directly.
    from app.diagnostics.health import build_health
    from app.core.embedding.factory import create_embedding_provider
    from app.storage.database import Database

    settings = _settings(POSTGRES_PORT=1, DB_CONNECT_RETRIES=1, DB_CONNECT_RETRY_SECONDS=0.1)
    payload = build_health(settings, Database(settings), create_embedding_provider(settings))
    assert payload["status"] == "error"
    assert payload["postgres"]["status"] == "error"
