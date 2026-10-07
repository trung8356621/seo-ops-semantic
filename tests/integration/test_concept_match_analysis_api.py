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
        ready = test_client.get("/health/ready")
        assert ready.status_code == 200
        yield test_client


def test_concept_match_endpoint_shape(client: TestClient) -> None:
    payload = {
        "scope_ref": "site:4",
        "language": "vi",
        "entities": [{"ref": "kw_1", "text": "tuyển nhân viên may balo"}],
        "concepts": [
            {
                "key": "custom.recruitment",
                "positive_examples": ["tuyển thợ may", "việc làm xưởng may"],
                "negative_examples": ["xưởng may balo"],
            }
        ],
    }
    response = client.post("/v1/concept-matches/analyses", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["scope_ref"] == "site:4"
    score = body["entities"][0]["concepts"][0]
    assert "positive_max" in score
    assert "lexical" in score
    assert score["matching_strategy"] == "semantic"
    assert score["suggested_match"] is None
    assert body["diagnostics"]["cache"] == "direct_embed_batch"


def test_validation_422(client: TestClient) -> None:
    response = client.post(
        "/v1/concept-matches/analyses",
        json={
            "scope_ref": "site:4",
            "entities": [{"ref": "a", "text": "x"}, {"ref": "a", "text": "y"}],
            "concepts": [{"key": "c", "positive_examples": ["z"]}],
        },
    )
    assert response.status_code == 422
