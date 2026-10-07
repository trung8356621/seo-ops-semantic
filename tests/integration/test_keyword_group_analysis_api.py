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
        "TOPIC_CLUSTER_SIMILARITY_THRESHOLD": 0.70,
        "TOPIC_MIN_MEMBER_SIMILARITY": 0.60,
        "TOPIC_ASSIGNMENT_MIN_SCORE": 0.60,
        "TOPIC_MIN_GROUP_SIZE": 2,
        "MODEL_CACHE_DIR": os.getenv("MODEL_CACHE_DIR", ".models"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def client() -> TestClient:
    if os.getenv("SEMANTIC_INTEGRATION") != "1" and not os.getenv("POSTGRES_HOST"):
        pytest.skip("integration DB not configured")
    app = create_app(_settings())
    with TestClient(app) as test_client:
        yield test_client


def test_keyword_groups_post_get_delete(client: TestClient) -> None:
    payload = {
        "scope_ref": "site-integration-kg",
        "language": "vi",
        "keywords": [
            {"ref": "1", "text": "balo học sinh"},
            {"ref": "2", "text": "balo sinh viên"},
            {"ref": "3", "text": "balo laptop"},
            {"ref": "99", "text": "vali kéo du lịch"},
        ],
    }
    created = client.post("/v1/keyword-groups/analyses", json=payload)
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["status"] == "completed"
    assert body["scope_ref"] == "site-integration-kg"
    assert "suggested_label" not in body
    assert "site_ref" not in body
    for group in body["groups"]:
        assert "representative_ref" in group
        assert "representative_text" in group
        assert "suggested_label" not in group
        for member in group["members"]:
            assert "ref" in member
            assert "keyword_ref" not in member
    analysis_id = body["analysis_id"]

    fetched = client.get(f"/v1/keyword-groups/analyses/{analysis_id}")
    assert fetched.status_code == 200
    assert fetched.json()["analysis_id"] == analysis_id

    # Topic GET must not surface keyword-group rows.
    topic_get = client.get(f"/v1/topic/analyses/{analysis_id}")
    assert topic_get.status_code == 404

    deleted = client.delete(f"/v1/keyword-groups/analyses/{analysis_id}")
    assert deleted.status_code == 204
    missing = client.get(f"/v1/keyword-groups/analyses/{analysis_id}")
    assert missing.status_code == 404


def test_keyword_groups_duplicate_refs_422(client: TestClient) -> None:
    payload = {
        "scope_ref": "site-integration-kg",
        "keywords": [
            {"ref": "kw-1", "text": "balo"},
            {"ref": "kw-1", "text": "túi"},
        ],
    }
    response = client.post("/v1/keyword-groups/analyses", json=payload)
    assert response.status_code == 422


def test_old_topic_analyses_still_works(client: TestClient) -> None:
    payload = {
        "site_ref": "integration-topic-compat",
        "language": "vi",
        "keywords": [
            {"ref": "kw-1", "text": "balo học sinh"},
            {"ref": "kw-2", "text": "balo sinh viên"},
            {"ref": "kw-3", "text": "cách giặt áo thun"},
        ],
    }
    created = client.post("/v1/topic/analyses", json=payload)
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["site_ref"] == "integration-topic-compat"
    assert "suggested_label" in body["groups"][0] if body["groups"] else True
    analysis_id = body["analysis_id"]
    assert client.get(f"/v1/topic/analyses/{analysis_id}").status_code == 200
    assert client.delete(f"/v1/topic/analyses/{analysis_id}").status_code == 204
