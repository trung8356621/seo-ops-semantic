from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.config import Settings


class FakeCursor:
    def __init__(self) -> None:
        self.calls = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql, params) -> None:
        self.calls.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_value = FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cursor_value

    def commit(self) -> None:
        self.committed = True


def test_feedback_batch_requires_auth_and_upserts_without_content() -> None:
    app = create_app(Settings(INTERNAL_API_TOKEN="test-secret"))
    connection = FakeConnection()

    @contextmanager
    def fake_connection():
        yield connection

    app.state.database = SimpleNamespace(connection=fake_connection)
    item_id = uuid4()
    payload = {"items": [{
        "review_id": str(item_id), "client_id": "install-1",
        "agent_app": "seo-ops", "service_id": "seo-ops",
        "module_id": "keywords", "operation_id": "keywords.inventory",
        "routing_group_id": None, "routing_version": "7",
        "routing_outcome": "confident",
        "review_kind": "candidate_selection", "rating": None,
        "selected_candidate_id": "operation:a|group:selected",
        "preferred_candidate_id": "operation:b|group:alternative",
        "none_of_above": False,
    }]}

    client = TestClient(app)
    assert client.post("/v1/internal/agent-routing-feedback/batch", json=payload).status_code == 401
    response = client.post(
        "/v1/internal/agent-routing-feedback/batch",
        json=payload,
        headers={"X-Internal-Token": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"acknowledged": [str(item_id)]}
    sql, params = connection.cursor_value.calls[0]
    assert "ON CONFLICT (id) DO UPDATE" in sql
    assert "created_at =" not in sql
    assert "question" not in sql.lower()
    assert "answer" not in sql.lower()
    assert params[-3:] == (
        "operation:a|group:selected",
        "operation:b|group:alternative",
        False,
    )

    payload["items"][0]["preferred_candidate_id"] = "operation:a|group:selected"
    changed = client.post(
        "/v1/internal/agent-routing-feedback/batch",
        json=payload,
        headers={"X-Internal-Token": "test-secret"},
    )
    assert changed.status_code == 200
    assert changed.json() == {"acknowledged": [str(item_id)]}
    assert connection.cursor_value.calls[1][1][0] == item_id


def test_candidate_review_accepts_none_and_rejects_ambiguous_choice() -> None:
    item = {
        "review_id": str(uuid4()), "client_id": "install-1",
        "agent_app": "seo-ops", "service_id": "seo-ops",
        "routing_outcome": "ambiguous", "review_kind": "candidate_selection",
        "selected_candidate_id": None, "preferred_candidate_id": None,
        "none_of_above": True,
    }
    app = create_app(Settings(INTERNAL_API_TOKEN="test-secret"))
    connection = FakeConnection()

    @contextmanager
    def fake_connection():
        yield connection

    app.state.database = SimpleNamespace(connection=fake_connection)
    client = TestClient(app)
    headers = {"X-Internal-Token": "test-secret"}
    assert client.post("/v1/internal/agent-routing-feedback/batch", json={"items": [item]}, headers=headers).status_code == 200
    item["preferred_candidate_id"] = "fabricated"
    assert client.post("/v1/internal/agent-routing-feedback/batch", json={"items": [item]}, headers=headers).status_code == 422
