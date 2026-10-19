from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from psycopg import connect
from psycopg.rows import dict_row

from app.api.app import create_app
from app.config import Settings
from app.core.embedding.contracts import EmbeddingResult
from app.modules.keyword_grouping.worker import process_available
from app.storage.database import Database

pytestmark = pytest.mark.integration


class HashEmbedding:
    provider_key = "test"
    model_key = "hash"
    model_version = "1"
    dimensions = 8
    is_loaded = True

    def load(self) -> None:
        return None

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        results = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            vector = tuple(seed[index] / 255 for index in range(8))
            if "may phat" in text:
                vector = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            results.append(EmbeddingResult(vector, self.model_key, self.model_version, 8, self.provider_key))
        return results


def _settings() -> Settings:
    return Settings(
        POSTGRES_HOST=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        POSTGRES_PORT=int(os.getenv("POSTGRES_PORT", "5433")),
        POSTGRES_DB=os.getenv("POSTGRES_DB", "semantic"),
        POSTGRES_USER=os.getenv("POSTGRES_USER", "semantic"),
        POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD", "semantic_local_change_me"),
        EMBEDDING_LAZY_LOAD=True,
        DB_CONNECT_RETRIES=5,
        DB_CONNECT_RETRY_SECONDS=1,
        KEYWORD_GROUP_RUN_STALE_SECONDS=1,
        KEYWORD_GROUP_RUN_MAX_ATTEMPTS=2,
    )  # type: ignore[arg-type]


@pytest.fixture()
def client() -> TestClient:
    if os.getenv("SEMANTIC_INTEGRATION") != "1" and not os.getenv("POSTGRES_HOST"):
        pytest.skip("integration DB not configured")
    app = create_app(_settings())
    with TestClient(app) as test_client:
        test_client.app.state.embedding_provider = HashEmbedding()
        yield test_client


def test_run_api_is_fast_and_worker_completes_without_blocking_health(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    started = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope})
    assert started.status_code == 202, started.text
    body = started.json()
    assert body["status"] == "queued"
    duplicate = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope})
    assert duplicate.status_code == 202, duplicate.text
    assert duplicate.json()["run_id"] == body["run_id"]

    database = Database(_settings())
    worker = threading.Thread(
        target=process_available,
        args=(database, _settings(), HashEmbedding(), "worker-a"),
    )
    worker.start()
    health = client.get("/health/live")
    worker.join(timeout=30)
    assert health.status_code == 200
    status = client.get(f"/v1/keyword-groups/runs/{body['run_id']}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["group_count"] >= 1
    page = client.get(f"/v1/keyword-groups/runs/{body['run_id']}/groups", params={"limit": 1})
    assert page.status_code == 200, page.text
    payload = page.json()
    assert payload["status"] == "completed"
    assert payload["input_hash"] == status.json()["dataset_hash"]
    assert payload["algorithm"] == "hybrid_semantic_lexical_v1"
    group = payload["groups"][0]
    assert group["group_ref"]
    assert group["representative_text"]
    assert group["representative_ref"]
    assert group["members"][0]["ref"]
    assert "similarity_score" in group["members"][0]
    _cleanup(scope)


def test_second_fingerprint_is_rejected_while_a_run_is_active(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    first = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope})
    assert first.status_code == 202
    client.app.state.settings = client.app.state.settings.model_copy(update={"keyword_group_semantic_floor": 0.61})
    second = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope})
    assert second.status_code == 409, second.text
    _cleanup(scope)


def test_stale_worker_is_requeued_and_missing_dataset_fails_cleanly(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope})
    run_id = created.json()["run_id"]
    _sql(
        """
        UPDATE keyword_group_runs
        SET status = 'running', attempt = 1, heartbeat_at = NOW() - INTERVAL '5 minutes'
        WHERE run_id = %s
        """,
        (run_id,),
    )
    database = Database(_settings())
    assert process_available(database, _settings(), HashEmbedding(), "worker-b") is True
    status = client.get(f"/v1/keyword-groups/runs/{run_id}")
    assert status.json()["status"] == "completed"

    scope_missing = _scope()
    _sync(client, scope_missing, ["may phat"])
    missing = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope_missing})
    missing_id = missing.json()["run_id"]
    _sql(
        """
        DELETE FROM keyword_group_dataset_items
        WHERE revision_id = (
            SELECT revision_id FROM keyword_group_runs WHERE run_id = %s
        )
        """,
        (missing_id,),
    )
    assert process_available(database, _settings(), HashEmbedding(), "worker-c") is True
    failed = client.get(f"/v1/keyword-groups/runs/{missing_id}")
    assert failed.json()["status"] == "failed"
    assert "missing" in failed.json()["error"]
    groups = client.get(f"/v1/keyword-groups/runs/{missing_id}/groups")
    assert groups.status_code == 409
    _cleanup(scope)
    _cleanup(scope_missing)


def test_active_run_keeps_its_dataset_revision_when_a_newer_snapshot_commits(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope})
    run_id = created.json()["run_id"]
    _sync(client, scope, ["on ap"])
    row = _sql(
        """
        SELECT revisions.status
        FROM keyword_group_runs AS runs
        JOIN keyword_group_dataset_revisions AS revisions ON revisions.id = runs.revision_id
        WHERE runs.run_id = %s
        """,
        (run_id,),
        fetch=True,
    )
    assert row["status"] == "retained"
    assert process_available(Database(_settings()), _settings(), HashEmbedding(), "worker-d") is True
    finished = client.get(f"/v1/keyword-groups/runs/{run_id}")
    assert finished.json()["status"] == "completed"
    assert finished.json()["dataset_revision"] == 1
    _cleanup(scope)


def test_completed_run_keeps_its_dataset_revision_when_a_newer_snapshot_commits(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope, "language": "vi"})
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]
    assert process_available(Database(_settings()), _settings(), HashEmbedding(), "worker-e") is True
    finished = client.get(f"/v1/keyword-groups/runs/{run_id}")
    assert finished.json()["status"] == "completed"
    _sync(client, scope, ["on ap"])
    row = _sql(
        """
        SELECT revisions.status
        FROM keyword_group_runs AS runs
        JOIN keyword_group_dataset_revisions AS revisions ON revisions.id = runs.revision_id
        WHERE runs.run_id = %s
        """,
        (run_id,),
        fetch=True,
    )
    assert row["status"] == "retained"
    current = client.get(f"/v1/keyword-groups/datasets/{scope}", params={"language": "vi"})
    assert current.status_code == 200, current.text
    assert current.json()["revision"] == 2
    still = client.get(f"/v1/keyword-groups/runs/{run_id}")
    assert still.json()["status"] == "completed"
    assert still.json()["dataset_revision"] == 1
    _cleanup(scope)


def test_create_run_language_matches_the_committed_dataset(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    matched = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope, "language": "vi"})
    assert matched.status_code == 202, matched.text
    assert matched.json()["status"] == "queued"
    other = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope, "language": "en"})
    assert other.status_code == 404, other.text
    _cleanup(scope)


def test_two_processes_cannot_claim_the_same_run(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope, "language": "vi"})
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]
    script = (
        "from app.modules.keyword_grouping.worker import process_available; "
        "from app.storage.database import Database; from tests.integration.test_keyword_group_runs import HashEmbedding, _settings; "
        "print(process_available(Database(_settings()), _settings(), HashEmbedding(), 'race'))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    first = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd="/app",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    second = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd="/app",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_a, err_a = first.communicate(timeout=60)
    out_b, err_b = second.communicate(timeout=60)
    assert first.returncode == 0, err_a.decode()
    assert second.returncode == 0, err_b.decode()
    results = {out_a.decode().strip(), out_b.decode().strip()}
    assert results == {"True", "False"}
    status = client.get(f"/v1/keyword-groups/runs/{run_id}")
    assert status.json()["status"] == "completed"
    members = _sql(
        "SELECT COUNT(*) AS n FROM keyword_group_run_members WHERE run_id = %s",
        (run_id,),
        fetch=True,
    )
    assert int(members["n"]) == int(status.json()["member_count"])
    _cleanup(scope)


def test_heartbeat_advances_while_embedding_is_still_running(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope, "language": "vi"})
    run_id = created.json()["run_id"]

    class SlowEmbedding(HashEmbedding):
        def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
            time.sleep(8)
            return super().embed_batch(texts)

    settings = _settings().model_copy(update={"keyword_group_run_heartbeat_seconds": 5})
    worker = threading.Thread(
        target=process_available,
        args=(Database(settings), settings, SlowEmbedding(), "heartbeat"),
    )
    worker.start()
    seen = False
    for _ in range(40):
        row = _sql(
            "SELECT status, progress_phase, heartbeat_at FROM keyword_group_runs WHERE run_id = %s",
            (run_id,),
            fetch=True,
        )
        if row["status"] == "running" and row["heartbeat_at"] is not None:
            seen = True
            break
        time.sleep(0.25)
    worker.join(timeout=30)
    assert seen is True
    finished = client.get(f"/v1/keyword-groups/runs/{run_id}")
    assert finished.json()["status"] == "completed"
    _cleanup(scope)


def test_failed_child_releases_the_advisory_lock_and_does_not_leave_the_run_running(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope, "language": "vi"})
    run_id = created.json()["run_id"]

    class BoomEmbedding(HashEmbedding):
        def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
            raise RuntimeError("child embed failed")

    assert process_available(Database(_settings()), _settings(), BoomEmbedding(), "boom") is True
    failed = client.get(f"/v1/keyword-groups/runs/{run_id}")
    assert failed.json()["status"] == "failed"
    assert "child embed failed" in failed.json()["error"]
    locked = _sql("SELECT pg_try_advisory_lock(748201617) AS locked", (), fetch=True)
    assert bool(locked["locked"]) is True
    _sql("SELECT pg_advisory_unlock(748201617)", ())
    assert process_available(Database(_settings()), _settings(), HashEmbedding(), "after-boom") is False
    _cleanup(scope)


def test_reprocessing_a_run_replaces_memberships(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope, "language": "vi"})
    run_id = created.json()["run_id"]
    assert process_available(Database(_settings()), _settings(), HashEmbedding(), "first") is True
    first = _sql(
        "SELECT COUNT(*) AS n FROM keyword_group_run_members WHERE run_id = %s",
        (run_id,),
        fetch=True,
    )
    _sql(
        """
        UPDATE keyword_group_runs
        SET status = 'queued', progress_phase = 'requeued', finished_at = NULL, error = NULL
        WHERE run_id = %s
        """,
        (run_id,),
    )
    assert process_available(Database(_settings()), _settings(), HashEmbedding(), "second") is True
    second = _sql(
        "SELECT COUNT(*) AS n FROM keyword_group_run_members WHERE run_id = %s",
        (run_id,),
        fetch=True,
    )
    assert int(second["n"]) == int(first["n"])
    assert int(second["n"]) > 0
    _cleanup(scope)


def test_completed_run_survives_a_new_connection(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, ["may phat dien", "may phat"])
    created = client.post("/v1/keyword-groups/runs", json={"scope_ref": scope})
    process_available(Database(_settings()), _settings(), HashEmbedding(), "worker-e")
    with connect(_settings().database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, group_count FROM keyword_group_runs WHERE run_id = %s",
                (created.json()["run_id"],),
            )
            row = cur.fetchone()
    assert row["status"] == "completed"
    assert row["group_count"] >= 1
    _cleanup(scope)


def _scope() -> str:
    return f"p3-{uuid.uuid4().hex}"


def _sync(client: TestClient, scope: str, texts: list[str]) -> None:
    from app.modules.keyword_grouping.dataset_hash import canonical_item, content_hash, dataset_hash

    items = [
        {
            "ref": str(index + 1),
            "text": text,
            "language": "vi",
            "industry_evidence": [],
            "metadata": {},
        }
        for index, text in enumerate(texts)
    ]
    digest = dataset_hash(
        scope,
        "vi",
        [
            canonical_item(
                ref=item["ref"],
                content_hash_value=content_hash(item["text"]),
                language="vi",
                industry_evidence=[],
                metadata={},
            )
            for item in items
        ],
    )
    session = client.post(
        "/v1/keyword-groups/datasets/sessions",
        json={
            "scope_ref": scope,
            "language": "vi",
            "expected_keyword_count": len(items),
            "expected_batch_count": 1,
        },
    )
    assert session.status_code == 201, session.text
    uploaded = client.put(
        f"/v1/keyword-groups/datasets/sessions/{session.json()['session_id']}/batches/1",
        json={"keywords": items},
    )
    assert uploaded.status_code == 200, uploaded.text
    committed = client.post(
        f"/v1/keyword-groups/datasets/sessions/{session.json()['session_id']}/commit",
        json={"dataset_hash": digest},
    )
    assert committed.status_code == 200, committed.text


def _sql(sql: str, params: tuple, fetch: bool = False):
    with connect(_settings().database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone() if fetch else None
        conn.commit()
    return row


def _cleanup(scope: str) -> None:
    with connect(_settings().database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM keyword_group_runs WHERE scope_ref = %s", (scope,))
            cur.execute("DELETE FROM keyword_group_dataset_revisions WHERE scope_ref = %s", (scope,))
        conn.commit()
