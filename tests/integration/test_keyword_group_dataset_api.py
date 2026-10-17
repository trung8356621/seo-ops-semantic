from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from psycopg import connect
from psycopg.rows import dict_row

from app.api.app import create_app
from app.config import Settings
from app.modules.keyword_grouping.dataset_hash import canonical_item, content_hash, dataset_hash

pytestmark = pytest.mark.integration


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
        MODEL_CACHE_DIR=os.getenv("MODEL_CACHE_DIR", ".models"),
    )  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def client() -> TestClient:
    if os.getenv("SEMANTIC_INTEGRATION") != "1" and not os.getenv("POSTGRES_HOST"):
        pytest.skip("integration DB not configured")
    app = create_app(_settings())
    with TestClient(app) as test_client:
        yield test_client


def test_full_sync_commit_and_idempotent_batch(client: TestClient) -> None:
    scope = _scope()
    session = _begin(client, scope, [_kw("1", "máy phát"), _kw("2", "ổn áp")], batches=1)
    first = _upload(client, session, 1, [_kw("1", "máy phát"), _kw("2", "ổn áp")])
    second = _upload(client, session, 1, [_kw("1", "máy phát"), _kw("2", "ổn áp")])
    assert second.status_code == 200, second.text
    assert second.json()["idempotent"] is True
    assert second.json()["batch_hash"] == first.json()["batch_hash"]
    committed = _commit(client, scope, session, [_kw("1", "máy phát"), _kw("2", "ổn áp")])
    assert committed["unchanged"] is False
    assert committed["keyword_count"] == 2
    current = client.get(f"/v1/keyword-groups/datasets/{scope}", params={"include_refs": True})
    assert current.status_code == 200
    assert current.json()["refs"] == ["1", "2"]
    _cleanup(scope)


def test_conflicting_batch_retry_is_rejected(client: TestClient) -> None:
    scope = _scope()
    session = _begin(client, scope, [_kw("1", "máy phát")], batches=1)
    assert _upload(client, session, 1, [_kw("1", "máy phát")]).status_code == 200
    conflict = _upload(client, session, 1, [_kw("1", "máy phát điện")])
    assert conflict.status_code == 409, conflict.text
    committed = _commit(client, scope, session, [_kw("1", "máy phát")])
    assert committed["keyword_count"] == 1
    _cleanup(scope)


def test_missing_batch_and_hash_mismatch_do_not_commit(client: TestClient) -> None:
    scope = _scope()
    session = _begin(
        client,
        scope,
        [_kw("1", "máy phát"), _kw("2", "ổn áp")],
        batches=2,
    )
    assert _upload(client, session, 1, [_kw("1", "máy phát")]).status_code == 200
    missing = client.post(
        f"/v1/keyword-groups/datasets/sessions/{session['session_id']}/commit",
        json={"dataset_hash": "0" * 64},
    )
    assert missing.status_code == 422, missing.text
    assert "missing batches" in missing.text
    assert _upload(client, session, 2, [_kw("2", "ổn áp")]).status_code == 200
    mismatch = client.post(
        f"/v1/keyword-groups/datasets/sessions/{session['session_id']}/commit",
        json={"dataset_hash": "1" * 64},
    )
    assert mismatch.status_code == 422, mismatch.text
    assert client.get(f"/v1/keyword-groups/datasets/{scope}").status_code == 404
    committed = _commit(client, scope, session, [_kw("1", "máy phát"), _kw("2", "ổn áp")])
    assert committed["status"] == "committed"
    _cleanup(scope)


def test_failed_later_session_keeps_previous_revision(client: TestClient) -> None:
    scope = _scope()
    first = _sync(client, scope, [[_kw("1", "máy phát"), _kw("2", "ổn áp")]])
    session = _begin(
        client,
        scope,
        [_kw("1", "máy phát"), _kw("9", "tủ điện"), _kw("8", "biến áp")],
        batches=3,
    )
    assert _upload(client, session, 1, [_kw("1", "máy phát")]).status_code == 200
    assert _upload(client, session, 2, [_kw("9", "tủ điện")]).status_code == 200
    rejected = client.post(
        f"/v1/keyword-groups/datasets/sessions/{session['session_id']}/commit",
        json={"dataset_hash": _hash(scope, [_kw("1", "máy phát"), _kw("9", "tủ điện"), _kw("8", "biến áp")])},
    )
    assert rejected.status_code == 422, rejected.text
    current = client.get(f"/v1/keyword-groups/datasets/{scope}", params={"include_refs": True})
    assert current.json()["revision"] == first["revision"]
    assert current.json()["refs"] == ["1", "2"]
    _cleanup(scope)


def test_deleted_keyword_is_absent_from_next_revision(client: TestClient) -> None:
    scope = _scope()
    _sync(client, scope, [[_kw("1", "máy phát"), _kw("2", "ổn áp"), _kw("3", "tủ điện")]])
    second = _sync(client, scope, [[_kw("1", "máy phát"), _kw("3", "tủ điện")]])
    current = client.get(f"/v1/keyword-groups/datasets/{scope}", params={"include_refs": True})
    assert current.json()["revision"] == second["revision"]
    assert current.json()["refs"] == ["1", "3"]
    _cleanup(scope)


def test_older_session_cannot_publish_over_newer_commit(client: TestClient) -> None:
    scope = _scope()
    older = _begin(client, scope, [_kw("1", "máy phát")], batches=1)
    newer = _begin(client, scope, [_kw("2", "ổn áp")], batches=1)
    assert newer["revision"] > older["revision"]
    assert _upload(client, older, 1, [_kw("1", "máy phát")]).status_code == 200
    assert _upload(client, newer, 1, [_kw("2", "ổn áp")]).status_code == 200
    published = _commit(client, scope, newer, [_kw("2", "ổn áp")])
    stale = client.post(
        f"/v1/keyword-groups/datasets/sessions/{older['session_id']}/commit",
        json={"dataset_hash": _hash(scope, [_kw("1", "máy phát")])},
    )
    assert stale.status_code == 409, stale.text
    current = client.get(f"/v1/keyword-groups/datasets/{scope}", params={"include_refs": True})
    assert current.json()["revision"] == published["revision"]
    assert current.json()["refs"] == ["2"]
    _cleanup(scope)


def test_identical_dataset_is_unchanged(client: TestClient) -> None:
    scope = _scope()
    first = _sync(client, scope, [[_kw("1", "máy phát")]])
    second = _sync(client, scope, [[_kw("1", "máy phát")]])
    assert second["unchanged"] is True
    assert second["revision"] == first["revision"]
    _cleanup(scope)


def test_abort_and_expiry_drop_staging_only(client: TestClient) -> None:
    scope = _scope()
    kept = _sync(client, scope, [[_kw("1", "máy phát")]])
    session = _begin(client, scope, [_kw("2", "ổn áp")], batches=1)
    assert _upload(client, session, 1, [_kw("2", "ổn áp")]).status_code == 200
    aborted = client.post(f"/v1/keyword-groups/datasets/sessions/{session['session_id']}/abort")
    assert aborted.status_code == 204, aborted.text
    current = client.get(f"/v1/keyword-groups/datasets/{scope}")
    assert current.json()["revision"] == kept["revision"]

    expiring = _begin(client, scope, [_kw("3", "tủ điện")], batches=1)
    _expire(expiring["session_id"])
    expired = _upload(client, expiring, 1, [_kw("3", "tủ điện")])
    assert expired.status_code == 422, expired.text
    assert "expired" in expired.text
    assert client.get(f"/v1/keyword-groups/datasets/{scope}").json()["revision"] == kept["revision"]
    _cleanup(scope)


def test_committed_dataset_survives_new_connection_and_can_be_rebuilt(client: TestClient) -> None:
    scope = _scope()
    committed = _sync(client, scope, [[_kw("1", "máy phát"), _kw("2", "ổn áp")]])
    with connect(_settings().database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT revision, dataset_hash, keyword_count
                FROM keyword_group_dataset_revisions
                WHERE scope_ref = %s AND status = 'committed'
                """,
                (scope,),
            )
            row = cur.fetchone()
    assert row["revision"] == committed["revision"]
    assert row["keyword_count"] == 2

    _cleanup(scope)
    assert client.get(f"/v1/keyword-groups/datasets/{scope}").status_code == 404
    rebuilt = _sync(client, scope, [[_kw("1", "máy phát"), _kw("2", "ổn áp")]])
    assert rebuilt["unchanged"] is False
    assert rebuilt["keyword_count"] == 2
    _cleanup(scope)


def _scope() -> str:
    return f"p1-{uuid.uuid4().hex}"


def _kw(ref: str, text: str) -> dict:
    return {
        "ref": ref,
        "text": text,
        "language": "vi",
        "industry_evidence": [],
        "metadata": {},
    }


def _hash(scope: str, keywords: list[dict]) -> str:
    items = [
        canonical_item(
            ref=keyword["ref"],
            content_hash_value=content_hash(keyword["text"]),
            language=keyword["language"],
            industry_evidence=keyword["industry_evidence"],
            metadata=keyword["metadata"],
        )
        for keyword in keywords
    ]
    return dataset_hash(scope, "vi", items)


def _begin(client: TestClient, scope: str, keywords: list[dict], batches: int) -> dict:
    response = client.post(
        "/v1/keyword-groups/datasets/sessions",
        json={
            "scope_ref": scope,
            "language": "vi",
            "expected_keyword_count": len(keywords),
            "expected_batch_count": batches,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _upload(client: TestClient, session: dict, batch_index: int, keywords: list[dict]):
    return client.put(
        f"/v1/keyword-groups/datasets/sessions/{session['session_id']}/batches/{batch_index}",
        json={"keywords": keywords},
    )


def _commit(client: TestClient, scope: str, session: dict, keywords: list[dict]) -> dict:
    response = client.post(
        f"/v1/keyword-groups/datasets/sessions/{session['session_id']}/commit",
        json={"dataset_hash": _hash(scope, keywords)},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _sync(client: TestClient, scope: str, batches: list[list[dict]]) -> dict:
    keywords = [keyword for batch in batches for keyword in batch]
    session = _begin(client, scope, keywords, batches=len(batches))
    for index, batch in enumerate(batches, start=1):
        uploaded = _upload(client, session, index, batch)
        assert uploaded.status_code == 200, uploaded.text
    return _commit(client, scope, session, keywords)


def _expire(session_id: str) -> None:
    with connect(_settings().database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE keyword_group_dataset_revisions
                SET expires_at = NOW() - INTERVAL '1 minute'
                WHERE session_id = %s
                """,
                (session_id,),
            )
        conn.commit()


def _cleanup(scope: str) -> None:
    with connect(_settings().database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM keyword_group_dataset_revisions WHERE scope_ref = %s",
                (scope,),
            )
        conn.commit()
