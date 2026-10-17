from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection, errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.modules.keyword_grouping.contracts import GroupOut


class RunConflict(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class RunNotFound(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class RunRejected(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class KeywordGroupRunRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def create(
        self,
        *,
        scope_ref: str,
        revision_id: int,
        dataset_revision: int,
        dataset_hash: str,
        fingerprint: str,
        fingerprint_payload: dict[str, Any],
        algorithm: str,
        keyword_count: int,
        language: str | None = None,
    ) -> dict[str, Any]:
        existing = self._same_fingerprint(scope_ref, fingerprint, language)
        if existing is not None:
            return existing
        active = self._active(scope_ref, language)
        if active is not None:
            raise RunConflict("scope already has a queued or running analysis")
        run_id = str(uuid.uuid4())
        try:
            with self._conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO keyword_group_runs (
                        run_id, scope_ref, language, revision_id, dataset_revision, dataset_hash,
                        fingerprint, fingerprint_payload, status, progress_total,
                        algorithm, input_hash
                    ) VALUES (
                        %(run_id)s, %(scope_ref)s, %(language)s, %(revision_id)s, %(dataset_revision)s,
                        %(dataset_hash)s, %(fingerprint)s, %(fingerprint_payload)s, 'queued',
                        %(progress_total)s, %(algorithm)s, %(input_hash)s
                    )
                    RETURNING run_id, scope_ref, dataset_revision, dataset_hash, fingerprint,
                              status, progress_phase, progress_completed, progress_total,
                              group_count, member_count, unassigned_count, algorithm,
                              input_hash, error, started_at, finished_at, heartbeat_at, created_at
                    """,
                    {
                        "run_id": run_id,
                        "scope_ref": scope_ref,
                        "language": language,
                        "revision_id": revision_id,
                        "dataset_revision": dataset_revision,
                        "dataset_hash": dataset_hash,
                        "fingerprint": fingerprint,
                        "fingerprint_payload": Jsonb(fingerprint_payload),
                        "progress_total": keyword_count,
                        "algorithm": algorithm,
                        "input_hash": dataset_hash,
                    },
                )
                row = cur.fetchone()
            self._conn.commit()
        except errors.UniqueViolation:
            self._conn.rollback()
            existing = self._same_fingerprint(scope_ref, fingerprint, language) or self._active(scope_ref, language)
            if existing is not None and existing["fingerprint"] == fingerprint:
                return existing
            raise RunConflict("scope already has a queued or running analysis") from None
        return _public_run(row)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT run_id, scope_ref, dataset_revision, dataset_hash, fingerprint, status,
                       progress_phase, progress_completed, progress_total, group_count,
                       member_count, unassigned_count, algorithm, input_hash, error,
                       started_at, finished_at, heartbeat_at, created_at
                FROM keyword_group_runs
                WHERE run_id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
        return None if row is None else _public_run(row)

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH next_run AS (
                    SELECT run_id
                    FROM keyword_group_runs
                    WHERE status = 'queued'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE keyword_group_runs AS runs
                SET status = 'running',
                    attempt = attempt + 1,
                    worker_id = %(worker_id)s,
                    heartbeat_at = NOW(),
                    started_at = COALESCE(started_at, NOW()),
                    progress_phase = 'claimed'
                FROM next_run, keyword_group_dataset_revisions AS revisions
                WHERE runs.run_id = next_run.run_id
                  AND revisions.id = runs.revision_id
                RETURNING runs.run_id, runs.scope_ref, runs.revision_id, runs.dataset_revision,
                          runs.dataset_hash, runs.fingerprint, runs.fingerprint_payload,
                          runs.algorithm, runs.input_hash, runs.progress_total,
                          revisions.language AS dataset_language
                """,
                {"worker_id": worker_id},
            )
            row = cur.fetchone()
        self._conn.commit()
        return row

    def heartbeat(self, run_id: str, *, phase: str, completed: int, total: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE keyword_group_runs
                SET heartbeat_at = NOW(),
                    progress_phase = %s,
                    progress_completed = %s,
                    progress_total = %s
                WHERE run_id = %s AND status = 'running'
                """,
                (phase, completed, total, run_id),
            )
        self._conn.commit()

    def recover_stale(self, stale_seconds: int, max_attempts: int) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE keyword_group_runs
                SET status = 'queued',
                    worker_id = NULL,
                    progress_phase = 'requeued',
                    heartbeat_at = NULL
                WHERE status = 'running'
                  AND heartbeat_at < NOW() - make_interval(secs => %s)
                  AND attempt < %s
                """,
                (stale_seconds, max_attempts),
            )
            requeued = cur.rowcount
            cur.execute(
                """
                DELETE FROM keyword_group_run_groups
                WHERE run_id IN (
                    SELECT run_id FROM keyword_group_runs WHERE status = 'queued' AND progress_phase = 'requeued'
                )
                """
            )
            cur.execute(
                """
                UPDATE keyword_group_runs
                SET status = 'failed',
                    error = 'worker heartbeat expired',
                    finished_at = NOW(),
                    progress_phase = 'failed'
                WHERE status = 'running'
                  AND heartbeat_at < NOW() - make_interval(secs => %s)
                  AND attempt >= %s
                """,
                (stale_seconds, max_attempts),
            )
            failed = cur.rowcount
            cur.execute(
                """
                DELETE FROM keyword_group_run_groups
                WHERE run_id IN (
                    SELECT run_id FROM keyword_group_runs
                    WHERE status = 'failed' AND error = 'worker heartbeat expired'
                )
                """
            )
        self._conn.commit()
        return requeued + failed

    def complete(self, run_id: str, groups: list[GroupOut], unassigned_count: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM keyword_group_run_groups WHERE run_id = %s", (run_id,))
            member_count = 0
            for ordinal, group in enumerate(groups):
                cur.execute(
                    """
                    INSERT INTO keyword_group_run_groups (
                        run_id, ordinal, group_ref, representative_ref, representative_text,
                        member_count, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        ordinal,
                        group.group_ref,
                        group.representative_ref,
                        group.representative_text,
                        group.member_count,
                        Jsonb(group.model_dump()),
                    ),
                )
                for member in group.members:
                    cur.execute(
                        """
                        INSERT INTO keyword_group_run_members (
                            run_id, group_ref, keyword_ref, text, similarity_score, is_representative
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            group.group_ref,
                            member.ref,
                            member.text,
                            member.similarity_score,
                            member.is_representative,
                        ),
                    )
                    member_count += 1
            updated = cur.execute(
                """
                UPDATE keyword_group_runs
                SET status = 'completed',
                    progress_phase = 'completed',
                    progress_completed = progress_total,
                    group_count = %s,
                    member_count = %s,
                    unassigned_count = %s,
                    error = NULL,
                    finished_at = NOW(),
                    heartbeat_at = NOW()
                WHERE run_id = %s AND status = 'running'
                """,
                (len(groups), member_count, unassigned_count, run_id),
            )
            if cur.rowcount != 1:
                raise RunRejected("run is no longer running")
        self._conn.commit()

    def fail(self, run_id: str, error: str) -> None:
        summary = " ".join(error.split())[:500]
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM keyword_group_run_groups WHERE run_id = %s", (run_id,))
            cur.execute(
                """
                UPDATE keyword_group_runs
                SET status = 'failed',
                    progress_phase = 'failed',
                    error = %s,
                    group_count = 0,
                    member_count = 0,
                    finished_at = NOW(),
                    heartbeat_at = NOW()
                WHERE run_id = %s AND status = 'running'
                """,
                (summary, run_id),
            )
        self._conn.commit()

    def groups_page(self, run_id: str, *, cursor: int, limit: int) -> dict[str, Any]:
        run = self._raw(run_id)
        if run is None:
            raise RunNotFound("run not found")
        if run["status"] != "completed":
            raise RunRejected(f"run is {run['status']}")
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT payload
                FROM keyword_group_run_groups
                WHERE run_id = %s AND ordinal >= %s
                ORDER BY ordinal
                LIMIT %s
                """,
                (run_id, cursor, limit + 1),
            )
            rows = list(cur.fetchall())
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit:
            next_cursor = cursor + limit
        return {
            "run_id": run_id,
            "status": "completed",
            "scope_ref": run["scope_ref"],
            "input_hash": run["input_hash"],
            "algorithm": run["algorithm"],
            "groups": [row["payload"] for row in page],
            "next_cursor": next_cursor,
        }

    def load_dataset_items(self, revision_id: int) -> list[dict[str, Any]]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT keyword_ref, raw_text, language, industry_evidence
                FROM keyword_group_dataset_items
                WHERE revision_id = %s
                ORDER BY keyword_ref
                """,
                (revision_id,),
            )
            return list(cur.fetchall())

    def _same_fingerprint(self, scope_ref: str, fingerprint: str, language: str | None) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT run_id, scope_ref, dataset_revision, dataset_hash, fingerprint, status,
                       progress_phase, progress_completed, progress_total, group_count,
                       member_count, unassigned_count, algorithm, input_hash, error,
                       started_at, finished_at, heartbeat_at, created_at
                FROM keyword_group_runs
                WHERE scope_ref = %s AND fingerprint = %s
                  AND language IS NOT DISTINCT FROM %s
                  AND status IN ('queued', 'running', 'completed')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (scope_ref, fingerprint, language),
            )
            row = cur.fetchone()
        return None if row is None else _public_run(row)

    def _active(self, scope_ref: str, language: str | None) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT run_id, scope_ref, dataset_revision, dataset_hash, fingerprint, status,
                       progress_phase, progress_completed, progress_total, group_count,
                       member_count, unassigned_count, algorithm, input_hash, error,
                       started_at, finished_at, heartbeat_at, created_at
                FROM keyword_group_runs
                WHERE scope_ref = %s AND language IS NOT DISTINCT FROM %s
                  AND status IN ('queued', 'running')
                """,
                (scope_ref, language),
            )
            row = cur.fetchone()
        return None if row is None else _public_run(row)

    def _raw(self, run_id: str) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM keyword_group_runs WHERE run_id = %s", (run_id,))
            return cur.fetchone()


def _public_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "scope_ref": row["scope_ref"],
        "dataset_revision": int(row["dataset_revision"]),
        "dataset_hash": row["dataset_hash"],
        "fingerprint": row["fingerprint"],
        "status": row["status"],
        "progress_phase": row["progress_phase"],
        "progress_completed": int(row["progress_completed"]),
        "progress_total": int(row["progress_total"]),
        "group_count": int(row["group_count"]),
        "member_count": int(row["member_count"]),
        "unassigned_count": int(row["unassigned_count"]),
        "algorithm": row["algorithm"],
        "input_hash": row["input_hash"],
        "error": row["error"],
        "started_at": _iso(row["started_at"]),
        "finished_at": _iso(row["finished_at"]),
        "heartbeat_at": _iso(row["heartbeat_at"]),
        "created_at": _iso(row["created_at"]),
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat()
