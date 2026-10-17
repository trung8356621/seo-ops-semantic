from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import Settings
from app.core.text.normalization import normalize_text
from app.modules.keyword_grouping.dataset_hash import (
    batch_hash,
    canonical_item,
    content_hash,
    dataset_hash,
)


class DatasetConflict(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class DatasetRejected(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class DatasetNotFound(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class KeywordGroupDatasetRepository:
    """Staging sessions that become one committed snapshot per scope and language."""

    def __init__(self, conn: Connection, settings: Settings) -> None:
        self._conn = conn
        self._settings = settings

    def begin_session(
        self,
        *,
        scope_ref: str,
        language: str | None,
        expected_keyword_count: int,
        expected_batch_count: int,
    ) -> dict[str, Any]:
        self._reject_limits(expected_keyword_count, expected_batch_count)
        self._lock(scope_ref)
        self._expire_stale(scope_ref)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                FROM keyword_group_dataset_revisions
                WHERE scope_ref = %s
                """,
                (scope_ref,),
            )
            revision = int(cur.fetchone()["revision"])
            session_id = str(uuid.uuid4())
            expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=self._settings.keyword_group_dataset_staging_ttl_seconds
            )
            cur.execute(
                """
                INSERT INTO keyword_group_dataset_revisions (
                    scope_ref, revision, session_id, language,
                    expected_keyword_count, expected_batch_count, status, expires_at
                ) VALUES (
                    %(scope_ref)s, %(revision)s, %(session_id)s, %(language)s,
                    %(expected_keyword_count)s, %(expected_batch_count)s, 'staging', %(expires_at)s
                )
                RETURNING session_id, scope_ref, revision, expected_keyword_count,
                          expected_batch_count, expires_at
                """,
                {
                    "scope_ref": scope_ref,
                    "revision": revision,
                    "session_id": session_id,
                    "language": language,
                    "expected_keyword_count": expected_keyword_count,
                    "expected_batch_count": expected_batch_count,
                    "expires_at": expires_at,
                },
            )
            row = cur.fetchone()
        self._conn.commit()
        return _session_payload(row)

    def put_batch(
        self,
        session_id: str,
        batch_index: int,
        keywords: list[dict[str, Any]],
    ) -> dict[str, Any]:
        session = self._session_for_update(session_id)
        self._lock(session["scope_ref"])
        session = self._session_for_update(session_id)
        self._ensure_staging(session)
        expected_batches = int(session["expected_batch_count"])
        if batch_index < 1 or batch_index > expected_batches:
            raise DatasetRejected(
                f"batch_index {batch_index} is outside 1..{expected_batches}"
            )
        batch_max = self._settings.keyword_group_dataset_batch_max
        if len(keywords) > batch_max:
            raise DatasetRejected(
                f"batch size {len(keywords)} exceeds KEYWORD_GROUP_DATASET_BATCH_MAX={batch_max}"
            )
        prepared = [self._prepare_keyword(item) for item in keywords]
        refs = [item["ref"] for item in prepared]
        if len(refs) != len(set(refs)):
            self._conn.rollback()
            raise DatasetRejected("duplicate keyword refs in batch")
        digest = batch_hash(
            [{key: value for key, value in item.items() if key != "raw_text"} for item in prepared]
        )
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT keyword_ref, content_hash, language, industry_evidence, metadata
                FROM keyword_group_dataset_items
                WHERE revision_id = %s AND batch_index = %s
                ORDER BY keyword_ref
                """,
                (session["id"], batch_index),
            )
            existing = list(cur.fetchall())
            if existing:
                stored = [
                    canonical_item(
                        ref=row["keyword_ref"],
                        content_hash_value=row["content_hash"],
                        language=row["language"],
                        industry_evidence=list(row["industry_evidence"]),
                        metadata=dict(row["metadata"]),
                    )
                    for row in existing
                ]
                if batch_hash(stored) == digest and len(stored) == len(prepared):
                    self._conn.rollback()
                    return {
                        "session_id": session_id,
                        "batch_index": batch_index,
                        "keyword_count": len(prepared),
                        "batch_hash": digest,
                        "idempotent": True,
                    }
                self._conn.rollback()
                raise DatasetConflict(
                    f"batch {batch_index} already stored with a different payload"
                )
            cur.execute(
                """
                SELECT COUNT(*) AS stored
                FROM keyword_group_dataset_items
                WHERE revision_id = %s
                """,
                (session["id"],),
            )
            stored_count = int(cur.fetchone()["stored"])
            if stored_count + len(prepared) > int(session["expected_keyword_count"]):
                raise DatasetRejected("uploaded keywords exceed expected_keyword_count")
            cur.execute(
                """
                SELECT keyword_ref FROM keyword_group_dataset_items
                WHERE revision_id = %s AND keyword_ref = ANY(%s)
                """,
                (session["id"], refs),
            )
            clash = cur.fetchone()
            if clash is not None:
                raise DatasetRejected(
                    f"duplicate keyword ref {clash['keyword_ref']} across batches"
                )
            for item in prepared:
                cur.execute(
                    """
                    INSERT INTO keyword_group_dataset_items (
                        revision_id, batch_index, keyword_ref, raw_text, content_hash,
                        language, industry_evidence, metadata
                    ) VALUES (
                        %(revision_id)s, %(batch_index)s, %(keyword_ref)s, %(raw_text)s,
                        %(content_hash)s, %(language)s, %(industry_evidence)s, %(metadata)s
                    )
                    """,
                    {
                        "revision_id": session["id"],
                        "batch_index": batch_index,
                        "keyword_ref": item["ref"],
                        "raw_text": item["raw_text"],
                        "content_hash": item["content_hash"],
                        "language": item["language"],
                        "industry_evidence": Jsonb(item["industry_evidence"]),
                        "metadata": Jsonb(item["metadata"]),
                    },
                )
        self._conn.commit()
        return {
            "session_id": session_id,
            "batch_index": batch_index,
            "keyword_count": len(prepared),
            "batch_hash": digest,
            "idempotent": False,
        }

    def commit_session(self, session_id: str, claimed_hash: str) -> dict[str, Any]:
        session = self._session_for_update(session_id)
        self._lock(session["scope_ref"])
        session = self._session_for_update(session_id)
        self._ensure_staging(session)
        items = self._items(int(session["id"]))
        expected_batches = int(session["expected_batch_count"])
        present = {int(item["batch_index"]) for item in items}
        missing = [index for index in range(1, expected_batches + 1) if index not in present]
        if missing:
            self._conn.rollback()
            raise DatasetRejected(f"missing batches: {missing}")
        expected_count = int(session["expected_keyword_count"])
        if len(items) != expected_count:
            self._conn.rollback()
            raise DatasetRejected(
                f"keyword count {len(items)} does not match expected {expected_count}"
            )
        canonical = [
            canonical_item(
                ref=item["keyword_ref"],
                content_hash_value=item["content_hash"],
                language=item["language"],
                industry_evidence=list(item["industry_evidence"]),
                metadata=dict(item["metadata"]),
            )
            for item in items
        ]
        computed = dataset_hash(session["scope_ref"], session["language"], canonical)
        if computed != claimed_hash:
            self._conn.rollback()
            raise DatasetRejected("dataset_hash mismatch")
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, revision, dataset_hash, keyword_count, session_id, committed_at
                FROM keyword_group_dataset_revisions
                WHERE scope_ref = %s AND status = 'committed'
                  AND language IS NOT DISTINCT FROM %s
                FOR UPDATE
                """,
                (session["scope_ref"], session["language"]),
            )
            current = cur.fetchone()
            if current is not None and int(current["revision"]) > int(session["revision"]):
                self._conn.rollback()
                raise DatasetConflict(
                    "stale session cannot overwrite a newer committed revision"
                )
            if (
                current is not None
                and current["dataset_hash"] == computed
                and int(current["keyword_count"]) == len(items)
            ):
                cur.execute(
                    "DELETE FROM keyword_group_dataset_revisions WHERE id = %s",
                    (session["id"],),
                )
                self._conn.commit()
                return {
                    "scope_ref": session["scope_ref"],
                    "session_id": current["session_id"],
                    "revision": int(current["revision"]),
                    "dataset_hash": computed,
                    "keyword_count": len(items),
                    "status": "unchanged",
                    "unchanged": True,
                }
            if current is not None:
                if self._revision_is_active(int(current["id"])):
                    cur.execute(
                        """
                        UPDATE keyword_group_dataset_revisions
                        SET status = 'retained'
                        WHERE id = %s AND status = 'committed'
                        """,
                        (current["id"],),
                    )
                else:
                    cur.execute(
                        "DELETE FROM keyword_group_dataset_revisions WHERE id = %s",
                        (current["id"],),
                    )
            cur.execute(
                """
                UPDATE keyword_group_dataset_revisions
                SET status = 'committed',
                    dataset_hash = %s,
                    keyword_count = %s,
                    committed_at = NOW()
                WHERE id = %s AND status = 'staging'
                """,
                (computed, len(items), session["id"]),
            )
        self._conn.commit()
        return {
            "scope_ref": session["scope_ref"],
            "session_id": session_id,
            "revision": int(session["revision"]),
            "dataset_hash": computed,
            "keyword_count": len(items),
            "status": "committed",
            "unchanged": False,
        }

    def abort_session(self, session_id: str) -> None:
        session = self._session_for_update(session_id)
        if session["status"] != "staging":
            self._conn.rollback()
            raise DatasetRejected("only a staging session can be aborted")
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE keyword_group_dataset_revisions
                SET status = 'aborted'
                WHERE id = %s AND status = 'staging'
                """,
                (session["id"],),
            )
            cur.execute(
                "DELETE FROM keyword_group_dataset_items WHERE revision_id = %s",
                (session["id"],),
            )
        self._conn.commit()

    def current(self, scope_ref: str, *, include_refs: bool, language: str | None = None) -> dict[str, Any] | None:
        language_clause = ""
        params: tuple[object, ...] = (scope_ref,)
        if language is not None:
            language_clause = " AND language IS NOT DISTINCT FROM %s"
            params = (scope_ref, language)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT id, scope_ref, revision, dataset_hash, language, keyword_count, committed_at
                FROM keyword_group_dataset_revisions
                WHERE scope_ref = %s AND status = 'committed'{language_clause}
                """,
                params,
            )
            row = cur.fetchone()
            if row is None:
                return None
            refs = None
            if include_refs:
                cur.execute(
                    """
                    SELECT keyword_ref
                    FROM keyword_group_dataset_items
                    WHERE revision_id = %s
                    ORDER BY keyword_ref
                    """,
                    (row["id"],),
                )
                refs = [item["keyword_ref"] for item in cur.fetchall()]
        return {
            "scope_ref": row["scope_ref"],
            "revision": int(row["revision"]),
            "dataset_hash": row["dataset_hash"],
            "language": row["language"],
            "keyword_count": int(row["keyword_count"]),
            "committed_at": row["committed_at"].isoformat(),
            "refs": refs,
        }

    def _prepare_keyword(self, keyword: dict[str, Any]) -> dict[str, Any]:
        text = str(keyword["text"])
        normalized = normalize_text(text)
        if normalized == "":
            raise DatasetRejected("keyword text is blank")
        if len(normalized) > self._settings.topic_max_text_length:
            raise DatasetRejected(
                f"keyword text length exceeds TOPIC_MAX_TEXT_LENGTH="
                f"{self._settings.topic_max_text_length}"
            )
        evidence = [
            {"key": item["key"], "group_type": item["group_type"]}
            for item in keyword.get("industry_evidence") or []
        ]
        if len(evidence) > self._settings.keyword_group_dataset_max_evidence:
            raise DatasetRejected("industry evidence exceeds the per-keyword limit")
        metadata = dict(keyword.get("metadata") or {})
        try:
            encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        except TypeError as exc:
            raise DatasetRejected("metadata is not JSON") from exc
        if len(encoded.encode("utf-8")) > self._settings.keyword_group_dataset_max_metadata_bytes:
            raise DatasetRejected("metadata exceeds the size limit")
        language = keyword.get("language")
        return canonical_item(
            ref=str(keyword["ref"]),
            content_hash_value=content_hash(text),
            language=language if language else None,
            industry_evidence=evidence,
            metadata=metadata,
        ) | {"raw_text": text}

    def _revision_is_active(self, revision_id: int) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM keyword_group_runs
                WHERE revision_id = %s AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (revision_id,),
            )
            return cur.fetchone() is not None

    def _reject_limits(self, expected_keyword_count: int, expected_batch_count: int) -> None:
        max_keywords = self._settings.keyword_group_dataset_max_keywords
        batch_max = self._settings.keyword_group_dataset_batch_max
        if expected_keyword_count > max_keywords:
            raise DatasetRejected(
                f"expected_keyword_count {expected_keyword_count} exceeds "
                f"KEYWORD_GROUP_DATASET_MAX_KEYWORDS={max_keywords}"
            )
        if expected_batch_count > expected_keyword_count:
            raise DatasetRejected("expected_batch_count cannot exceed keyword count")
        if expected_keyword_count > expected_batch_count * batch_max:
            raise DatasetRejected("expected_keyword_count does not fit in the declared batches")

    def _ensure_staging(self, session: dict[str, Any]) -> None:
        if session["status"] != "staging":
            self._conn.rollback()
            raise DatasetRejected(f"session is {session['status']}")
        expires_at = session["expires_at"]
        if expires_at <= datetime.now(timezone.utc):
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE keyword_group_dataset_revisions
                    SET status = 'expired'
                    WHERE id = %s AND status = 'staging'
                    """,
                    (session["id"],),
                )
                cur.execute(
                    "DELETE FROM keyword_group_dataset_items WHERE revision_id = %s",
                    (session["id"],),
                )
            self._conn.commit()
            raise DatasetRejected("staging session expired")

    def _expire_stale(self, scope_ref: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE keyword_group_dataset_revisions
                SET status = 'expired'
                WHERE scope_ref = %s AND status = 'staging' AND expires_at <= NOW()
                """,
                (scope_ref,),
            )
            cur.execute(
                """
                DELETE FROM keyword_group_dataset_items
                WHERE revision_id IN (
                    SELECT id FROM keyword_group_dataset_revisions
                    WHERE scope_ref = %s AND status = 'expired'
                )
                """,
                (scope_ref,),
            )

    def _items(self, revision_id: int) -> list[dict[str, Any]]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT batch_index, keyword_ref, content_hash, language,
                       industry_evidence, metadata
                FROM keyword_group_dataset_items
                WHERE revision_id = %s
                ORDER BY keyword_ref
                """,
                (revision_id,),
            )
            return list(cur.fetchall())

    def _session_for_update(self, session_id: str) -> dict[str, Any]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, session_id, scope_ref, revision, language, status, expires_at,
                       expected_keyword_count, expected_batch_count
                FROM keyword_group_dataset_revisions
                WHERE session_id = %s
                FOR UPDATE
                """,
                (session_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise DatasetNotFound("session not found")
        return row

    def _lock(self, scope_ref: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"kg-dataset:{scope_ref}",))


def _session_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "scope_ref": row["scope_ref"],
        "revision": int(row["revision"]),
        "expected_keyword_count": int(row["expected_keyword_count"]),
        "expected_batch_count": int(row["expected_batch_count"]),
        "expires_at": row["expires_at"].isoformat(),
    }
