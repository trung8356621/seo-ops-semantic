from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

from app.config import Settings, get_settings
from app.core.embedding.contracts import EmbeddingProvider
from app.core.embedding.factory import create_embedding_provider
from app.core.vector.postgres import PostgresVectorStore
from app.modules.keyword_grouping.analyzer import KeywordGroupAnalyzer, model_cache_key
from app.modules.keyword_grouping.contracts import (
    IndustryEvidenceIn,
    IndustryMembershipIn,
    KeywordGroupAnalysisRequest,
    KeywordIn,
)
from app.modules.keyword_grouping.fingerprint import settings_for_fingerprint
from app.modules.keyword_grouping.run_repository import KeywordGroupRunRepository
from app.storage.database import Database, get_database

_RUN_LOCK = threading.Lock()
_ADVISORY_LOCK_KEY = 748_201_617


def _release_embedding(embedding: EmbeddingProvider) -> None:
    release = getattr(embedding, "release", None)
    if release is not None:
        release()
    gc.collect()
    if os.name != "posix":
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        return


def execute_claimed_run(
    database: Database,
    settings: Settings,
    embedding: EmbeddingProvider,
    claimed: dict[str, Any],
) -> None:
    run_id = str(claimed["run_id"])
    stop = threading.Event()

    def beat(phase: str, completed: int, total: int) -> None:
        with database.connection() as conn:
            KeywordGroupRunRepository(conn).heartbeat(run_id, phase=phase, completed=completed, total=total)

    def loop() -> None:
        while not stop.wait(max(5, int(settings.keyword_group_run_heartbeat_seconds))):
            beat("running", 0, int(claimed["progress_total"]))

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        payload = claimed["fingerprint_payload"]
        if model_cache_key(embedding) != str(payload["model_key"]):
            raise RuntimeError("embedding model does not match the run fingerprint")
        frozen = settings_for_fingerprint(settings, payload)
        with database.connection() as conn:
            items = KeywordGroupRunRepository(conn).load_dataset_items(int(claimed["revision_id"]))
        if not items:
            raise RuntimeError("dataset revision is missing")
        keywords = [KeywordIn(ref=item["keyword_ref"], text=item["raw_text"]) for item in items]
        evidence = _evidence(items)
        embeddings: dict = {}
        cache_stats = {"hits": 0, "misses": 0}
        batch_size = max(1, min(1000, int(settings.keyword_group_dataset_batch_max)))
        started = time.perf_counter()
        with database.connection() as conn:
            analyzer = KeywordGroupAnalyzer(
                settings=frozen,
                embedding=embedding,
                vectors=PostgresVectorStore(conn),
                repository=None,
            )
            for offset in range(0, len(keywords), batch_size):
                chunk = keywords[offset:offset + batch_size]
                part, stats = analyzer._embed_keywords(
                    str(claimed["scope_ref"]),
                    chunk,
                    claimed.get("dataset_language"),
                )
                embeddings.update(part)
                cache_stats["hits"] += stats["hits"]
                cache_stats["misses"] += stats["misses"]
                conn.commit()
                beat("embed", offset + len(chunk), len(keywords))
        request = KeywordGroupAnalysisRequest(
            scope_ref=str(claimed["scope_ref"]),
            language=claimed.get("dataset_language"),
            keywords=keywords,
            industry_evidence=evidence or None,
        )
        beat("cluster", len(keywords), len(keywords))
        analyzer = KeywordGroupAnalyzer(settings=frozen, embedding=embedding, vectors=None, repository=None)
        response = analyzer._cluster(request, keywords, embeddings, cache_stats, 0, started)
        del embeddings
        del keywords
        del analyzer
        with database.connection() as conn:
            KeywordGroupRunRepository(conn).complete(run_id, response.groups, len(response.unassigned))
    except Exception as exc:
        with database.connection() as conn:
            KeywordGroupRunRepository(conn).fail(run_id, exc.__class__.__name__ + ": " + str(exc))
        raise
    finally:
        stop.set()
        thread.join(timeout=1)
        _release_embedding(embedding)


def process_available(database: Database, settings: Settings, embedding: EmbeddingProvider, worker_id: str) -> bool:
    if not _RUN_LOCK.acquire(blocking=False):
        return False
    lock_conn = None
    try:
        lock_conn = database.connect()
        with lock_conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (_ADVISORY_LOCK_KEY,))
            locked = cur.fetchone()
        if locked is None or not bool(locked["pg_try_advisory_lock"]):
            return False
        lock_conn.commit()
        with database.connection() as conn:
            KeywordGroupRunRepository(conn).recover_stale(
                int(settings.keyword_group_run_stale_seconds),
                int(settings.keyword_group_run_max_attempts),
            )
            claimed = KeywordGroupRunRepository(conn).claim(worker_id)
        if claimed is None:
            return False
        try:
            execute_claimed_run(database, settings, embedding, claimed)
        except Exception:
            return True
        return True
    finally:
        if lock_conn is not None:
            try:
                with lock_conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (_ADVISORY_LOCK_KEY,))
            finally:
                lock_conn.close()
        _RUN_LOCK.release()


def _queued(database: Database, settings: Settings) -> bool:
    with database.connection() as conn:
        KeywordGroupRunRepository(conn).recover_stale(
            int(settings.keyword_group_run_stale_seconds),
            int(settings.keyword_group_run_max_attempts),
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM keyword_group_runs WHERE status = 'queued' LIMIT 1")
            return cur.fetchone() is not None


def _run_once() -> None:
    settings = get_settings()
    database = get_database()
    embedding = create_embedding_provider(settings)
    process_available(database, settings, embedding, str(uuid.uuid4()))


def main() -> None:
    if "--once" in sys.argv:
        _run_once()
        return
    settings = get_settings()
    database = get_database()
    database.wait_until_ready()
    database.migrate()
    poll = max(1, int(settings.keyword_group_run_poll_seconds))
    while True:
        if not _queued(database, settings):
            time.sleep(poll)
            continue
        subprocess.run(
            [sys.executable, "-m", "app.modules.keyword_grouping.worker", "--once"],
            check=False,
        )
        time.sleep(1)


def _evidence(items: list[dict[str, Any]]) -> list[IndustryEvidenceIn]:
    evidence: list[IndustryEvidenceIn] = []
    for item in items:
        memberships = [
            IndustryMembershipIn(key=row["key"], group_type=row["group_type"])
            for row in item["industry_evidence"] or []
        ]
        if memberships:
            evidence.append(IndustryEvidenceIn(ref=item["keyword_ref"], memberships=memberships))
    return evidence


if __name__ == "__main__":
    main()
