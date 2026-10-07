from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from psycopg import Connection, connect
from psycopg.rows import dict_row

from app.config import Settings, get_settings


class Database:
    """Lightweight psycopg access + SQL file migrations.

    No heavy ORM: only two infrastructure tables in this scaffold.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._migrations_dir = Path(__file__).resolve().parent / "migrations"

    def connect(self) -> Connection:
        return connect(self._settings.database_dsn, row_factory=dict_row)

    def wait_until_ready(self) -> None:
        retries = max(1, self._settings.db_connect_retries)
        delay = max(0.1, self._settings.db_connect_retry_seconds)
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                with self.connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
                return
            except Exception as exc:  # noqa: BLE001 — retry any connect failure
                last_error = exc
                time.sleep(delay)
        raise RuntimeError(f"postgres unavailable after {retries} retries: {last_error}")

    def migrate(self) -> list[str]:
        applied: list[str] = []
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute("SELECT version FROM schema_migrations")
                existing = {row["version"] for row in cur.fetchall()}

                files = sorted(self._migrations_dir.glob("*.sql"))
                for path in files:
                    version = path.name
                    if version in existing:
                        continue
                    sql = path.read_text(encoding="utf-8")
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)",
                        (version,),
                    )
                    applied.append(version)
            conn.commit()
        return applied

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()


@lru_cache
def get_database() -> Database:
    return Database(get_settings())
