from __future__ import annotations

from typing import Any

from app.config import Settings
from app.diagnostics.health import build_health, build_ready, check_live


class _FakeDb:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def connect(self):  # noqa: ANN201
        if self.fail:
            raise RuntimeError("connection refused")
        return _FakeConn()


class _FakeConn:
    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self):  # noqa: ANN201
        return _FakeCursor()


class _FakeCursor:
    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._sql = sql

    def fetchone(self) -> dict[str, str]:
        if "pg_extension" in self._sql:
            return {"extversion": "0.8.0"}
        return {"ok": 1, "version": "16.8"}


class _FakeProvider:
    def __init__(self, *, loaded: bool = False, fail_load: bool = False) -> None:
        self._loaded = loaded
        self._fail_load = fail_load

    @property
    def provider_key(self) -> str:
        return "fake"

    @property
    def model_key(self) -> str:
        return "fake-model"

    @property
    def model_version(self) -> str:
        return "v0"

    @property
    def dimensions(self) -> int | None:
        return 4 if self._loaded else None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._fail_load:
            raise RuntimeError("model missing")
        self._loaded = True


def test_live_always_ok() -> None:
    settings = Settings()
    assert check_live(settings)["status"] == "ok"


def test_health_degraded_when_model_not_loaded() -> None:
    payload = build_health(Settings(), _FakeDb(), _FakeProvider(loaded=False))  # type: ignore[arg-type]
    assert payload["status"] == "degraded"
    assert payload["embedding"]["status"] == "not_loaded"
    assert payload["postgres"]["status"] == "ok"
    assert payload["pgvector"]["status"] == "ok"


def test_health_reports_db_error() -> None:
    payload = build_health(Settings(), _FakeDb(fail=True), _FakeProvider(loaded=True))  # type: ignore[arg-type]
    assert payload["status"] == "error"
    assert payload["postgres"]["status"] == "error"


def test_ready_loads_model() -> None:
    provider = _FakeProvider(loaded=False)
    payload, ready = build_ready(Settings(), _FakeDb(), provider)  # type: ignore[arg-type]
    assert ready is True
    assert payload["embedding"]["status"] == "ok"
    assert provider.is_loaded is True
