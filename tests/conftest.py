from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires PostgreSQL + pgvector")


def postgres_configured() -> bool:
    return bool(os.getenv("POSTGRES_HOST") or os.getenv("SEMANTIC_INTEGRATION") == "1")


@pytest.fixture(scope="session")
def integration_enabled() -> bool:
    return postgres_configured()
