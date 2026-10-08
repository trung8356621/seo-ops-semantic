"""Production install must cover third-party modules imported by the API."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

IMPORT_TO_DIST = {
    "fastapi": "fastapi",
    "fastembed": "fastembed",
    "httpx": "httpx",
    "numpy": "numpy",
    "psycopg": "psycopg",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "scipy": "scipy",
    "uvicorn": "uvicorn",
}


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _dist_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for raw in requirements:
        name = raw.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0]
        names.add(name.strip().lower())
    return names


def _imported_roots() -> set[str]:
    found: set[str] = set()
    for path in APP.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".", 1)[0])
    return found


def test_third_party_imports_are_production_dependencies() -> None:
    dists = _dist_names(_project()["dependencies"])
    missing: list[str] = []
    for module in sorted(_imported_roots()):
        if module == "app" or module in sys.stdlib_module_names:
            continue
        dist = IMPORT_TO_DIST.get(module)
        assert dist is not None, f"unmapped third-party import: {module}"
        if dist not in dists:
            missing.append(f"{module} ({dist})")
    assert missing == []


def test_dev_extra_keeps_pytest_and_httpx() -> None:
    dev = _dist_names(_project()["optional-dependencies"]["dev"])
    assert {"pytest", "httpx"} <= dev


def test_api_boot_import() -> None:
    from app.api.app import create_app

    assert callable(create_app)
