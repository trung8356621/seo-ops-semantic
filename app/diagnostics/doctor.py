from __future__ import annotations

import math
import sys

from app.config import get_settings
from app.core.embedding.factory import create_embedding_provider
from app.core.similarity import cosine_similarity
from app.diagnostics.health import build_ready, check_pgvector, check_postgres
from app.storage.database import Database


def _ok(label: str, detail: str = "OK") -> None:
    print(f"{label:<20} {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"{label:<20} FAIL — {detail}")


def run() -> int:
    print("SEO-OPS Semantic Doctor")
    print("")

    settings = get_settings()
    database = Database(settings)
    provider = create_embedding_provider(settings)

    failures = 0

    try:
        _ok("API/runtime", f"OK  ({settings.app_name})")
    except Exception as exc:  # noqa: BLE001
        _fail("API/runtime", str(exc))
        failures += 1

    postgres = check_postgres(database)
    if postgres["status"] == "ok":
        _ok("PostgreSQL", f"OK  ({postgres.get('version')})")
    else:
        _fail("PostgreSQL", str(postgres.get("error")))
        failures += 1

    pgvector = check_pgvector(database)
    if pgvector["status"] == "ok":
        _ok("pgvector", f"OK  ({pgvector.get('version')})")
    else:
        _fail("pgvector", str(pgvector.get("error")))
        failures += 1

    try:
        provider.load()
        _ok("Embedding provider", f"OK  ({provider.provider_key})")
        _ok("Model", provider.model_key)
        _ok("Dimensions", str(provider.dimensions))
    except Exception as exc:  # noqa: BLE001
        _fail("Embedding provider", str(exc))
        failures += 1
        print("")
        return 1

    try:
        sample_vi = provider.embed("balo học sinh")
        if sample_vi.dimensions <= 0:
            raise RuntimeError("dimensions must be > 0")
        if any(not math.isfinite(v) for v in sample_vi.vector):
            raise RuntimeError("non-finite values in embedding")
        _ok(
            "Embed sample",
            f"OK  ({sample_vi.dimensions}d, model={sample_vi.model_key})",
        )

        sample_en = provider.embed("school backpack")
        sim = cosine_similarity(sample_vi.vector, sample_en.vector)
        _ok("Cosine (diagnostic)", f"{sim:.4f}  (not a quality gate)")
    except Exception as exc:  # noqa: BLE001
        _fail("Embed sample", str(exc))
        failures += 1

    ready_payload, ready = build_ready(settings, database, provider)
    if ready:
        _ok("Ready", "OK")
    else:
        _fail("Ready", str(ready_payload.get("status")))
        failures += 1

    print("")
    return 0 if failures == 0 else 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
