# seo-ops-semantic

Standalone **semantic analytics infrastructure**: text prep, ONNX embeddings, and PostgreSQL/pgvector storage.

This is **not** a Topic service, Keyword service, or Laravel business database. Keyword/Topic will be the first consumer later. Other modules (Internal Link, Article, GSC, …) may compose the same core primitives.

## Architecture

```text
Feature module (later)
    ↓
Reusable core primitives (text / embedding / vector / similarity / clustering / ranking)
    ↓
Embedding runtime + PostgreSQL/pgvector
```

| Layer | Role |
| --- | --- |
| `app/core/*` | Small reusable tools. No Topic/Keyword/Laravel knowledge. |
| `app/modules/topic` | Placeholder only in this scaffold. |
| `app/storage` | Internal semantic tables + SQL migrations. |
| `app/api` | Health / diagnostics only. No production `/embed` API yet. |

## Non-goals (this scaffold)

- Not Topic authority / clustering / proposal apply
- Not Laravel business DB or mirrored Topic tables
- Not Redis / Celery / workers
- Not authentication
- Not quality benchmarks or Topic thresholds

## Stack

| Component | Pin |
| --- | --- |
| Python | 3.12 (image `python:3.12.10-slim-bookworm`) |
| FastAPI | 0.115.12 |
| Uvicorn | 0.34.2 (workers = **1**) |
| PostgreSQL | 16 (`pgvector/pgvector:0.8.0-pg16`) |
| Embedding | FastEmbed + ONNX Runtime |
| Default model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, multilingual, ~0.22 GB download) |

Workers stay at **1** so the ONNX model is not loaded once per process.

## Local startup

```bash
cp .env.example .env
docker compose up -d --build
```

Wait until both services are healthy:

```bash
docker compose ps
curl http://127.0.0.1:8088/health
curl http://127.0.0.1:8088/health/ready
```

Logs:

```bash
docker compose logs -f semantic-api
```

Stop (keep volumes):

```bash
docker compose down
```

Start again (DB + model cache persist):

```bash
docker compose up -d
```

### Destroy disposable semantic data

```bash
docker compose down -v
```

`-v` deletes Compose **named volumes** (`postgres_data`, `model_cache`). That wipes the semantic database **and** downloaded model weights for this project. It does not touch Laravel DBs.

## Doctor

Inside the API container:

```bash
docker compose exec semantic-api python -m app.diagnostics.doctor
```

Embeds `balo học sinh` (and prints a diagnostic cosine vs `school backpack`). Similarity here is **not** a quality gate.

## Health

| Endpoint | Meaning |
| --- | --- |
| `GET /health/live` | Process alive |
| `GET /health` | Component status (embedding may be `not_loaded` when lazy) |
| `GET /health/ready` | DB + pgvector + model loadable |

## Model cache

- Container path: `/models` (`MODEL_CACHE_DIR`)
- Compose volume: named `model_cache`, or host override via `SEMANTIC_MODEL_CACHE`
- Weights are downloaded on first embed / `/health/ready` / doctor — **not** committed to Git

Optional Windows override (example only — not baked into repo defaults):

```bash
# .env
SEMANTIC_MODEL_CACHE=E:/docker-data/seo-ops-semantic/models
POSTGRES_DATA=E:/docker-data/seo-ops-semantic/postgres
```

## Configuration

See `.env.example`. Important keys:

- `EMBEDDING_PROVIDER=onnx_fastembed`
- `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `MODEL_CACHE_DIR=/models`
- `UVICORN_WORKERS=1`

## Database

SQL migrations under `app/storage/migrations/`:

1. `CREATE EXTENSION vector`
2. `semantic_models`, `semantic_embeddings` (generic, namespaced)

No Topic / article / GSC / Laravel tables.

Mechanism: plain SQL files applied by `Database.migrate()` (tracked in `schema_migrations`). No heavy ORM — only two infrastructure tables.

## Tests

Unit (no Docker):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/unit -q
```

Integration (Compose stack up):

```bash
set SEMANTIC_INTEGRATION=1
set POSTGRES_HOST=127.0.0.1
set POSTGRES_PORT=5433
pytest tests/integration -q
```

## Resource footprint (measured on this workstation)

Values below were observed after a successful local `docker compose up -d --build` and model load. They are machine-specific.

| Item | Observed |
| --- | --- |
| Image `seo-ops-semantic-semantic-api` | 551 MB |
| Image `pgvector/pgvector:0.8.0-pg16` | 622 MB |
| Volume `seo-ops-semantic_model_cache` | ~252 MB (`/models` ≈ 241 MB after first download) |
| Volume `seo-ops-semantic_postgres_data` | ~48 MB |
| API container RAM after model load | ~889 MiB |
| Postgres container RAM | ~37 MiB |

Default API workers = **1** so the ONNX model is not duplicated across processes.

### FastEmbed note

FastEmbed 0.6.x emits a warning that `paraphrase-multilingual-MiniLM-L12-v2` now uses **mean pooling** instead of CLS. This scaffold accepts the current FastEmbed default. Pinning `fastembed==0.5.1` or registering a custom model would be needed only if CLS parity with an older run is required.
