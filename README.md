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
| `app/modules/topic` | Topic analysis V1 (proposal only — not Laravel Topic authority). |
| `app/modules/concept_matching` | Generic semantic concept matching (evidence/scores only). |
| `app/storage` | Internal semantic tables + SQL migrations. |
| `app/api` | Health + topic / keyword-groups / concept-matches analyses. No unrestricted `/embed`. |

## Topic analysis V1

```bash
# CLI
docker compose exec semantic-api \
  python -m app.modules.topic.cli analyze /app/tests/fixtures/topic_vi_keywords.json --persist

# HTTP
curl -s http://127.0.0.1:8088/v1/topic/analyses -H 'Content-Type: application/json' -d @request.json
```

Defaults (site_id=4 real-data calibration, TASK 3.1):

- `TOPIC_CLUSTER_ALGORITHM=average_linkage` → `cosine_average_linkage_v1`
- `TOPIC_CLUSTER_SIMILARITY_THRESHOLD=0.74` (average-linkage cut; distance = 1 − similarity)
- `TOPIC_ASSIGNMENT_MIN_SCORE=0.74` (post-cluster guard vs representative + confidence floor)
- `TOPIC_MIN_MEMBER_SIMILARITY=0.77` (used by `greedy_medoid_v2` only)
- `TOPIC_LOW_CONFIDENCE_SCORE=0.35` (heuristic confidence flag; not a probability)

Threshold semantics are non-redundant. Legacy `cosine_threshold_greedy_medoid_v1` remains available but is not the default (its member floor was dead when ≤ discovery threshold).

`input_hash`: server always computes the canonical hash from the normalized payload. A client-supplied hash must match or the request is rejected (422).

Disposable analysis tables: `topic_analysis_runs`, `topic_analysis_groups`, `topic_analysis_members`.

Real-data note (site 4, 883 keywords): average-linkage @ 0.74 cuts the prior 260-member star hub; analysis remains proposal/evidence only.

## Concept Matching

Generic “is this text semantically similar to these examples?” evidence API.

```bash
curl.exe -s http://127.0.0.1:8088/v1/concept-matches/analyses `
  -H "Content-Type: application/json" `
  -d @concept_request.json
```

### What it can do

- Embed entities + concept positive/negative examples with the shared multilingual ONNX model
- Return cosine similarity evidence (`positive_max`, `positive_top_k_mean`, `negative_max`, `margin`, best examples)
- Optionally evaluate a caller-supplied `decision_policy` → `suggested_match`

### What it cannot do

- Translation / LLM / research / sentence grammar classification
- Mutate Laravel data, create tags, or decide Topic exclusion policy
- Treat scores as probabilities or confidence percentages

**Scores are cosine similarities, not probabilities.** No LLM / translation / research is performed.

Positive examples are required (≥1). Negative examples are optional contrast signals. When `decision_policy` is omitted, `suggested_match` is `null`.

V1 is **stateless** (`direct_embed_batch`); it does not write a concept-matching cache namespace and does not change Topic/Keyword Group embedding caches.

## Non-goals

- Not Topic authority / Preview-Apply / Laravel mutation
- Not Laravel business DB or mirrored Topic tables
- Not Redis / Celery / workers
- Not authentication
- Not HNSW (not required at ~1k keywords)

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
| Image `seo-ops-semantic-semantic-api` | ~748 MB (post-scipy / average-linkage) |
| Image `pgvector/pgvector:0.8.0-pg16` | ~622 MB |
| Volume `seo-ops-semantic_model_cache` | ~252 MB |
| Volume `seo-ops-semantic_postgres_data` | ~49 MB |
| API container RAM idle (pre-model) | ~65 MiB |
| API container RAM after `/health/ready` | ~640 MiB |
| API peak during ~100–883 keyword analysis | ~888 MiB observed (below 1.2 GiB concern line) |
| Postgres container RAM | ~37 MiB |

Default API workers = **1** so the ONNX model is not duplicated across processes.

### FastEmbed note

FastEmbed 0.6.x emits a warning that `paraphrase-multilingual-MiniLM-L12-v2` now uses **mean pooling** instead of CLS. This scaffold accepts the current FastEmbed default. Pinning `fastembed==0.5.1` or registering a custom model would be needed only if CLS parity with an older run is required.
