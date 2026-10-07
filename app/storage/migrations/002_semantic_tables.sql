-- Generic embedding registry. Not a Topic/Keyword business schema.

CREATE TABLE IF NOT EXISTS semantic_models (
    id BIGSERIAL PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS semantic_embeddings (
    id BIGSERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    entity_ref TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    model_key TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    embedding vector NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT semantic_embeddings_unique UNIQUE (namespace, entity_ref, model_key)
);

CREATE INDEX IF NOT EXISTS semantic_embeddings_namespace_model_idx
    ON semantic_embeddings (namespace, model_key);

-- No HNSW/IVFFlat yet: those indexes need a fixed vector(N).
-- Sequential <=> is enough for infrastructure smoke tests and small namespaces.
