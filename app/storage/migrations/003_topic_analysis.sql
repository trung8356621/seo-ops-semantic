-- Disposable Topic *analysis* storage. NOT business Topics / memberships.
-- Destroyed by `docker compose down -v` with no Laravel impact.

CREATE TABLE IF NOT EXISTS topic_analysis_runs (
    analysis_id TEXT PRIMARY KEY,
    site_ref TEXT NOT NULL,
    language TEXT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    request_id TEXT NULL,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_dimensions INTEGER NOT NULL,
    algorithm TEXT NOT NULL,
    algorithm_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    keyword_count INTEGER NOT NULL,
    group_count INTEGER NOT NULL,
    unassigned_count INTEGER NOT NULL,
    singleton_count INTEGER NOT NULL,
    diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_json JSONB NOT NULL,
    error TEXT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    duration_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS topic_analysis_runs_site_created_idx
    ON topic_analysis_runs (site_ref, created_at DESC);

CREATE TABLE IF NOT EXISTS topic_analysis_groups (
    analysis_id TEXT NOT NULL REFERENCES topic_analysis_runs (analysis_id) ON DELETE CASCADE,
    group_ref TEXT NOT NULL,
    suggested_label TEXT NOT NULL,
    member_count INTEGER NOT NULL,
    mean_similarity DOUBLE PRECISION NOT NULL,
    min_similarity DOUBLE PRECISION NOT NULL,
    cohesion DOUBLE PRECISION NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (analysis_id, group_ref)
);

CREATE TABLE IF NOT EXISTS topic_analysis_members (
    analysis_id TEXT NOT NULL,
    group_ref TEXT NOT NULL,
    keyword_ref TEXT NOT NULL,
    text TEXT NOT NULL,
    similarity_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    is_representative BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (analysis_id, group_ref, keyword_ref),
    FOREIGN KEY (analysis_id, group_ref)
        REFERENCES topic_analysis_groups (analysis_id, group_ref)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS topic_analysis_members_keyword_idx
    ON topic_analysis_members (analysis_id, keyword_ref);
