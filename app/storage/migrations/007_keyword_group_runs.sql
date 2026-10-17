-- Keyword-group analysis runs. Additive. Does not alter embedding or topic_analysis data.
-- A revision referenced by a queued/running run is retained instead of deleted.

ALTER TABLE keyword_group_dataset_revisions
    DROP CONSTRAINT IF EXISTS keyword_group_dataset_revisions_status_check;

ALTER TABLE keyword_group_dataset_revisions
    ADD CONSTRAINT keyword_group_dataset_revisions_status_check
    CHECK (status IN ('staging', 'committed', 'aborted', 'expired', 'retained'));

CREATE TABLE IF NOT EXISTS keyword_group_runs (
    run_id TEXT PRIMARY KEY,
    scope_ref TEXT NOT NULL,
    revision_id BIGINT NOT NULL REFERENCES keyword_group_dataset_revisions (id),
    dataset_revision INTEGER NOT NULL,
    dataset_hash TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    fingerprint_payload JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    progress_phase TEXT NOT NULL DEFAULT 'queued',
    progress_completed INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT NULL,
    heartbeat_at TIMESTAMPTZ NULL,
    error TEXT NULL,
    group_count INTEGER NOT NULL DEFAULT 0,
    member_count INTEGER NOT NULL DEFAULT 0,
    unassigned_count INTEGER NOT NULL DEFAULT 0,
    algorithm TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    started_at TIMESTAMPTZ NULL,
    finished_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS keyword_group_runs_one_active_idx
    ON keyword_group_runs (scope_ref)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS keyword_group_runs_scope_created_idx
    ON keyword_group_runs (scope_ref, created_at DESC);

CREATE TABLE IF NOT EXISTS keyword_group_run_groups (
    run_id TEXT NOT NULL REFERENCES keyword_group_runs (run_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    group_ref TEXT NOT NULL,
    representative_ref TEXT NOT NULL,
    representative_text TEXT NOT NULL,
    member_count INTEGER NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, group_ref)
);

CREATE UNIQUE INDEX IF NOT EXISTS keyword_group_run_groups_ordinal_idx
    ON keyword_group_run_groups (run_id, ordinal);

CREATE TABLE IF NOT EXISTS keyword_group_run_members (
    run_id TEXT NOT NULL,
    group_ref TEXT NOT NULL,
    keyword_ref TEXT NOT NULL,
    text TEXT NOT NULL,
    similarity_score DOUBLE PRECISION NOT NULL,
    is_representative BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (run_id, group_ref, keyword_ref),
    FOREIGN KEY (run_id, group_ref)
        REFERENCES keyword_group_run_groups (run_id, group_ref)
        ON DELETE CASCADE
);
