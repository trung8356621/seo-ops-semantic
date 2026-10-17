-- Disposable keyword-group dataset snapshots. Reconstructible from Laravel.
-- Does not alter semantic_embeddings or topic_analysis_*.

CREATE TABLE IF NOT EXISTS keyword_group_dataset_revisions (
    id BIGSERIAL PRIMARY KEY,
    scope_ref TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    session_id TEXT NOT NULL UNIQUE,
    language TEXT NULL,
    dataset_hash TEXT NULL,
    keyword_count INTEGER NOT NULL DEFAULT 0 CHECK (keyword_count >= 0),
    expected_keyword_count INTEGER NOT NULL CHECK (expected_keyword_count > 0),
    expected_batch_count INTEGER NOT NULL CHECK (expected_batch_count > 0),
    status TEXT NOT NULL CHECK (status IN ('staging', 'committed', 'aborted', 'expired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    committed_at TIMESTAMPTZ NULL,
    CONSTRAINT keyword_group_dataset_revisions_scope_revision_uq UNIQUE (scope_ref, revision)
);

CREATE UNIQUE INDEX IF NOT EXISTS keyword_group_dataset_one_committed_idx
    ON keyword_group_dataset_revisions (scope_ref)
    WHERE status = 'committed';

CREATE INDEX IF NOT EXISTS keyword_group_dataset_revisions_scope_status_idx
    ON keyword_group_dataset_revisions (scope_ref, status);

CREATE TABLE IF NOT EXISTS keyword_group_dataset_items (
    revision_id BIGINT NOT NULL
        REFERENCES keyword_group_dataset_revisions (id) ON DELETE CASCADE,
    batch_index INTEGER NOT NULL CHECK (batch_index >= 1),
    keyword_ref TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    language TEXT NULL,
    industry_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_id, keyword_ref)
);

CREATE INDEX IF NOT EXISTS keyword_group_dataset_items_batch_idx
    ON keyword_group_dataset_items (revision_id, batch_index);
