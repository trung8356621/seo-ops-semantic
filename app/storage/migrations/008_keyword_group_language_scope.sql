-- Keep one committed dataset and one active run per scope AND language.
-- Does not delete revisions, runs, groups, or embeddings.

ALTER TABLE keyword_group_runs
    ADD COLUMN IF NOT EXISTS language TEXT NULL;

UPDATE keyword_group_runs AS runs
SET language = revisions.language
FROM keyword_group_dataset_revisions AS revisions
WHERE runs.revision_id = revisions.id
  AND runs.language IS NULL
  AND revisions.language IS NOT NULL;

DROP INDEX IF EXISTS keyword_group_dataset_one_committed_idx;

CREATE UNIQUE INDEX IF NOT EXISTS keyword_group_dataset_one_committed_idx
    ON keyword_group_dataset_revisions (scope_ref, COALESCE(language, ''))
    WHERE status = 'committed';

DROP INDEX IF EXISTS keyword_group_runs_one_active_idx;

CREATE UNIQUE INDEX IF NOT EXISTS keyword_group_runs_one_active_idx
    ON keyword_group_runs (scope_ref, COALESCE(language, ''))
    WHERE status IN ('queued', 'running');
