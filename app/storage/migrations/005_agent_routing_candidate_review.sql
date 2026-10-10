ALTER TABLE agent_routing_feedback
    ALTER COLUMN rating DROP NOT NULL,
    ADD COLUMN IF NOT EXISTS review_kind TEXT NOT NULL DEFAULT 'answer_rating_legacy',
    ADD COLUMN IF NOT EXISTS selected_candidate_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS preferred_candidate_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS none_of_above BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE agent_routing_feedback
    ADD CONSTRAINT agent_routing_feedback_candidate_choice_check
    CHECK (
        review_kind = 'answer_rating_legacy'
        OR (
            review_kind = 'candidate_selection'
            AND (
                (none_of_above = TRUE AND preferred_candidate_id IS NULL)
                OR (none_of_above = FALSE AND preferred_candidate_id IS NOT NULL)
            )
        )
    );

CREATE INDEX IF NOT EXISTS agent_routing_feedback_candidate_created_idx
    ON agent_routing_feedback (
        service_id,
        selected_candidate_id,
        preferred_candidate_id,
        created_at DESC
    );
