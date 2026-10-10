CREATE TABLE IF NOT EXISTS agent_routing_feedback (
    id UUID PRIMARY KEY,
    client_id TEXT NOT NULL,
    agent_app TEXT NOT NULL,
    service_id TEXT NOT NULL,
    module_id TEXT NULL,
    operation_id TEXT NULL,
    routing_group_id TEXT NULL,
    routing_version TEXT NULL,
    routing_outcome TEXT NOT NULL,
    rating BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS agent_routing_feedback_client_created_idx
    ON agent_routing_feedback (client_id, created_at DESC);

CREATE INDEX IF NOT EXISTS agent_routing_feedback_version_created_idx
    ON agent_routing_feedback (routing_version, created_at DESC);
