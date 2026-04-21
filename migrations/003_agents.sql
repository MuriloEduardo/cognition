-- Agents: one per tenant, configures the LLM provider and defaults
CREATE TABLE IF NOT EXISTS agents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    model       TEXT NOT NULL DEFAULT 'gpt-4o-mini',
    temperature FLOAT NOT NULL DEFAULT 0.7,
    max_tokens  INT NOT NULL DEFAULT 1024,
    api_key     TEXT,
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agents_tenant_idx ON agents (tenant_id);

-- Agent nodes: LangGraph nodes (not workflow nodes)
-- Each node defines a step in the LangGraph pipeline for that agent
CREATE TABLE IF NOT EXISTS agent_nodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        UUID NOT NULL REFERENCES agents (id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    system_prompt   TEXT,
    response_format JSONB,
    node_order      INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (agent_id, name)
);

CREATE INDEX IF NOT EXISTS agent_nodes_agent_idx ON agent_nodes (agent_id);

-- Agent edges: LangGraph edges (not workflow edges)
-- from_node_id NULL = START, to_node_id NULL = END
CREATE TABLE IF NOT EXISTS agent_edges (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id         UUID NOT NULL REFERENCES agents (id) ON DELETE CASCADE,
    from_node_id     UUID REFERENCES agent_nodes (id) ON DELETE CASCADE,
    to_node_id       UUID REFERENCES agent_nodes (id) ON DELETE CASCADE,
    condition_prompt TEXT,
    priority         INT NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_edges_agent_idx ON agent_edges (agent_id);
