-- Per-node LLM call detail (one row per node per request)
CREATE TABLE IF NOT EXISTS node_inference_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id      TEXT        NOT NULL,
    thread_id       TEXT        NOT NULL,
    node_id         TEXT        NOT NULL,  -- extraction | evaluation | writing
    system_prompt   TEXT,
    response        TEXT,
    model_name      TEXT,
    input_tokens    INT,
    output_tokens   INT,
    total_tokens    INT,
    latency_ms      INT,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS node_inference_logs_request_idx ON node_inference_logs (request_id);
CREATE INDEX IF NOT EXISTS node_inference_logs_thread_idx  ON node_inference_logs (thread_id);
