CREATE TABLE IF NOT EXISTS inference_logs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id   TEXT NOT NULL,
    thread_id    TEXT NOT NULL,
    model        TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    response     TEXT,
    tokens_used  INT,
    latency_ms   INT,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
