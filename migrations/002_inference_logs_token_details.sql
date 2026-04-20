ALTER TABLE inference_logs
    ADD COLUMN IF NOT EXISTS model_name      TEXT,
    ADD COLUMN IF NOT EXISTS input_tokens     INT,
    ADD COLUMN IF NOT EXISTS output_tokens    INT,
    ADD COLUMN IF NOT EXISTS total_tokens     INT,
    ADD COLUMN IF NOT EXISTS input_token_details  JSONB,
    ADD COLUMN IF NOT EXISTS output_token_details JSONB;
