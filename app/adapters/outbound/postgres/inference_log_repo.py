import json

import structlog

from app.infrastructure.database import PostgresConnection

logger = structlog.get_logger(__name__)


class InferenceLogRepository:
    def __init__(self, database: PostgresConnection) -> None:
        self._db = database

    async def record(
        self,
        *,
        request_id: str,
        thread_id: str,
        model: str,
        prompt: str,
        response: str | None = None,
        tokens_used: int | None = None,
        latency_ms: int | None = None,
        error: str | None = None,
        model_name: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        input_token_details: dict | None = None,
        output_token_details: dict | None = None,
    ) -> None:
        pool = await self._db.get_pool()
        await pool.execute(
            """
            INSERT INTO inference_logs
                (request_id, thread_id, model, prompt, response,
                 tokens_used, latency_ms, error,
                 model_name, input_tokens, output_tokens, total_tokens,
                 input_token_details, output_token_details)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8,
                    $9, $10, $11, $12, $13, $14)
            """,
            request_id,
            thread_id,
            model,
            prompt,
            response,
            tokens_used,
            latency_ms,
            error,
            model_name,
            input_tokens,
            output_tokens,
            total_tokens,
            json.dumps(input_token_details) if input_token_details else None,
            json.dumps(output_token_details) if output_token_details else None,
        )
