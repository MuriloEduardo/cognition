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
    ) -> None:
        pool = await self._db.get_pool()
        await pool.execute(
            """
            INSERT INTO inference_logs
                (request_id, thread_id, model, prompt, response,
                 tokens_used, latency_ms, error)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            request_id,
            thread_id,
            model,
            prompt,
            response,
            tokens_used,
            latency_ms,
            error,
        )
