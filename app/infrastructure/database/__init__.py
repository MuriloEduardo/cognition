from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import asyncpg

if TYPE_CHECKING:
    from app.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class PostgresConnection:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: Any = None
        self._closed = True

    async def connect(self) -> Any:
        if self._pool and not self._closed:
            return self._pool

        self._pool = await asyncpg.create_pool(
            str(self._settings.database_url),
            min_size=1,
            max_size=5,
        )
        self._closed = False
        logger.info("Connected to PostgreSQL")
        return self._pool

    async def get_pool(self) -> Any:
        if self._pool is None or self._closed:
            return await self.connect()
        return self._pool

    async def close(self) -> None:
        if self._pool and not self._closed:
            await self._pool.close()
            self._closed = True
            logger.info("PostgreSQL connection closed")
