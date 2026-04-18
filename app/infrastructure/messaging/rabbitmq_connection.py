import asyncio
import logging

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractRobustConnection

from app.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class RabbitMQConnection:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None

    async def connect(self) -> AbstractRobustConnection:
        if self._connection and not self._connection.is_closed:
            return self._connection

        for attempt in range(1, self._settings.rabbitmq_max_retries + 1):
            try:
                self._connection = await aio_pika.connect_robust(
                    str(self._settings.rabbitmq_url),
                )
                logger.info(
                    "Connected to RabbitMQ at %s", self._settings.rabbitmq_url.host
                )
                return self._connection
            except Exception:
                logger.warning(
                    "Connection attempt %d/%d failed",
                    attempt,
                    self._settings.rabbitmq_max_retries,
                )
                if attempt < self._settings.rabbitmq_max_retries:
                    await asyncio.sleep(self._settings.rabbitmq_reconnect_delay)
                else:
                    logger.exception(
                        "Failed to connect to RabbitMQ after %d attempts", attempt
                    )
                    raise

        raise RuntimeError("Unreachable")

    async def get_channel(self) -> AbstractChannel:
        if self._channel and not self._channel.is_closed:
            return self._channel

        connection = await self.connect()
        self._channel = await connection.channel()
        return self._channel

    async def close(self) -> None:
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        logger.info("RabbitMQ connection closed")
