import logging

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from app.infrastructure.messaging.rabbitmq_connection import RabbitMQConnection
from app.ports.inbound.message_handler import MessageHandler

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    def __init__(self, connection: RabbitMQConnection, handler: MessageHandler) -> None:
        self._connection = connection
        self._handler = handler

    async def start_consuming(
        self,
        queue_name: str,
        exchange_name: str = "",
        routing_key: str = "",
        prefetch_count: int = 10,
    ) -> None:
        channel = await self._connection.get_channel()
        await channel.set_qos(prefetch_count=prefetch_count)

        if exchange_name:
            exchange = await channel.declare_exchange(
                exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
            )
        else:
            exchange = None

        queue = await channel.declare_queue(queue_name, durable=True)

        if exchange and routing_key:
            await queue.bind(exchange, routing_key=routing_key)

        logger.info("Started consuming from queue '%s'", queue_name)
        await queue.consume(self._on_message)

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        async with message.process():
            try:
                headers = dict(message.headers) if message.headers else None
                await self._handler.handle(
                    message=message.body,
                    routing_key=message.routing_key or "",
                    headers=headers,
                )
                logger.debug("Message processed: %s", message.message_id)
            except Exception:
                logger.exception("Error processing message: %s", message.message_id)
                raise
