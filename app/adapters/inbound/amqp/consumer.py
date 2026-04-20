import aio_pika
import structlog
from aio_pika.abc import AbstractIncomingMessage

from app.infrastructure.messaging.rabbitmq_connection import RabbitMQConnection
from app.ports.inbound.message_handler import MessageHandler

logger = structlog.get_logger(__name__)

DLX_EXCHANGE = "dead.letter"
DLQ_NAME = "dlq.messages"


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

        dlx = await channel.declare_exchange(
            DLX_EXCHANGE, aio_pika.ExchangeType.TOPIC, durable=True
        )
        dlq = await channel.declare_queue(DLQ_NAME, durable=True)
        await dlq.bind(dlx, routing_key="#")

        if exchange_name:
            exchange = await channel.declare_exchange(
                exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
            )
        else:
            exchange = None

        queue = await channel.declare_queue(
            queue_name, durable=True, arguments={"x-dead-letter-exchange": DLX_EXCHANGE}
        )

        if exchange and routing_key:
            await queue.bind(exchange, routing_key=routing_key)

        logger.info("Started consuming from queue '%s'", queue_name)
        await queue.consume(self._on_message)

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        async with message.process(requeue=False):
            try:
                headers = dict(message.headers) if message.headers else None
                await self._handler.handle(
                    message=message.body,
                    routing_key=message.routing_key or "",
                    headers=headers,
                )
            except Exception:
                logger.exception(
                    "message.failed",
                    queue=message.routing_key,
                    message_id=message.message_id,
                )
                raise
