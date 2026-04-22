import aio_pika
import structlog
from aio_pika.abc import AbstractIncomingMessage

from app.infrastructure.messaging.rabbitmq_connection import RabbitMQConnection
from app.ports.inbound.message_handler import MessageHandler

logger = structlog.get_logger(__name__)

DLX_EXCHANGE = "dead.letter"
DLQ_NAME = "dlq.messages"
RETRY_HEADER = "x-retry-count"
MAX_RETRIES = 3


class RabbitMQConsumer:
    def __init__(self, connection: RabbitMQConnection, handler: MessageHandler) -> None:
        self._connection = connection
        self._handler = handler
        self._channel = None
        self._queue_name: str = ""

    async def start_consuming(
        self,
        queue_name: str,
        exchange_name: str = "",
        routing_key: str = "",
        prefetch_count: int = 10,
    ) -> None:
        channel = await self._connection.get_channel()
        await channel.set_qos(prefetch_count=prefetch_count)
        self._channel = channel
        self._queue_name = queue_name

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
        headers = dict(message.headers) if message.headers else {}
        retry_count = int(headers.get(RETRY_HEADER, 0))

        try:
            await self._handler.handle(
                message=message.body,
                routing_key=message.routing_key or "",
                headers=headers,
            )
            await message.ack()
        except Exception:
            logger.exception(
                "message.failed",
                message_id=message.message_id,
                retry_count=retry_count,
                max_retries=MAX_RETRIES,
            )
            if retry_count < MAX_RETRIES:
                await message.ack()
                retry_msg = aio_pika.Message(
                    body=message.body,
                    headers={**headers, RETRY_HEADER: retry_count + 1},
                    message_id=message.message_id,
                    content_type=message.content_type,
                )
                await self._channel.default_exchange.publish(
                    retry_msg,
                    routing_key=self._queue_name,
                )
                logger.warning(
                    "message.retrying",
                    message_id=message.message_id,
                    attempt=retry_count + 1,
                    max_retries=MAX_RETRIES,
                )
            else:
                await message.nack(requeue=False)
                logger.error(
                    "message.dead_lettered",
                    message_id=message.message_id,
                    queue=self._queue_name,
                )
