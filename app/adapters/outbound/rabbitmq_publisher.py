import logging

import aio_pika

from app.infrastructure.messaging.rabbitmq_connection import RabbitMQConnection
from app.ports.outbound.message_publisher import MessagePublisher

logger = logging.getLogger(__name__)


class RabbitMQPublisher(MessagePublisher):
    def __init__(self, connection: RabbitMQConnection) -> None:
        self._connection = connection

    async def publish(
        self,
        message: bytes,
        routing_key: str,
        exchange_name: str = "",
        headers: dict | None = None,
    ) -> None:
        channel = await self._connection.get_channel()

        if exchange_name:
            exchange = await channel.declare_exchange(
                exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
            )
        else:
            exchange = channel.default_exchange

        amqp_message = aio_pika.Message(
            body=message,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers=headers,
        )

        await exchange.publish(amqp_message, routing_key=routing_key)
        logger.debug("Published message to '%s' with routing_key '%s'", exchange_name or "default", routing_key)

    async def close(self) -> None:
        await self._connection.close()
