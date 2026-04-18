import asyncio
import logging
import signal

from app.container import Container
from app.ports.inbound.message_handler import MessageHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class ExampleHandler(MessageHandler):
    async def handle(self, message: bytes, routing_key: str, headers: dict | None = None) -> None:
        logger.info("Received [%s]: %s", routing_key, message.decode())


async def main() -> None:
    container = Container()
    logger.info("Starting %s...", container.settings.app_name)

    # Connect
    await container.connection.connect()

    # Example: publish a message
    await container.publisher.publish(
        message=b'{"hello": "world"}',
        routing_key="test.queue",
    )
    logger.info("Message published successfully")

    # Example: start consuming
    handler = ExampleHandler()
    consumer = container.consumer(handler)
    await consumer.start_consuming(queue_name="test.queue")

    # Keep running until interrupted
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    await container.shutdown()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
