import asyncio
import signal

import structlog

from app.adapters.inbound.generate_handler import GenerateHandler
from app.container import Container

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger(__name__)

EXCHANGE = "cognition.exchange"
QUEUE = "generate.request"
ROUTING_KEY = "generate.request"


async def main() -> None:
    container = Container()
    logger.info("starting", app=container.settings.app_name)

    await container.connection.connect()

    handler = GenerateHandler(publisher=container.publisher)
    consumer = container.consumer(handler)
    await consumer.start_consuming(
        queue_name=QUEUE,
        exchange_name=EXCHANGE,
        routing_key=ROUTING_KEY,
        prefetch_count=container.settings.rabbitmq_prefetch_count,
    )

    logger.info("consuming", queue=QUEUE, exchange=EXCHANGE)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    await container.shutdown()
    logger.info("shutdown.complete")


if __name__ == "__main__":
    asyncio.run(main())
