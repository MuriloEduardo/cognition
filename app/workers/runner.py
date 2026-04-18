import structlog

from app.container import Container
from app.workers import WorkerConfig, get_workers

logger = structlog.get_logger(__name__)


class WorkerRunner:
    def __init__(self, container: Container) -> None:
        self._container = container

    async def start(self, *names: str) -> None:
        workers = get_workers(*names)
        if not workers:
            logger.warning("no.workers.found", requested=names)
            return

        await self._container.connection.connect()

        for w in workers:
            await self._start_worker(w)

    async def _start_worker(self, config: WorkerConfig) -> None:
        handler = config.handler_factory(self._container)
        consumer = self._container.consumer(handler)
        prefetch = (
            config.prefetch_count or self._container.settings.rabbitmq_prefetch_count
        )

        await consumer.start_consuming(
            queue_name=config.queue,
            exchange_name=config.exchange,
            routing_key=config.routing_key,
            prefetch_count=prefetch,
        )
        logger.info(
            "worker.started",
            worker=config.name,
            queue=config.queue,
            exchange=config.exchange,
        )
