import argparse
import asyncio
import signal

import structlog
import uvicorn
from fastapi import FastAPI

from app.container import Container
from app.workers import available_workers
from app.workers.runner import WorkerRunner

# Import workers to trigger registration
import app.workers.cognition  # noqa: F401  # pylint: disable=unused-import

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cognition service")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Start the HTTP health server.",
    )
    parser.add_argument(
        "--workers",
        nargs="*",
        default=None,
        help=f"Start workers (available: {', '.join(available_workers())}). Without names = all.",
    )
    return parser.parse_args()


def create_health_app() -> FastAPI:
    health_app = FastAPI(title="Cognition Service")

    @health_app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "cognition"}

    return health_app


async def run_http(container: Container) -> None:
    settings = container.settings
    config = uvicorn.Config(
        create_health_app(),
        host=settings.http_host,
        port=settings.http_port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_workers(container: Container, names: list[str]) -> None:
    runner = WorkerRunner(container)
    await runner.start(*names)
    resolved = names or available_workers()
    logger.info("workers.running", workers=resolved)


async def main() -> None:
    args = parse_args()

    if not args.http and args.workers is None:
        logger.error("Specify at least --http or --workers")
        return

    container = Container()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    tasks: list[asyncio.Task] = []

    if args.workers is not None:
        worker_names = args.workers if args.workers else []
        tasks.append(asyncio.create_task(run_workers(container, worker_names)))

    if args.http:
        tasks.append(asyncio.create_task(run_http(container)))

    if not args.http:
        await stop.wait()

    if tasks:
        await asyncio.gather(*tasks)

    await container.shutdown()
    logger.info("shutdown.complete")


if __name__ == "__main__":
    asyncio.run(main())
