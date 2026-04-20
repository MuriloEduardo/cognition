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
    parser = argparse.ArgumentParser(description="Cognition workers")
    parser.add_argument(
        "--workers",
        nargs="*",
        default=[],
        help=f"Workers to start (available: {', '.join(available_workers())}). Empty = all.",
    )
    return parser.parse_args()


def create_health_app() -> FastAPI:
    app = FastAPI(title="Cognition Service")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "cognition"}

    return app


async def main() -> None:
    args = parse_args()
    container = Container()

    runner = WorkerRunner(container)
    await runner.start(*args.workers)

    names = args.workers or available_workers()
    logger.info("workers.running", workers=names)

    settings = container.settings
    config = uvicorn.Config(
        create_health_app(),
        host=settings.http_host,
        port=settings.http_port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    http_task = asyncio.create_task(server.serve())

    await stop.wait()
    server.should_exit = True
    await http_task
    await container.shutdown()
    logger.info("shutdown.complete")


if __name__ == "__main__":
    asyncio.run(main())
