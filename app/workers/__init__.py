from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

type HandlerFactory = Callable[..., Any]


@dataclass(frozen=True)
class WorkerConfig:
    name: str
    queue: str
    exchange: str
    routing_key: str
    handler_factory: HandlerFactory
    prefetch_count: int | None = None


_registry: dict[str, WorkerConfig] = {}


def worker(
    name: str,
    queue: str,
    exchange: str,
    routing_key: str,
    prefetch_count: int | None = None,
):
    def decorator(factory: HandlerFactory) -> HandlerFactory:
        _registry[name] = WorkerConfig(
            name=name,
            queue=queue,
            exchange=exchange,
            routing_key=routing_key,
            handler_factory=factory,
            prefetch_count=prefetch_count,
        )
        return factory

    return decorator


def get_workers(*names: str) -> list[WorkerConfig]:
    if not names:
        return list(_registry.values())
    return [_registry[n] for n in names if n in _registry]


def available_workers() -> list[str]:
    return list(_registry.keys())
