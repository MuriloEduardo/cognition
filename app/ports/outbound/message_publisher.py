from abc import ABC, abstractmethod


class MessagePublisher(ABC):
    @abstractmethod
    async def publish(
        self,
        message: bytes,
        routing_key: str,
        exchange_name: str = "",
        headers: dict | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
