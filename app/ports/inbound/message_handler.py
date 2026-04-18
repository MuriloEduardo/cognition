from abc import ABC, abstractmethod


class MessageHandler(ABC):
    @abstractmethod
    async def handle(self, message: bytes, routing_key: str, headers: dict | None = None) -> None:
        raise NotImplementedError
