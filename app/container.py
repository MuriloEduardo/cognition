import structlog

from app.adapters.inbound.amqp.consumer import RabbitMQConsumer
from app.adapters.outbound.amqp.publisher import RabbitMQPublisher
from app.domain.services.llm_service import LLMService
from app.infrastructure.config.settings import Settings
from app.infrastructure.messaging.rabbitmq_connection import RabbitMQConnection
from app.ports.inbound.message_handler import MessageHandler

logger = structlog.get_logger(__name__)


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._connection: RabbitMQConnection | None = None
        self._publisher: RabbitMQPublisher | None = None
        self._llm_service: LLMService | None = None

    @property
    def connection(self) -> RabbitMQConnection:
        if self._connection is None:
            self._connection = RabbitMQConnection(self.settings)
        return self._connection

    @property
    def publisher(self) -> RabbitMQPublisher:
        if self._publisher is None:
            self._publisher = RabbitMQPublisher(self.connection)
        return self._publisher

    @property
    def llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._llm_service = LLMService(self.settings)
        return self._llm_service

    def consumer(self, handler: MessageHandler) -> RabbitMQConsumer:
        return RabbitMQConsumer(self.connection, handler)

    async def shutdown(self) -> None:
        if self._connection:
            await self._connection.close()
        logger.info("container.shutdown")
