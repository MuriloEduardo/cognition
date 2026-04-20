import structlog

from app.adapters.inbound.amqp.consumer import RabbitMQConsumer
from app.adapters.outbound.amqp.publisher import RabbitMQPublisher
from app.adapters.outbound.postgres import InferenceLogRepository
from app.domain.services.llm_service import LLMService
from app.infrastructure.config.settings import Settings
from app.infrastructure.database import PostgresConnection
from app.infrastructure.messaging.rabbitmq_connection import RabbitMQConnection
from app.ports.inbound.message_handler import MessageHandler

logger = structlog.get_logger(__name__)


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._connection: RabbitMQConnection | None = None
        self._publisher: RabbitMQPublisher | None = None
        self._llm_service: LLMService | None = None
        self._database: PostgresConnection | None = None
        self._inference_logs: InferenceLogRepository | None = None

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

    @property
    def database(self) -> PostgresConnection:
        if self._database is None:
            self._database = PostgresConnection(self.settings)
        return self._database

    @property
    def inference_logs(self) -> InferenceLogRepository:
        if self._inference_logs is None:
            self._inference_logs = InferenceLogRepository(self.database)
        return self._inference_logs

    def consumer(self, handler: MessageHandler) -> RabbitMQConsumer:
        return RabbitMQConsumer(self.connection, handler)

    async def shutdown(self) -> None:
        if self._llm_service:
            await self._llm_service.close()
        if self._database:
            await self._database.close()
        if self._connection:
            await self._connection.close()
        logger.info("container.shutdown")
