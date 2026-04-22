import structlog

from app.adapters.inbound.amqp.consumer import RabbitMQConsumer
from app.adapters.outbound.amqp.publisher import RabbitMQPublisher
from app.adapters.outbound.postgres.agent_repo import (
    PostgresAgentRepository,
    PostgresWorkflowEdgeRepository,
    PostgresWorkflowNodeRepository,
)
from app.adapters.outbound.postgres.inference_log_repo import InferenceLogRepository
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
        self._agent_repo: PostgresAgentRepository | None = None
        self._agent_node_repo: PostgresWorkflowNodeRepository | None = None
        self._agent_edge_repo: PostgresWorkflowEdgeRepository | None = None

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
            self._llm_service = LLMService(self.settings, log_repo=self.inference_logs)
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

    @property
    def agent_repo(self) -> PostgresAgentRepository:
        if self._agent_repo is None:
            self._agent_repo = PostgresAgentRepository(self.database)
        return self._agent_repo

    @property
    def agent_node_repo(self) -> PostgresWorkflowNodeRepository:
        if self._agent_node_repo is None:
            self._agent_node_repo = PostgresWorkflowNodeRepository(self.database)
        return self._agent_node_repo

    @property
    def agent_edge_repo(self) -> PostgresWorkflowEdgeRepository:
        if self._agent_edge_repo is None:
            self._agent_edge_repo = PostgresWorkflowEdgeRepository(self.database)
        return self._agent_edge_repo

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
