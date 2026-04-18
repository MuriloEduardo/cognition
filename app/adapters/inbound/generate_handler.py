import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters.outbound.rabbitmq_publisher import RabbitMQPublisher
from app.domain.entities.generate import GenerateRequest, GenerateResponse
from app.ports.inbound.message_handler import MessageHandler

logger = structlog.get_logger(__name__)

EXCHANGE = "cognition.exchange"
RESPONSE_KEY = "generate.response"


class GenerateHandler(MessageHandler):
    def __init__(self, publisher: RabbitMQPublisher) -> None:
        self._publisher = publisher

    async def handle(self, message: bytes, routing_key: str, headers: dict | None = None) -> None:
        if not message:
            logger.warning("generate.empty_message", routing_key=routing_key)
            return

        request = GenerateRequest.model_validate_json(message)
        log = logger.bind(request_id=request.request_id, model=request.model)
        log.info("generate.received")

        try:
            content = await self._process(request)
            response = GenerateResponse(
                request_id=request.request_id,
                content=content,
                model=request.model,
            )
        except Exception as exc:
            log.error("generate.failed", error=str(exc))
            response = GenerateResponse(
                request_id=request.request_id,
                content="",
                model=request.model,
                error=str(exc),
            )

        await self._publisher.publish(
            message=response.model_dump_json().encode(),
            routing_key=RESPONSE_KEY,
            exchange_name=EXCHANGE,
        )
        log.info("generate.responded", has_error=response.error is not None)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    async def _process(self, request: GenerateRequest) -> str:
        # TODO: plug your LLM call here (LangChain, OpenAI, etc.)
        return f"Echo: {request.prompt}"
