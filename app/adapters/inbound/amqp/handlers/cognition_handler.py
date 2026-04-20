import asyncio
import time

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters.outbound.amqp.publisher import RabbitMQPublisher
from app.adapters.outbound.postgres import InferenceLogRepository
from app.domain.entities.cognition import CognitionRequest, CognitionResponse
from app.domain.services.llm_service import LLMService
from app.ports.inbound.message_handler import MessageHandler

logger = structlog.get_logger(__name__)

EXCHANGE = "cognition.exchange"
RESPONSE_KEY = "cognition.response"


class CognitionHandler(MessageHandler):
    """
    Handles cognition/AI processing requests.
    Cognition service processes LLM requests and returns results.
    """

    def __init__(
        self,
        publisher: RabbitMQPublisher,
        llm_service: LLMService,
        inference_logs: InferenceLogRepository | None = None,
    ) -> None:
        self._publisher = publisher
        self._llm_service = llm_service
        self._inference_logs = inference_logs

    async def handle(
        self, message: bytes, routing_key: str, headers: dict | None = None
    ) -> None:
        if not message:
            logger.warning("cognition.empty_message", routing_key=routing_key)
            return

        request = CognitionRequest.model_validate_json(message)
        effective_model = request.model or "default"
        log = logger.bind(request_id=request.request_id, model=effective_model)
        log.info("cognition.received")

        t0 = time.monotonic()
        error_text: str | None = None
        content = ""

        try:
            content = await self._process(request)
            response = CognitionResponse(
                request_id=request.request_id,
                content=content,
                model=effective_model,
                context=request.context,
            )
        except Exception as exc:
            error_text = str(exc)
            log.error("cognition.failed", error=error_text)
            response = CognitionResponse(
                request_id=request.request_id,
                content="",
                model=effective_model,
                error=error_text,
                context=request.context,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)

        if self._inference_logs:
            asyncio.create_task(
                self._inference_logs.record(
                    request_id=request.request_id,
                    thread_id=getattr(request.context, "session_id", "")
                    or request.request_id,
                    model=effective_model,
                    prompt=request.prompt,
                    response=content or None,
                    latency_ms=latency_ms,
                    error=error_text,
                )
            )

        await self._publisher.publish(
            message=response.model_dump_json().encode(),
            routing_key=RESPONSE_KEY,
            exchange_name=EXCHANGE,
        )
        log.info("cognition.responded", has_error=response.error is not None)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True
    )
    async def _process(self, request: CognitionRequest) -> str:
        """Process LLM request using LangChain."""
        return await self._llm_service.process(request)
