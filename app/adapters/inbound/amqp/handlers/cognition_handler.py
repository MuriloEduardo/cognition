import asyncio
import time

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters.outbound.amqp.publisher import RabbitMQPublisher
from app.adapters.outbound.postgres import InferenceLogRepository
from app.domain.entities.cognition import CognitionRequest, CognitionResponse, LLMResult
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
        llm_result: LLMResult | None = None

        try:
            llm_result = await self._process(request)
            content = llm_result.content
            response = CognitionResponse(
                request_id=request.request_id,
                content=content,
                model=effective_model,
                model_name=llm_result.model_name,
                input_tokens=llm_result.input_tokens,
                output_tokens=llm_result.output_tokens,
                total_tokens=llm_result.total_tokens,
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
                    tokens_used=llm_result.total_tokens if llm_result else None,
                    latency_ms=latency_ms,
                    error=error_text,
                    model_name=llm_result.model_name if llm_result else None,
                    input_tokens=llm_result.input_tokens if llm_result else None,
                    output_tokens=llm_result.output_tokens if llm_result else None,
                    total_tokens=llm_result.total_tokens if llm_result else None,
                    input_token_details=(
                        llm_result.input_token_details if llm_result else None
                    ),
                    output_token_details=(
                        llm_result.output_token_details if llm_result else None
                    ),
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
    async def _process(self, request: CognitionRequest) -> LLMResult:
        """Process LLM request using LangChain."""
        return await self._llm_service.process(request)
