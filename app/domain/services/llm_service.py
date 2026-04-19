"""
Simple LLM service using LangChain.
Based on references from previous projects but simplified for current architecture.
"""

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.domain.entities.cognition import CognitionRequest, WorkflowContext
from app.infrastructure.config.settings import Settings

logger = structlog.get_logger(__name__)


class LLMService:
    """Simple LLM service using LangChain ChatOpenAI."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm: ChatOpenAI | None = None

    def _get_llm(self, model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
        """Get or create LLM instance."""
        model_name = model or self._settings.default_model
        temp = temperature if temperature is not None else self._settings.default_temperature

        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            max_tokens=self._settings.default_max_tokens,
            api_key=self._settings.openai_api_key,
        )

    def _build_system_prompt(self, context: WorkflowContext | None) -> str:
        """Build system prompt based on context."""
        base_prompt = """Você é um assistente de atendimento ao cliente profissional e prestativo.

Suas responsabilidades:
- Responder perguntas de forma clara e objetiva
- Manter um tom profissional mas amigável
- Solicitar informações adicionais quando necessário
- Fornecer soluções práticas para problemas dos clientes
"""

        if not context:
            return base_prompt

        # Add context information
        context_info = []

        if context.user_id:
            context_info.append(f"Cliente ID: {context.user_id}")

        if context.conversation_id:
            context_info.append(f"Conversa ID: {context.conversation_id}")

        if context.state:
            context_info.append(f"Estado atual: {context.state}")

        if context.metadata:
            language = context.metadata.get("language", "pt-BR")
            context_info.append(f"Idioma: {language}")

        if context_info:
            context_section = "\n\nContexto da conversa:\n" + "\n".join(f"- {info}" for info in context_info)
            return base_prompt + context_section

        return base_prompt

    async def process(self, request: CognitionRequest) -> str:
        """
        Process LLM request.

        Args:
            request: CognitionRequest with prompt and optional context

        Returns:
            Generated response content
        """
        log = logger.bind(
            request_id=request.request_id,
            model=request.model,
            prompt_length=len(request.prompt),
        )

        log.info("llm.processing")

        try:
            llm = self._get_llm(
                model=request.model,
                temperature=request.temperature,
            )

            system_prompt = self._build_system_prompt(request.context)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=request.prompt),
            ]

            response = await llm.ainvoke(messages)

            content = response.content

            log.info(
                "llm.success",
                response_length=len(content) if isinstance(content, str) else 0,
            )

            return content if isinstance(content, str) else str(content)

        except Exception as exc:
            log.error("llm.failed", error=str(exc), error_type=type(exc).__name__)
            raise
