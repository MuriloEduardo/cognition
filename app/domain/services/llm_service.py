"""
LLM service using LangGraph with checkpointer for automatic conversation persistence.
The checkpointer stores all messages per thread_id — no need to pass history between services.
"""

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.domain.entities.cognition import CognitionRequest, LLMResult
from app.infrastructure.config.settings import Settings

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """Você é um assistente de atendimento ao cliente profissional e prestativo.

Suas responsabilidades:
- Responder perguntas de forma clara e objetiva
- Manter um tom profissional mas amigável
- Solicitar informações adicionais quando necessário
- Fornecer soluções práticas para problemas dos clientes
"""


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._graph = None

    async def _ensure_graph(self) -> None:
        if self._graph is not None:
            return

        checkpointer = await self._init_checkpointer()
        self._graph = self._compile_graph(checkpointer)
        logger.info("langgraph.compiled")

    async def _init_checkpointer(self):
        logger.info("checkpointer.memory")
        return MemorySaver()

    def _compile_graph(self, checkpointer):
        settings = self._settings

        async def call_model(state: MessagesState) -> dict:
            model = settings.default_model
            llm = ChatOpenAI(
                model=model,
                temperature=settings.default_temperature,
                max_tokens=settings.default_max_tokens,
                api_key=settings.openai_api_key,
            )

            messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", call_model)
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        return graph.compile(checkpointer=checkpointer)

    async def process(self, request: CognitionRequest) -> LLMResult:
        log = logger.bind(
            request_id=request.request_id,
            model=request.model,
            prompt_length=len(request.prompt),
        )

        log.info("llm.processing")

        try:
            await self._ensure_graph()

            thread_id = (
                request.context.session_id if request.context else request.request_id
            )

            result = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=request.prompt)]},
                config={"configurable": {"thread_id": thread_id}},
            )

            ai_message = result["messages"][-1]
            content = ai_message.content
            if not isinstance(content, str):
                content = str(content)

            usage = getattr(ai_message, "usage_metadata", None) or {}
            resp_meta = getattr(ai_message, "response_metadata", None) or {}

            llm_result = LLMResult(
                content=content,
                model_name=resp_meta.get("model_name") or resp_meta.get("model"),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                input_token_details=usage.get("input_token_details"),
                output_token_details=usage.get("output_token_details"),
            )

            log.info(
                "llm.success",
                thread_id=thread_id,
                response_length=len(content),
                input_tokens=llm_result.input_tokens,
                output_tokens=llm_result.output_tokens,
                total_tokens=llm_result.total_tokens,
                model_name=llm_result.model_name,
            )
            return llm_result

        except Exception as exc:
            log.error("llm.failed", error=str(exc), error_type=type(exc).__name__)
            raise

    async def close(self) -> None:
        pass
