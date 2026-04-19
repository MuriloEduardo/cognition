"""
LLM service using LangGraph with checkpointer for automatic conversation persistence.
The checkpointer stores all messages per thread_id — no need to pass history between services.
"""

from contextlib import AsyncExitStack

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.domain.entities.cognition import CognitionRequest
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
        self._exit_stack = AsyncExitStack()

    async def _ensure_graph(self) -> None:
        if self._graph is not None:
            return

        checkpointer = await self._init_checkpointer()
        self._graph = self._compile_graph(checkpointer)
        logger.info("langgraph.compiled")

    async def _init_checkpointer(self):
        if not self._settings.database_url:
            logger.info("checkpointer.memory", reason="no database_url")
            return MemorySaver()

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpointer = await self._exit_stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(self._settings.database_url)
            )
            await checkpointer.setup()
            logger.info("checkpointer.postgres")
            return checkpointer
        except Exception as exc:
            logger.warning(
                "checkpointer.fallback_to_memory",
                error=str(exc),
                error_type=type(exc).__name__,
            )
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

    async def process(self, request: CognitionRequest) -> str:
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

            content = result["messages"][-1].content
            log.info(
                "llm.success",
                thread_id=thread_id,
                response_length=len(content) if isinstance(content, str) else 0,
            )
            return content if isinstance(content, str) else str(content)

        except Exception as exc:
            log.error("llm.failed", error=str(exc), error_type=type(exc).__name__)
            raise

    async def close(self) -> None:
        await self._exit_stack.aclose()
