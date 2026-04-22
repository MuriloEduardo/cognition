"""
LLM service — thin orchestrator over the context-flow pipeline.

Pipeline: extraction → evaluation → writing
Built in: app/frameworks/langchain/workflows/context_flow/

The graph is compiled once and reused across requests (MemorySaver preserves
conversation history per thread_id). Per-invocation workflow context (flow)
is injected via the LangGraph Runtime context mechanism.
"""

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage

from app.domain.entities.cognition import CognitionRequest, LLMResult
from app.frameworks.langchain.workflows.context_flow.main import build_context_flow
from app.infrastructure.config.settings import Settings

logger = structlog.get_logger(__name__)


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._graph: Any = None

    async def _ensure_graph(self) -> None:
        if self._graph is not None:
            return
        self._graph = await build_context_flow(settings=self._settings)
        logger.info("langgraph.compiled")

    async def process(self, request: CognitionRequest) -> LLMResult:
        log = logger.bind(request_id=request.request_id)
        log.info("llm.processing")

        await self._ensure_graph()

        thread_id = (
            request.context.session_id if request.context else request.request_id
        )

        raw_flow = (
            (request.context.state or {}).get("flow") if request.context else None
        )
        if isinstance(raw_flow, str):
            try:
                flow: dict = json.loads(raw_flow)
            except (json.JSONDecodeError, TypeError):
                flow = {}
        elif isinstance(raw_flow, dict):
            flow = raw_flow
        else:
            flow = {}

        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=request.prompt)]},
            config={"configurable": {"thread_id": thread_id}},
            context={"flow": flow},
        )

        # Last message is always the writing node's reply
        ai_message = result["messages"][-1]
        content = (
            ai_message.content
            if isinstance(ai_message.content, str)
            else str(ai_message.content)
        )

        evaluation: dict = (result.get("workdata") or {}).get("evaluation") or {}

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
            selected_edge_id=evaluation.get("selected_edge_id"),
            justification=evaluation.get("justification"),
            confidence=evaluation.get("confidence"),
        )

        log.info(
            "llm.success",
            thread_id=thread_id,
            selected_edge_id=llm_result.selected_edge_id,
            confidence=llm_result.confidence,
        )
        return llm_result

    async def close(self) -> None:
        pass
