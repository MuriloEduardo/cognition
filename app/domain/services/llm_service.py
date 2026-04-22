"""
LLM service using LangGraph with a 3-node pipeline driven by workflow context.

Graph: extraction → evaluation → writing

- extraction: extracts structured data from the user message using the node's
  response_format (properties schema) defined in the workflow.
- evaluation: decides which next_edge to take based on the extracted data and
  the edge condition_prompts. Returns selected_edge_id, justification, confidence.
- writing: composes the final human-facing reply based on the current node prompt
  and the evaluation result.
"""

import json
from typing import Any, Annotated, NotRequired, TypedDict

import structlog
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langchain_core.runnables import RunnableConfig

from app.domain.entities.cognition import CognitionRequest, LLMResult
from app.infrastructure.config.settings import Settings

logger = structlog.get_logger(__name__)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    workdata: NotRequired[dict[str, Any]]


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._graph = None

    async def _ensure_graph(self) -> None:
        if self._graph is not None:
            return
        self._graph = self._compile_graph(MemorySaver())
        logger.info("langgraph.compiled")

    def _llm(self, *, json_mode: bool = False) -> ChatOpenAI:
        kwargs: dict = dict(
            model=self._settings.default_model,
            temperature=self._settings.default_temperature,
            max_tokens=self._settings.default_max_tokens,
            api_key=self._settings.openai_api_key,
        )
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    def _compile_graph(self, checkpointer) -> Any:
        extraction_llm = self._llm(json_mode=True)
        evaluation_llm = self._llm(json_mode=True)
        writing_llm = self._llm(json_mode=False)

        async def extraction(state: State, *, flow: dict) -> dict:
            system_prompt = flow.get("system_prompt") or ""
            node_prompt = flow.get("current_node_prompt") or ""
            schema_props: list[dict] = flow.get("properties") or []

            system_parts = []
            if system_prompt:
                system_parts.append(f"# Instruções Gerais\n{system_prompt}")
            if node_prompt:
                system_parts.append(f"# Instrução do Nó\n{node_prompt}")
            system_parts.append(
                "# Tarefa\n"
                "Extraia SOMENTE os campos solicitados.\n"
                "Se o valor não estiver explícito na fala do usuário, retorne null.\n"
                "Não invente, não deduza. Responda em JSON."
            )
            if schema_props:
                system_parts.append(
                    f"# Schema esperado\n{json.dumps(schema_props, ensure_ascii=False)}"
                )

            resp = await extraction_llm.ainvoke(
                [SystemMessage(content="\n\n".join(system_parts))]
                + list(state["messages"])
            )
            try:
                extracted = json.loads(resp.content)
            except json.JSONDecodeError:
                extracted = {"raw": resp.content}

            return {
                "messages": [resp],
                "workdata": {"extraction": extracted},
            }

        async def evaluation(state: State, *, flow: dict) -> dict:
            system_prompt = flow.get("system_prompt") or ""
            node_prompt = flow.get("current_node_prompt") or ""
            next_edges = flow.get("next_edges") or []
            extracted = (state.get("workdata") or {}).get("extraction", {})

            edges_desc = json.dumps(
                [
                    {
                        "id": e.get("id"),
                        "condition_prompt": e.get("condition_prompt"),
                        "target_node_id": e.get("target_node_id"),
                    }
                    for e in next_edges
                ],
                ensure_ascii=False,
            )

            system_parts = []
            if system_prompt:
                system_parts.append(f"# Instruções Gerais\n{system_prompt}")
            if node_prompt:
                system_parts.append(f"# Instrução do Nó\n{node_prompt}")
            system_parts.append(
                f"# Tarefa\n"
                f"Dados extraídos: {json.dumps(extracted, ensure_ascii=False)}\n"
                f"Edges disponíveis: {edges_desc}\n\n"
                "Avalie se os dados satisfazem alguma condição de edge.\n"
                "Regras: se algum campo obrigatório for null/inválido, selected_edge_id deve ser null.\n"
                "Responda em JSON com os campos: selected_edge_id (string uuid ou null), "
                "justification (string), confidence (float 0-1)."
            )

            resp = await evaluation_llm.ainvoke(
                [SystemMessage(content="\n\n".join(system_parts))]
                + list(state["messages"])
            )
            try:
                evaluation_result = json.loads(resp.content)
            except json.JSONDecodeError:
                evaluation_result = {
                    "selected_edge_id": None,
                    "justification": resp.content,
                    "confidence": 0.0,
                }

            return {
                "messages": [resp],
                "workdata": {
                    **(state.get("workdata") or {}),
                    "evaluation": evaluation_result,
                },
            }

        async def writing(state: State, *, flow: dict) -> dict:
            system_prompt = flow.get("system_prompt") or ""
            node_prompt = flow.get("current_node_prompt") or ""
            evaluation_result = (state.get("workdata") or {}).get("evaluation", {})

            system_parts = []
            if system_prompt:
                system_parts.append(f"# Instruções Gerais\n{system_prompt}")
            if node_prompt:
                system_parts.append(f"# Instrução do Nó\n{node_prompt}")
            system_parts.append(
                f"# Tarefa\n"
                f"Avaliação: {json.dumps(evaluation_result, ensure_ascii=False)}\n\n"
                "Com base na avaliação, elabore a próxima resposta ao usuário de forma clara e humana.\n"
                "Se a condição não foi satisfeita, solicite os dados faltantes."
            )

            resp = await writing_llm.ainvoke(
                [SystemMessage(content="\n\n".join(system_parts))]
                + list(state["messages"])
            )
            return {"messages": [resp]}

        # Graph is compiled once; flow context is injected per-invocation via closure trick:
        # We store flow in config["configurable"]["flow"] and access via the node functions.
        # LangGraph passes only (state, config) — we use a wrapper to extract flow.

        def make_node(fn):
            async def wrapper(state: State, config: RunnableConfig) -> dict:
                raw = (config.get("configurable") or {}).get("flow")
                if isinstance(raw, str):
                    try:
                        flow = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        flow = {}
                elif isinstance(raw, dict):
                    flow = raw
                else:
                    flow = {}
                return await fn(state, flow=flow)

            return wrapper

        graph = StateGraph(State)
        graph.add_node("extraction", make_node(extraction))
        graph.add_node("evaluation", make_node(evaluation))
        graph.add_node("writing", make_node(writing))
        graph.add_edge(START, "extraction")
        graph.add_edge("extraction", "evaluation")
        graph.add_edge("evaluation", "writing")
        graph.add_edge("writing", END)
        return graph.compile(checkpointer=checkpointer)

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
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "flow": flow or {},
                }
            },
        )

        # Final content = last writing node message
        ai_message = result["messages"][-1]
        content = (
            ai_message.content
            if isinstance(ai_message.content, str)
            else str(ai_message.content)
        )

        # Evaluation result carries selected_edge_id, justification, confidence
        evaluation_result: dict = (result.get("workdata") or {}).get("evaluation") or {}

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
            selected_edge_id=evaluation_result.get("selected_edge_id"),
            justification=evaluation_result.get("justification"),
            confidence=evaluation_result.get("confidence"),
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
