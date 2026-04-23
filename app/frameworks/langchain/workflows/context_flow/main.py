"""
Context-flow pipeline: extraction → evaluation → writing.

Each node receives:
  - state: the LangGraph State (messages + workdata)
  - runtime: Runtime[Context] with flow dict injected per-invocation

System prompts are Jinja2 templates rendered with `flow` and `workdata` via AttrView.

Node responsibilities:
  - extraction : JSON-mode LLM → parse content → store in workdata["extraction"]
  - evaluation : structured-output LLM (EvaluationResult) → store in workdata["evaluation"]
  - writing    : plain LLM → final human-facing reply → appended to messages
"""

import json
import time
from typing import Any

import structlog
from deepagents import create_deep_agent
from jinja2 import Environment
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime

from app.adapters.outbound.postgres.inference_log_repo import InferenceLogRepository
from app.frameworks.langchain.utils.graph_builder import build_graph_from_definition
from app.frameworks.langchain.utils.template import AttrView
from app.frameworks.langchain.workflows.context_flow.classes import (
    Context,
    EvaluationResult,
    State,
)
from app.infrastructure.config.settings import Settings

logger = structlog.get_logger(__name__)

_jinja_env = Environment()

_PYDANTIC_SCHEMAS: dict[str, type] = {
    "evaluation": EvaluationResult,
}

# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

_PIPELINE_DEFINITION: dict = {
    "nodes": [
        {
            "id": "extraction",
            "json_mode": True,
            "system_prompt": """\
{%- if flow.system_prompt %}
# Instruções Gerais
{{ flow.system_prompt }}

{%- endif %}
{%- if flow.current_node_prompt %}
# Instrução do Nó
{{ flow.current_node_prompt }}

{%- endif %}
# Tarefa
Extraia SOMENTE os campos solicitados da mensagem do usuário.
Se o valor não estiver explícito na fala, retorne null. Não invente, não deduza. Responda em JSON.
{%- if flow.properties %}

# Schema esperado
{{ flow.properties }}
{%- endif %}""",
        },
        {
            "id": "evaluation",
            "json_mode": True,
            "pydantic_schema": "evaluation",
            "system_prompt": """\
{%- if flow.system_prompt %}
# Instruções Gerais
{{ flow.system_prompt }}

{%- endif %}
{%- if flow.current_node_prompt %}
# Instrução do Nó
{{ flow.current_node_prompt }}

{%- endif %}
# Tarefa
Dados extraídos: {{ workdata.extraction }}
Edges disponíveis: {{ flow.next_edges }}

Avalie se os dados extraídos satisfazem alguma condição de edge.
Regras: se algum campo obrigatório for null/inválido, selected_edge_id deve ser null.

Responda SOMENTE em JSON com o seguinte schema:
{"selected_edge_id": "<uuid ou null>", "justification": "<texto>", "confidence": <0.0-1.0>}""",
        },
        {
            "id": "writing",
            "system_prompt": """\
{%- if flow.system_prompt %}
# Instruções Gerais
{{ flow.system_prompt }}

{%- endif %}
{%- if flow.current_node_prompt %}
# Instrução do Nó Atual
{{ flow.current_node_prompt }}

{%- endif %}
{%- if next_node_prompt %}
# Transição de Nó
A condição foi satisfeita — {{ workdata.evaluation.justification }}
O usuário será encaminhado para a próxima etapa. Use a instrução abaixo para orientar sua resposta:
{{ next_node_prompt }}

{%- elif missing_properties %}
# Campos Obrigatórios Pendentes
Os seguintes campos ainda não foram fornecidos: {{ missing_properties | join(', ') }}
Solicite-os ao usuário de forma natural e conversacional.

{%- else %}
# Estado Atual
Dados extraídos até agora: {{ workdata.extraction }}
Avaliação: {{ workdata.evaluation.justification }}

{%- endif %}
# Tarefa
Com base no histórico da conversa e nas instruções acima, elabore a próxima resposta ao usuário de forma clara, natural e humana.""",
        },
    ],
    "edges": [
        {"from": "START", "to": "extraction"},
        {"from": "extraction", "to": "evaluation"},
        {"from": "evaluation", "to": "writing"},
        {"from": "writing", "to": "END"},
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_system_prompt(
    template_str: str, *, flow: dict, workdata: dict, **extra: Any
) -> str:
    template = _jinja_env.from_string(template_str)
    return template.render(
        flow=AttrView(flow),
        workdata=AttrView(workdata),
        **extra,
    )


def _compute_missing_properties(flow: dict, workdata: dict) -> list[str]:
    """Required property names whose extracted value is null."""
    extraction = workdata.get("extraction") or {}
    missing = []
    for prop in flow.get("properties") or []:
        if isinstance(prop, dict) and prop.get("required"):
            if extraction.get(prop.get("name")) is None:
                missing.append(prop["name"])
    return missing


def _find_next_node_prompt(flow: dict, workdata: dict) -> str | None:
    """If evaluation selected an edge, return the target node's prompt."""
    selected_edge_id = (workdata.get("evaluation") or {}).get("selected_edge_id")
    if not selected_edge_id:
        return None
    for edge in flow.get("next_edges") or []:
        if isinstance(edge, dict) and str(edge.get("id")) == str(selected_edge_id):
            return edge.get("target_node_prompt")
    return None


def _build_agent(
    *,
    model_str: str,
    api_key: str,
    tools: list,
    node_def: dict,
    system_prompt: str,
) -> Any:
    """Build a deep agent for a given node.

    model_str: provider:model format (e.g. "openai:gpt-4o-mini").
    The system_prompt is delegated to deepagents, which injects it automatically.
    """
    provider = model_str.split(":")[0] if ":" in model_str else "openai"
    extra_model_kwargs: dict = {}
    if provider == "openai":
        extra_model_kwargs["api_key"] = api_key

    model = init_chat_model(model_str, **extra_model_kwargs)

    # Structured output via deepagents response_format (pydantic schema)
    response_format = None
    schema_name = node_def.get("pydantic_schema")
    if schema_name and schema_name in _PYDANTIC_SCHEMAS:
        response_format = _PYDANTIC_SCHEMAS[schema_name]

    return create_deep_agent(
        model=model,
        tools=tools or None,
        system_prompt=system_prompt,
        response_format=response_format,
    )


# ---------------------------------------------------------------------------
# Node runner
# ---------------------------------------------------------------------------


async def _run_node(
    state: State,
    *,
    runtime: Runtime[Context],
    node_def: dict,
    settings: Settings,
    log_repo: InferenceLogRepository | None = None,
) -> dict:
    flow: dict = (runtime.context or {}).get("flow") or {}
    workdata: dict = state.get("workdata") or {}
    request_id: str = (runtime.context or {}).get("request_id") or ""
    thread_id: str = (runtime.context or {}).get("thread_id") or ""

    extra: dict[str, Any] = {
        "missing_properties": _compute_missing_properties(flow, workdata),
        "next_node_prompt": _find_next_node_prompt(flow, workdata),
    }

    system_prompt_str = _render_system_prompt(
        node_def["system_prompt"], flow=flow, workdata=workdata, **extra
    )

    node_config: dict = flow.get("node_config") or {}
    model_str: str = node_config.get("model") or f"openai:{settings.default_model}"  # type: ignore[attr-defined]
    tools: list = node_config.get("tools") or []
    agent = _build_agent(
        model_str=model_str,
        api_key=settings.openai_api_key,  # type: ignore[attr-defined]
        tools=tools,
        node_def=node_def,
        system_prompt=system_prompt_str,
    )

    logger.debug(
        "node.system_prompt",
        node_id=node_def["id"],
        system_prompt=system_prompt_str,
        flow_keys=list(flow.keys()),
        workdata_keys=list(workdata.keys()),
    )

    history: list[AnyMessage] = state.get("messages", [])

    node_id = node_def["id"]

    t0 = time.monotonic()
    error_str: str | None = None
    resp_content: str | None = None
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    try:
        result = await agent.ainvoke({"messages": history})
    except Exception as exc:
        error_str = str(exc)
        latency_ms = int((time.monotonic() - t0) * 1000)
        if log_repo and request_id:
            import asyncio

            asyncio.ensure_future(
                log_repo.record_node(
                    request_id=request_id,
                    thread_id=thread_id,
                    node_id=node_id,
                    system_prompt=system_prompt_str,
                    error=error_str,
                    latency_ms=latency_ms,
                )
            )
        raise

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Extract token usage / metadata from the last AI message in the result
    result_messages: list[AnyMessage] = result.get("messages") or []
    last_ai = result_messages[-1] if result_messages else None
    if last_ai is not None:
        usage = getattr(last_ai, "usage_metadata", None) or {}
        resp_meta = getattr(last_ai, "response_metadata", None) or {}
        model_name = resp_meta.get("model_name") or resp_meta.get("model")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        resp_content = (
            last_ai.content
            if isinstance(last_ai.content, str)
            else json.dumps(last_ai.content)
        )

    # JSON mode → parse content, store in workdata only
    if node_def.get("json_mode"):
        structured = result.get("structured_response")
        if structured is not None:
            # deepagents structured output (pydantic response_format)
            parsed = (
                structured.model_dump()
                if hasattr(structured, "model_dump")
                else dict(structured)
            )
        else:
            # Plain JSON string in the last message
            try:
                parsed = json.loads(resp_content or "")
            except (json.JSONDecodeError, TypeError):
                parsed = {"raw": resp_content or ""}

        if log_repo and request_id:
            import asyncio

            asyncio.ensure_future(
                log_repo.record_node(
                    request_id=request_id,
                    thread_id=thread_id,
                    node_id=node_id,
                    system_prompt=system_prompt_str,
                    response=resp_content,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                )
            )
        return {"workdata": {node_id: parsed}}

    # Plain LLM (writing) → the last message is the AI reply
    if log_repo and request_id:
        import asyncio

        asyncio.ensure_future(
            log_repo.record_node(
                request_id=request_id,
                thread_id=thread_id,
                node_id=node_id,
                system_prompt=system_prompt_str,
                response=resp_content,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
            )
        )
    # Return only the new AI messages (those beyond the original history)
    new_messages = result_messages[len(history) :]
    return {
        "messages": new_messages if new_messages else ([last_ai] if last_ai else [])
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


async def build_context_flow(
    *, settings: Settings, log_repo: InferenceLogRepository | None = None
) -> Any:
    """Build and compile the cognition pipeline graph."""

    def create_node_handler(node_def: dict):
        async def _node(state: State, runtime: Runtime[Context]) -> dict:
            return await _run_node(
                state,
                runtime=runtime,
                node_def=node_def,
                settings=settings,
                log_repo=log_repo,
            )

        return _node

    return await build_graph_from_definition(
        _PIPELINE_DEFINITION,
        state_schema=State,
        node_handler=create_node_handler,
        context_schema=Context,
        checkpointer=MemorySaver(),
    )
