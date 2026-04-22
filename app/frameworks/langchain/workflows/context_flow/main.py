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
from jinja2 import Environment
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
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


def _build_llm(settings: Settings, *, node_def: dict) -> Any:
    kwargs: dict = dict(
        model=settings.default_model,
        temperature=settings.default_temperature,
        max_tokens=settings.default_max_tokens,
        api_key=settings.openai_api_key,
    )
    if node_def.get("json_mode"):
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


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

    # Extra template vars injected only for specific nodes
    extra: dict[str, Any] = {}
    if node_def["id"] == "writing":
        extra["missing_properties"] = _compute_missing_properties(flow, workdata)
        extra["next_node_prompt"] = _find_next_node_prompt(flow, workdata)

    system_prompt_str = _render_system_prompt(
        node_def["system_prompt"], flow=flow, workdata=workdata, **extra
    )
    llm = _build_llm(settings, node_def=node_def)

    logger.debug(
        "node.system_prompt",
        node_id=node_def["id"],
        system_prompt=system_prompt_str,
        flow_keys=list(flow.keys()),
        workdata_keys=list(workdata.keys()),
        **(
            {k: v for k, v in extra.items() if k != "missing_properties"}
            if extra
            else {}
        ),
    )

    # Always inject a fresh SystemMessage; strip any previous ones from history
    history: list[AnyMessage] = [
        m for m in state.get("messages", []) if not isinstance(m, SystemMessage)
    ]
    input_messages = [SystemMessage(content=system_prompt_str)] + history

    node_id = node_def["id"]

    t0 = time.monotonic()
    error_str: str | None = None
    resp_content: str | None = None
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    try:
        resp = await llm.ainvoke(input_messages)
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
    usage = getattr(resp, "usage_metadata", None) or {}
    resp_meta = getattr(resp, "response_metadata", None) or {}
    model_name = resp_meta.get("model_name") or resp_meta.get("model")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    resp_content = (
        resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
    )

    # JSON mode → parse content, store in workdata only
    if node_def.get("json_mode"):
        try:
            parsed = json.loads(resp.content)
        except (json.JSONDecodeError, AttributeError):
            parsed = {"raw": getattr(resp, "content", str(resp))}
        # Optional Pydantic validation (e.g. evaluation node)
        schema_name = node_def.get("pydantic_schema")
        if schema_name and schema_name in _PYDANTIC_SCHEMAS:
            try:
                model_cls = _PYDANTIC_SCHEMAS[schema_name]
                parsed = model_cls.model_validate(parsed).model_dump()
            except Exception:
                pass  # keep raw dict if validation fails
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

    # Plain LLM (writing) → append to messages for conversation continuity
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
    return {"messages": [resp]}


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
