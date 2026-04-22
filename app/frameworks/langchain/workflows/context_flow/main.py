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
from typing import Any

import structlog
from jinja2 import Environment
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime

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
# Instrução do Nó
{{ flow.current_node_prompt }}

{%- endif %}
# Tarefa
Avaliação: {{ workdata.evaluation }}

Com base na avaliação, elabore a próxima resposta ao usuário de forma clara e humana.
Se a condição não foi satisfeita, solicite os dados faltantes.""",
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


def _render_system_prompt(template_str: str, *, flow: dict, workdata: dict) -> str:
    template = _jinja_env.from_string(template_str)
    return template.render(
        flow=AttrView(flow),
        workdata=AttrView(workdata),
    )


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
) -> dict:
    flow: dict = (runtime.context or {}).get("flow") or {}
    workdata: dict = state.get("workdata") or {}

    system_prompt_str = _render_system_prompt(
        node_def["system_prompt"], flow=flow, workdata=workdata
    )
    llm = _build_llm(settings, node_def=node_def)

    # Always inject a fresh SystemMessage; strip any previous ones from history
    history: list[AnyMessage] = [
        m for m in state.get("messages", []) if not isinstance(m, SystemMessage)
    ]
    input_messages = [SystemMessage(content=system_prompt_str)] + history

    node_id = node_def["id"]

    resp = await llm.ainvoke(input_messages)

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
        return {"workdata": {node_id: parsed}}

    # Plain LLM (writing) → append to messages for conversation continuity
    return {"messages": [resp]}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


async def build_context_flow(*, settings: Settings) -> Any:
    """Build and compile the cognition pipeline graph."""

    def create_node_handler(node_def: dict):
        async def _node(state: State, runtime: Runtime[Context]) -> dict:
            return await _run_node(
                state,
                runtime=runtime,
                node_def=node_def,
                settings=settings,
            )

        return _node

    return await build_graph_from_definition(
        _PIPELINE_DEFINITION,
        state_schema=State,
        node_handler=create_node_handler,
        context_schema=Context,
        checkpointer=MemorySaver(),
    )
