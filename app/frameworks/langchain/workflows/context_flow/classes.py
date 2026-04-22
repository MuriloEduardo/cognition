from typing import Annotated, Any, Dict, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


def _merge_workdata(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge workdata across nodes instead of overwriting."""
    return {**a, **b}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    workdata: Annotated[NotRequired[Dict[str, Any]], _merge_workdata]


class Context(TypedDict):
    """Per-invocation runtime context injected via graph.ainvoke(context=...)."""

    flow: Dict[str, Any]


class EvaluationResult(BaseModel):
    """Structured output schema for the evaluation node."""

    selected_edge_id: str | None = None
    justification: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
