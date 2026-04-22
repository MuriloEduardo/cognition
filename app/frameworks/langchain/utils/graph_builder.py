from typing import Any, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph


def add_nodes_from_definition(
    builder: StateGraph,
    nodes: list[dict],
    node_handler: Callable[[dict], Callable],
) -> None:
    for node_def in nodes:
        handler = node_handler(node_def)
        builder.add_node(node_def["id"], handler)


def add_edges_from_definition(
    builder: StateGraph,
    edges: list[dict],
) -> None:
    for edge in edges:
        edge_from = START if edge["from"] == "START" else edge["from"]
        edge_to = END if edge["to"] == "END" else edge["to"]
        builder.add_edge(edge_from, edge_to)


async def build_graph_from_definition(
    definition: dict,
    state_schema: type,
    node_handler: Callable[[dict], Callable],
    *,
    context_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    builder = StateGraph(state_schema=state_schema, context_schema=context_schema)
    add_nodes_from_definition(builder, definition.get("nodes", []), node_handler)
    add_edges_from_definition(builder, definition.get("edges", []))
    return builder.compile(checkpointer=checkpointer)
