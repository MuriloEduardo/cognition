from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.domain.entities.agent import WorkflowEdge

router = APIRouter(prefix="/agents/{agent_id}/edges", tags=["Workflow Edges"])


class CreateEdgeRequest(BaseModel):
    from_node_id: UUID | None = None  # None = START
    to_node_id: UUID | None = None  # None = END
    condition_prompt: str | None = None
    priority: int = 0


class UpdateEdgeRequest(BaseModel):
    from_node_id: UUID | None = None
    to_node_id: UUID | None = None
    condition_prompt: str | None = None
    priority: int | None = None


def _repo(request: Request):
    return request.app.state.container.agent_edge_repo


@router.post("", response_model=WorkflowEdge, status_code=201)
async def create_edge(
    agent_id: UUID, body: CreateEdgeRequest, request: Request
) -> WorkflowEdge:
    return await _repo(request).create(
        agent_id=agent_id,
        from_node_id=body.from_node_id,
        to_node_id=body.to_node_id,
        condition_prompt=body.condition_prompt,
        priority=body.priority,
    )


@router.get("", response_model=list[WorkflowEdge])
async def list_edges(agent_id: UUID, request: Request) -> list[WorkflowEdge]:
    return await _repo(request).list_by_agent(agent_id)


@router.get("/{edge_id}", response_model=WorkflowEdge)
async def get_edge(agent_id: UUID, edge_id: UUID, request: Request) -> WorkflowEdge:
    edge = await _repo(request).get(edge_id)
    if not edge or edge.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge


@router.patch("/{edge_id}", response_model=WorkflowEdge)
async def update_edge(
    agent_id: UUID, edge_id: UUID, body: UpdateEdgeRequest, request: Request
) -> WorkflowEdge:
    fields: dict[str, Any] = {
        k: v for k, v in body.model_dump().items() if v is not None
    }
    edge = await _repo(request).update(edge_id, fields)
    if not edge or edge.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge


@router.delete("/{edge_id}", status_code=204)
async def delete_edge(agent_id: UUID, edge_id: UUID, request: Request) -> None:
    edge = await _repo(request).get(edge_id)
    if not edge or edge.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Edge not found")
    await _repo(request).delete(edge_id)
