from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.domain.entities.agent import WorkflowNode

router = APIRouter(prefix="/agents/{agent_id}/nodes", tags=["Workflow Nodes"])


class CreateNodeRequest(BaseModel):
    name: str
    system_prompt: str | None = None
    response_format: dict[str, Any] | None = None
    node_order: int = 0


class UpdateNodeRequest(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    response_format: dict[str, Any] | None = None
    node_order: int | None = None


def _repo(request: Request):
    return request.app.state.container.agent_node_repo


@router.post("", response_model=WorkflowNode, status_code=201)
async def create_node(
    agent_id: UUID, body: CreateNodeRequest, request: Request
) -> WorkflowNode:
    return await _repo(request).create(
        agent_id=agent_id,
        name=body.name,
        system_prompt=body.system_prompt,
        response_format=body.response_format,
        node_order=body.node_order,
    )


@router.get("", response_model=list[WorkflowNode])
async def list_nodes(agent_id: UUID, request: Request) -> list[WorkflowNode]:
    return await _repo(request).list_by_agent(agent_id)


@router.get("/{node_id}", response_model=WorkflowNode)
async def get_node(agent_id: UUID, node_id: UUID, request: Request) -> WorkflowNode:
    node = await _repo(request).get(node_id)
    if not node or node.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.patch("/{node_id}", response_model=WorkflowNode)
async def update_node(
    agent_id: UUID, node_id: UUID, body: UpdateNodeRequest, request: Request
) -> WorkflowNode:
    fields: dict[str, Any] = {
        k: v for k, v in body.model_dump().items() if v is not None
    }
    node = await _repo(request).update(node_id, fields)
    if not node or node.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.delete("/{node_id}", status_code=204)
async def delete_node(agent_id: UUID, node_id: UUID, request: Request) -> None:
    node = await _repo(request).get(node_id)
    if not node or node.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Node not found")
    await _repo(request).delete(node_id)
