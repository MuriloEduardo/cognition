from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.domain.entities.agent import Agent

router = APIRouter(prefix="/agents", tags=["Agents"])


class CreateAgentRequest(BaseModel):
    tenant_id: str
    name: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str | None = None
    is_active: bool = True


class UpdateAgentRequest(BaseModel):
    name: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    is_active: bool | None = None


def _repo(request: Request):
    return request.app.state.container.agent_repo


@router.post("", response_model=Agent, status_code=201)
async def create_agent(body: CreateAgentRequest, request: Request) -> Agent:
    return await _repo(request).create(
        tenant_id=body.tenant_id,
        name=body.name,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        api_key=body.api_key,
        is_active=body.is_active,
    )


@router.get("", response_model=list[Agent])
async def list_agents(tenant_id: str, request: Request) -> list[Agent]:
    return await _repo(request).list_by_tenant(tenant_id)


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(agent_id: UUID, request: Request) -> Agent:
    agent = await _repo(request).get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.patch("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: UUID, body: UpdateAgentRequest, request: Request
) -> Agent:
    fields: dict[str, Any] = {
        k: v for k, v in body.model_dump().items() if v is not None
    }
    agent = await _repo(request).update(agent_id, fields)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: UUID, request: Request) -> None:
    deleted = await _repo(request).delete(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
