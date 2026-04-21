import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class Agent(BaseModel):
    id: UUID
    tenant_id: str
    name: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str | None = None
    is_active: bool = True
    created_at: datetime.datetime
    updated_at: datetime.datetime


class WorkflowNode(BaseModel):
    id: UUID
    agent_id: UUID
    name: str
    system_prompt: str | None = None
    response_format: dict[str, Any] | None = None
    node_order: int = 0
    created_at: datetime.datetime
    updated_at: datetime.datetime


class WorkflowEdge(BaseModel):
    id: UUID
    agent_id: UUID
    from_node_id: UUID | None = None  # None = START sentinel
    to_node_id: UUID | None = None  # None = END sentinel
    condition_prompt: str | None = None
    priority: int = 0
    created_at: datetime.datetime
    updated_at: datetime.datetime
