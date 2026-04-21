from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from app.domain.entities.agent import Agent, WorkflowEdge, WorkflowNode


class AgentRepository(ABC):
    @abstractmethod
    async def create(
        self,
        *,
        tenant_id: str,
        name: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str | None = None,
        is_active: bool = True,
    ) -> Agent: ...

    @abstractmethod
    async def get(self, agent_id: UUID) -> Agent | None: ...

    @abstractmethod
    async def list_by_tenant(self, tenant_id: str) -> list[Agent]: ...

    @abstractmethod
    async def update(self, agent_id: UUID, fields: dict[str, Any]) -> Agent | None: ...

    @abstractmethod
    async def delete(self, agent_id: UUID) -> bool: ...


class WorkflowNodeRepository(ABC):
    @abstractmethod
    async def create(
        self,
        *,
        agent_id: UUID,
        name: str,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        node_order: int = 0,
    ) -> WorkflowNode: ...

    @abstractmethod
    async def get(self, node_id: UUID) -> WorkflowNode | None: ...

    @abstractmethod
    async def list_by_agent(self, agent_id: UUID) -> list[WorkflowNode]: ...

    @abstractmethod
    async def update(
        self, node_id: UUID, fields: dict[str, Any]
    ) -> WorkflowNode | None: ...

    @abstractmethod
    async def delete(self, node_id: UUID) -> bool: ...


class WorkflowEdgeRepository(ABC):
    @abstractmethod
    async def create(
        self,
        *,
        agent_id: UUID,
        from_node_id: UUID | None = None,
        to_node_id: UUID | None = None,
        condition_prompt: str | None = None,
        priority: int = 0,
    ) -> WorkflowEdge: ...

    @abstractmethod
    async def get(self, edge_id: UUID) -> WorkflowEdge | None: ...

    @abstractmethod
    async def list_by_agent(self, agent_id: UUID) -> list[WorkflowEdge]: ...

    @abstractmethod
    async def update(
        self, edge_id: UUID, fields: dict[str, Any]
    ) -> WorkflowEdge | None: ...

    @abstractmethod
    async def delete(self, edge_id: UUID) -> bool: ...
