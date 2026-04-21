import json
from typing import Any
from uuid import UUID

from app.domain.entities.agent import Agent, WorkflowEdge, WorkflowNode
from app.infrastructure.database import PostgresConnection
from app.ports.outbound.agent_repository import (
    AgentRepository,
    WorkflowEdgeRepository,
    WorkflowNodeRepository,
)


def _row_to_agent(row: dict) -> Agent:
    return Agent(
        id=row["id"],
        tenant_id=row["tenant_id"],
        name=row["name"],
        model=row["model"],
        temperature=row["temperature"],
        max_tokens=row["max_tokens"],
        api_key=row["api_key"],
        is_active=row["is_active"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_node(row: dict) -> WorkflowNode:
    rf = row["response_format"]
    if isinstance(rf, str):
        try:
            rf = json.loads(rf)
        except (json.JSONDecodeError, TypeError):
            rf = None
    return WorkflowNode(
        id=row["id"],
        agent_id=row["agent_id"],
        name=row["name"],
        system_prompt=row["system_prompt"],
        response_format=rf,
        node_order=row["node_order"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: dict) -> WorkflowEdge:
    return WorkflowEdge(
        id=row["id"],
        agent_id=row["agent_id"],
        from_node_id=row["from_node_id"],
        to_node_id=row["to_node_id"],
        condition_prompt=row["condition_prompt"],
        priority=row["priority"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PostgresAgentRepository(AgentRepository):
    def __init__(self, database: PostgresConnection) -> None:
        self._db = database

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
    ) -> Agent:
        pool = await self._db.get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO agents (tenant_id, name, model, temperature, max_tokens, api_key, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            tenant_id,
            name,
            model,
            temperature,
            max_tokens,
            api_key,
            is_active,
        )
        return _row_to_agent(dict(row))

    async def get(self, agent_id: UUID) -> Agent | None:
        pool = await self._db.get_pool()
        row = await pool.fetchrow("SELECT * FROM agents WHERE id = $1", agent_id)
        return _row_to_agent(dict(row)) if row else None

    async def list_by_tenant(self, tenant_id: str) -> list[Agent]:
        pool = await self._db.get_pool()
        rows = await pool.fetch(
            "SELECT * FROM agents WHERE tenant_id = $1 ORDER BY created_at",
            tenant_id,
        )
        return [_row_to_agent(dict(r)) for r in rows]

    async def update(self, agent_id: UUID, fields: dict[str, Any]) -> Agent | None:
        allowed = {"name", "model", "temperature", "max_tokens", "api_key", "is_active"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get(agent_id)
        clauses = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(updates))
        values = list(updates.values())
        pool = await self._db.get_pool()
        row = await pool.fetchrow(
            f"UPDATE agents SET {clauses}, updated_at = now() WHERE id = $1 RETURNING *",
            agent_id,
            *values,
        )
        return _row_to_agent(dict(row)) if row else None

    async def delete(self, agent_id: UUID) -> bool:
        pool = await self._db.get_pool()
        result = await pool.execute("DELETE FROM agents WHERE id = $1", agent_id)
        return result.split()[-1] != "0"


class PostgresWorkflowNodeRepository(WorkflowNodeRepository):
    def __init__(self, database: PostgresConnection) -> None:
        self._db = database

    async def create(
        self,
        *,
        agent_id: UUID,
        name: str,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        node_order: int = 0,
    ) -> WorkflowNode:
        pool = await self._db.get_pool()
        rf_json = json.dumps(response_format) if response_format else None
        row = await pool.fetchrow(
            """
            INSERT INTO agent_nodes (agent_id, name, system_prompt, response_format, node_order)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            RETURNING *
            """,
            agent_id,
            name,
            system_prompt,
            rf_json,
            node_order,
        )
        return _row_to_node(dict(row))

    async def get(self, node_id: UUID) -> WorkflowNode | None:
        pool = await self._db.get_pool()
        row = await pool.fetchrow("SELECT * FROM agent_nodes WHERE id = $1", node_id)
        return _row_to_node(dict(row)) if row else None

    async def list_by_agent(self, agent_id: UUID) -> list[WorkflowNode]:
        pool = await self._db.get_pool()
        rows = await pool.fetch(
            "SELECT * FROM agent_nodes WHERE agent_id = $1 ORDER BY node_order, created_at",
            agent_id,
        )
        return [_row_to_node(dict(r)) for r in rows]

    async def update(
        self, node_id: UUID, fields: dict[str, Any]
    ) -> WorkflowNode | None:
        allowed = {"name", "system_prompt", "response_format", "node_order"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get(node_id)

        set_parts = []
        values = []
        for i, (col, val) in enumerate(updates.items()):
            if col == "response_format":
                set_parts.append(f"{col} = ${i + 2}::jsonb")
                values.append(json.dumps(val) if val is not None else None)
            else:
                set_parts.append(f"{col} = ${i + 2}")
                values.append(val)

        pool = await self._db.get_pool()
        row = await pool.fetchrow(
            f"UPDATE agent_nodes SET {', '.join(set_parts)}, updated_at = now() WHERE id = $1 RETURNING *",
            node_id,
            *values,
        )
        return _row_to_node(dict(row)) if row else None

    async def delete(self, node_id: UUID) -> bool:
        pool = await self._db.get_pool()
        result = await pool.execute("DELETE FROM agent_nodes WHERE id = $1", node_id)
        return result.split()[-1] != "0"


class PostgresWorkflowEdgeRepository(WorkflowEdgeRepository):
    def __init__(self, database: PostgresConnection) -> None:
        self._db = database

    async def create(
        self,
        *,
        agent_id: UUID,
        from_node_id: UUID | None = None,
        to_node_id: UUID | None = None,
        condition_prompt: str | None = None,
        priority: int = 0,
    ) -> WorkflowEdge:
        pool = await self._db.get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO agent_edges (agent_id, from_node_id, to_node_id, condition_prompt, priority)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            agent_id,
            from_node_id,
            to_node_id,
            condition_prompt,
            priority,
        )
        return _row_to_edge(dict(row))

    async def get(self, edge_id: UUID) -> WorkflowEdge | None:
        pool = await self._db.get_pool()
        row = await pool.fetchrow("SELECT * FROM agent_edges WHERE id = $1", edge_id)
        return _row_to_edge(dict(row)) if row else None

    async def list_by_agent(self, agent_id: UUID) -> list[WorkflowEdge]:
        pool = await self._db.get_pool()
        rows = await pool.fetch(
            "SELECT * FROM agent_edges WHERE agent_id = $1 ORDER BY priority, created_at",
            agent_id,
        )
        return [_row_to_edge(dict(r)) for r in rows]

    async def update(
        self, edge_id: UUID, fields: dict[str, Any]
    ) -> WorkflowEdge | None:
        allowed = {"from_node_id", "to_node_id", "condition_prompt", "priority"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get(edge_id)
        clauses = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(updates))
        values = list(updates.values())
        pool = await self._db.get_pool()
        row = await pool.fetchrow(
            f"UPDATE agent_edges SET {clauses}, updated_at = now() WHERE id = $1 RETURNING *",
            edge_id,
            *values,
        )
        return _row_to_edge(dict(row)) if row else None

    async def delete(self, edge_id: UUID) -> bool:
        pool = await self._db.get_pool()
        result = await pool.execute("DELETE FROM agent_edges WHERE id = $1", edge_id)
        return result.split()[-1] != "0"
