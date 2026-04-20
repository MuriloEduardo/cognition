"""
Cognition service entities.
Cognition processes AI/LLM requests and returns processing results.
"""

from typing import Any

from pydantic import BaseModel, Field


class WorkflowContext(BaseModel):
    model_config = {"extra": "allow"}

    session_id: str
    conversation_id: str | None = None
    user_id: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CognitionRequest(BaseModel):
    request_id: str
    prompt: str
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    context: WorkflowContext | None = None


class LLMResult(BaseModel):
    content: str
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_token_details: dict[str, Any] | None = None
    output_token_details: dict[str, Any] | None = None


class CognitionResponse(BaseModel):
    request_id: str
    content: str
    messages: list[str] = []
    model: str
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    context: WorkflowContext | None = None


__all__ = [
    "WorkflowContext",
    "CognitionRequest",
    "CognitionResponse",
    "LLMResult",
]
