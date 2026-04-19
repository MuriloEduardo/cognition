"""
Cognition-specific entities.
Focuses on AI/LLM processing.
"""

from pydantic import BaseModel, Field
from typing import Any


class ModelConfig(BaseModel):
    """Configuration for LLM model."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class PromptTemplate(BaseModel):
    """Template for constructing prompts."""

    template_id: str
    template: str
    variables: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CognitionMetrics(BaseModel):
    """Metrics from cognition processing."""

    request_id: str
    model: str
    tokens_used: int
    processing_time_ms: int
    cached: bool = False
