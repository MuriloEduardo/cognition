"""
Cognition service entities.
Uses shared schemas for cross-service communication.
"""

import sys
from pathlib import Path

# Add parent directory to path to import shared_schemas
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_schemas import (
    WorkflowContext,
    CognitionRequest as GenerateRequest,
    CognitionResponse as GenerateResponse,
)

__all__ = [
    "WorkflowContext",
    "GenerateRequest",
    "GenerateResponse",
]

