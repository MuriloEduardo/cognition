"""
Cognition service entities.
Cognition processes AI/LLM requests and returns processing results.
It doesn't know about channels or workflows - just processes cognition tasks.
"""

import sys
from pathlib import Path

# Add parent directory to path to import shared_schemas
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from shared_schemas import (
    WorkflowContext,
    CognitionRequest,
    CognitionResponse,
)

__all__ = [
    "WorkflowContext",
    "CognitionRequest",
    "CognitionResponse",
]

