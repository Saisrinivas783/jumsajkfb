"""Orchestrator workflow state - pure data container."""

from typing import Any

from pydantic import BaseModel, Field

from src.schemas.api import InvocationContext
from src.schemas.tools import SelectedTool, ToolResult, ErrorInfo


class OrchestratorState(BaseModel):
    """
    LangGraph state for the orchestrator workflow.

    This is a pure data container - no business logic.
    State flows through nodes which transform it.
    """

    # Input (from request)
    query: str
    session_id: str
    context: InvocationContext  # Required in standardized API contract

    # Intent analysis result (from analyzer node)
    selected_tool: SelectedTool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Tool execution result (from executor node)
    tool_result: ToolResult | None = None

    # Tool metadata (for multi-agent aggregation - future use)
    tool_metadata: list[Any] = Field(
        default_factory=list,
        description="Metadata from tool responses (AgentMetadata objects)"
    )

    # Authorization header (forwarded to downstream agents)
    authorization: str | None = None

    # Final output
    final_answer: list[str] | None = None

    # Guardrail tracking
    guardrail_blocked: bool = False
    guardrail_action: str = "NONE"  # "NONE" | "BLOCKED" | "PII_MASKED"

    # Error tracking
    error: ErrorInfo | None = None  # Serializable; read by fallback node
