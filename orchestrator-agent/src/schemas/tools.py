"""Tool-related schemas for the orchestrator workflow."""

from typing import Any

from pydantic import BaseModel, Field


class SelectedTool(BaseModel):
    """Result of LLM tool selection."""

    tool_name: str = Field(..., description="Name of the selected tool or 'NO_TOOL'")
    confidence: float = Field(..., ge=0.0, le=10.0, description="Confidence score (0-10)")
    reasoning: str = Field("", description="Explanation for selection")
    reformulated_query: str | None = None


class ToolResult(BaseModel):
    """Result from tool execution."""

    tool_name: str
    success: bool = True
    response: Any = None
    error: str | None = None


class ErrorInfo(BaseModel):
    """Serializable error descriptor stored in state.error."""

    error_type: str  # "tool_timeout" | "tool_unavailable" | "unknown"
    message: str     # Technical message (for logs/debugging)
    tool_name: str | None = None
