"""API request/response schemas matching the Orchestrator Agent Technical Specification."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# Base Models
# =============================================================================

class BaseWorkflowRequest(BaseModel):
    """Base model for all workflow requests."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId", description="Conversation session ID")

    @field_validator("session_id")
    @classmethod
    def validate_session_id_not_empty(cls, v: str) -> str:
        """Validate that sessionId is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("cannot be empty")
        return v


class BaseWorkflowResponse(BaseModel):
    """Base model for all workflow responses."""

    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0


# =============================================================================
# Metadata Models (Standardized API Contract)
# =============================================================================

class MetadataItem(BaseModel):
    """Single key-value pair in agent metadata."""

    key: str = Field(..., description="Metadata key (e.g., 'confidence', 'selectedTool')")
    value: Any = Field(..., description="Metadata value (string, number, boolean, etc.)")


class AgentMetadata(BaseModel):
    """Metadata from a single agent."""

    agent: str = Field(..., description="Agent name (e.g., 'orchestrator', 'ibt', 'sitemap')")
    data: list[MetadataItem] = Field(default_factory=list, description="Key-value metadata pairs")


# =============================================================================
# Invocation Context (from DXAIService)
# =============================================================================

class InvocationContext(BaseModel):
    """Additional context information passed from DXAIService."""

    userName: str = Field(..., description="User identifier (REQUIRED)")
    userType: str = Field(..., description="User type: member, provider, admin, csr (REQUIRED)")
    source: str = Field(..., description="Request origin (e.g., IBTPage, HomePage) (REQUIRED)")
    productId: str = Field(..., description="Member insurance product ID (REQUIRED)")
    promptId: str | None = Field(None, description="Optional prompt template identifier")

    @field_validator("userName", "userType", "source", "productId")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that required string fields are not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("cannot be empty")
        return v


# =============================================================================
# Invocation Request/Response (POST /invocations)
# =============================================================================

class InvocationRequest(BaseWorkflowRequest):
    """
    Request body for POST /invocations endpoint.
    Matches Standardized Agent API Contract.
    """

    user_prompt: str = Field(..., alias="userPrompt", description="User's question or request")
    context: InvocationContext = Field(..., description="Required context information")

    @field_validator("user_prompt")
    @classmethod
    def validate_user_prompt_not_empty(cls, v: str) -> str:
        """Validate that userPrompt is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("cannot be empty")
        return v


class InvocationResponse(BaseWorkflowResponse):
    """
    Response body for POST /invocations endpoint.
    Matches Standardized Agent API Contract.
    """

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId", description="Echo of request sessionId")
    response_text: list[str] = Field(..., alias="responseText", description="List of NCCT IDs (or single-element fallback message) returned by the tool")
    metadata: list[AgentMetadata] = Field(
        default_factory=list,
        description="Array of agent metadata (orchestrator, tools, etc.)"
    )


# =============================================================================
# Health Check (GET /ping)
# =============================================================================

class HealthResponse(BaseModel):
    """Response for GET /ping endpoint."""

    status: str = "ok"