"""API request/response schemas matching the IBT Agent Technical Specification."""

from datetime import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InvocationContext(BaseModel):
    """Context information passed from orchestrator agent."""

    userName: str = Field(..., description="User identifier")
    userType: str = Field(..., description="User type: member, provider, admin, csr")
    source: str = Field(..., description="Request origin (e.g., IBTPage, HomePage)")
    productId: str = Field(..., description="Member insurance product ID")
    promptId: Optional[str] = Field(None, description="Optional prompt template identifier")

    @field_validator("userName", "userType", "source", "productId")
    @classmethod
    def validate_not_empty(cls, value: str) -> str:
        """Validate that required context strings are not empty."""
        if not value or not value.strip():
            raise ValueError("cannot be empty")
        return value


class InvocationRequest(BaseModel):
    """
    Request body for POST /invocations endpoint.
    Matches DXAIService InvokeAgent Request format.
    """

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId", description="Conversation session ID")
    user_prompt: str = Field(..., alias="userPrompt", description="User's question or request")
    context: InvocationContext = Field(..., description="Context information from orchestrator")


class InvocationResponse(BaseModel):
    """Response body for POST /invocations endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId")
    confidence: float = Field(0.0, ge=0.0, le=10.0)
    response_text: Union[str, List[str]] = Field(
        "",
        alias="responseText",
        description="Response text or array of NCCT IDs from direct Kendra search",
    )
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response for GET /ping endpoint."""

    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
