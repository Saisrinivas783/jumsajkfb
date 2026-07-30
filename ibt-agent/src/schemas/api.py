"""API request/response schemas matching the IBT Agent Technical Specification."""

from datetime import datetime
from typing import Union, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Context Model
# =============================================================================

class InvocationContext(BaseModel):
    """Context information passed from orchestrator agent."""
    
    userName: str = Field(..., description="User identifier")
    userType: str = Field(..., description="User type: member, provider, admin, csr")
    source: str = Field(default="IBTPage", description="Request origin (e.g., IBTPage, HomePage)")
    productId: str = Field(..., description="Member insurance product ID")
    promptId: Optional[str] = Field(None, description="Optional prompt template identifier")


# =============================================================================
# Invocation Request/Response (POST /invocations)
# =============================================================================

class InvocationRequest(BaseModel):
    """
    Request body for POST /invocations endpoint.
    Matches DXAIService InvokeAgent Request format.
    """
    
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId", description="Conversation session ID")
    user_prompt: str = Field(..., alias="userPrompt", description="User's question or request")
    context: Optional[InvocationContext] = Field(None, description="Context information from orchestrator")


class InvocationResponse(BaseModel):
    """
    Response body for POST /invocations endpoint.
    """
    
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId")
    confidence: float = Field(0.0, ge=0.0, le=10.0)
    response_text: Union[str, List[str]] = Field("", alias="responseText", description="Response text as string (LLM mode) or array of NCCT IDs (Direct Kendra mode)")
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0


# =============================================================================
# Mode Request (POST /mode)
# =============================================================================

class ModeRequest(BaseModel):
    """Request body for mode switching endpoint."""
    
    use_llm: bool = Field(..., description="Whether to use LLM-enhanced mode")


# =============================================================================
# Health Check (GET /ping)
# =============================================================================

class HealthResponse(BaseModel):
    """Response for GET /ping endpoint."""

    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)