"""Invocation routes for the IBT agent."""

from fastapi import APIRouter, Depends
from src.schemas.api import InvocationRequest, InvocationResponse
from src.agent.hybrid_ibt import HybridIBTAgent
from src.api.dependencies import get_ibt

router = APIRouter()

@router.post("/invocations", response_model=InvocationResponse)
def invocations(
    payload: InvocationRequest,
    agent: HybridIBTAgent = Depends(get_ibt),
):
    """Process benefit and coverage inquiries per FEPOC specification."""
    context_dict = payload.context.model_dump()

    result = agent.process_query(
        user_prompt=payload.user_prompt,
        session_id=payload.session_id,
        context=context_dict
    )
    
    return InvocationResponse(
        sessionId=result["sessionId"],
        responseText=result["responseText"],
        confidence=result["confidence"],
        success=result["success"],
        execution_time_ms=result["execution_time_ms"]
    )
