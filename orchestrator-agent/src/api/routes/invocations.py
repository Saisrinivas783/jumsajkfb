"""Invocation routes for the orchestrator agent."""

from fastapi import APIRouter, Depends, Header
from typing import Optional

from src.schemas.api import InvocationRequest, InvocationResponse
from src.graph.orchestrator import OrchestratorAgent
from src.api.dependencies import get_orchestrator

router = APIRouter()


@router.post("/invocations", response_model=InvocationResponse)
def invocations(
    payload: InvocationRequest,
    agent: OrchestratorAgent = Depends(get_orchestrator),
    authorization: Optional[str] = Header(None),
):
    """Process user query and route to appropriate tool."""
    return agent.handle_invocation(payload, authorization=authorization)
