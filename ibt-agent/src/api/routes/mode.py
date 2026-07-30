"""Mode switching routes for the IBT agent."""

from fastapi import APIRouter, Depends
from src.agent.hybrid_ibt import HybridIBTAgent
from src.api.dependencies import get_ibt
from src.schemas.api import ModeRequest

router = APIRouter()

@router.post("/mode")
def set_mode(
    request: ModeRequest,
    agent: HybridIBTAgent = Depends(get_ibt),
):
    """Switch between LLM-enhanced and direct Kendra modes."""
    agent.set_mode(request.use_llm)
    return agent.get_mode_info()

@router.get("/mode")
def get_mode(
    agent: HybridIBTAgent = Depends(get_ibt),
):
    """Get current agent mode information."""
    return agent.get_mode_info()