"""Health check routes."""

from fastapi import APIRouter, Depends
from src.schemas.api import HealthResponse
from src.agent.hybrid_ibt import HybridIBTAgent
from src.api.dependencies import get_ibt
from src.config.constants import STATUS_HEALTHY, SERVICE_NAME

router = APIRouter()


@router.get("/ping", response_model=HealthResponse)
def ping():
    """Health check endpoint."""
    return HealthResponse()


@router.get("/health")
def health(
    agent: HybridIBTAgent = Depends(get_ibt),
):
    """Detailed health check endpoint."""
    mode_info = agent.get_mode_info()
    return {
        "status": STATUS_HEALTHY,
        "service": SERVICE_NAME.lower().replace(" - ", "-"),
        "current_mode": mode_info["current_mode"],
        "kendra_index_id": mode_info["kendra_index_id"],
        "aws_region": mode_info["aws_region"]
    }
