"""Health check routes."""

from fastapi import APIRouter, Depends

from src.agent.hybrid_ibt import HybridIBTAgent
from src.api.dependencies import get_ibt
from src.config.constants import SERVICE_NAME, STATUS_HEALTHY
from src.schemas.api import HealthResponse

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
    return {
        "status": STATUS_HEALTHY,
        "service": SERVICE_NAME.lower().replace(" - ", "-"),
        "kendra_index_id": agent.kendra_index_id,
        "aws_region": agent.aws_region,
    }
