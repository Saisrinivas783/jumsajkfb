"""FastAPI dependency injection."""

from functools import lru_cache

from src.config.settings import get_settings
from src.graph.orchestrator import OrchestratorAgent


@lru_cache
def get_orchestrator() -> OrchestratorAgent:
    """
    Get singleton OrchestratorAgent instance.

    Uses lru_cache to ensure only one instance is created.
    """
    settings = get_settings()
    return OrchestratorAgent(
        registry_path=settings.registry_path
    )
