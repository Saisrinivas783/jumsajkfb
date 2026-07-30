"""FastAPI dependency injection."""

from functools import lru_cache
from src.config.settings import get_settings
from src.agent.hybrid_ibt import HybridIBTAgent

@lru_cache
def get_ibt() -> HybridIBTAgent:
    """Get singleton HybridIBTAgent instance."""
    return HybridIBTAgent()
