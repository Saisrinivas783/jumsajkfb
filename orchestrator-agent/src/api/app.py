"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import get_orchestrator
from src.api.error_handlers import register_exception_handlers
from src.api.routes import health, invocations
from src.http_client import close_http_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: validate tools.yaml. If invalid, exception propagates → server aborts."""
    logger.info("Initializing Orchestrator Agent to validate configuration...")
    get_orchestrator()           # raises ToolRegistryError if tools.yaml is broken
    logger.info("Orchestrator Agent initialized successfully.")
    yield
    close_http_client()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Orchestrator Agent",
        description="Intelligent routing service using LangGraph for workflow orchestration",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # Register exception handlers (BCBSA format)
    register_exception_handlers(app)

    # Include routers
    app.include_router(health.router, prefix="/OrchestratorAgent/v2", tags=["Health"])
    app.include_router(invocations.router, prefix="/OrchestratorAgent/v2", tags=["Invocations"])

    return app
