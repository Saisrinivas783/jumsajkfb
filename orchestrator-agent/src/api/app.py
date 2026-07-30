"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import get_orchestrator
from src.api.error_handlers import register_exception_handlers
from src.api.routes import health, invocations
from src.llm.client import get_chat_models
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup validates configuration and starts per-process credential refresh."""
    logger.info("Initializing Orchestrator Agent to validate configuration...")
    get_orchestrator()  # raises ToolRegistryError if tools.yaml is broken
    chat_models = get_chat_models()
    if chat_models.settings.bedrock_role_arn:
        try:
            await asyncio.to_thread(chat_models.refresh_credentials)
        except Exception as e:
            logger.warning("Initial Bedrock credential warmup failed; startup will continue: %s", e)
    chat_models.start_credential_refresh()
    logger.info("Orchestrator Agent initialized successfully.")

    try:
        yield
    finally:
        chat_models.stop_credential_refresh()


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
