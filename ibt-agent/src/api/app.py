"""FastAPI application factory."""

from fastapi import FastAPI

from src.api.routes import health, invocations
from src.config.constants import SERVICE_NAME, SERVICE_DESCRIPTION, SERVICE_VERSION, API_PREFIX


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        version=SERVICE_VERSION,
    )

    # Include routers
    app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
    app.include_router(invocations.router, prefix=API_PREFIX, tags=["Invocations"])

    return app
