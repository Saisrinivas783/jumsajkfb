"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health, invocations
from src.config.constants import SERVICE_NAME, SERVICE_DESCRIPTION, SERVICE_VERSION, API_PREFIX
from src.config.settings import get_settings
from src.services.kendra_service import get_kendra_service
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup starts per-process credential refresh workers when role ARNs are configured."""
    settings = get_settings()
    kendra_service = None
    if settings.kendra_role_arn:
        kendra_service = get_kendra_service()
        try:
            await asyncio.to_thread(kendra_service.warm_credentials)
        except Exception as e:
            logger.warning("Initial Kendra credential warmup failed; startup will continue: %s", e)
        kendra_service.start_credential_refresh()

    try:
        yield
    finally:
        if kendra_service is not None:
            kendra_service.stop_credential_refresh()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        version=SERVICE_VERSION,
        lifespan=_lifespan,
    )

    app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
    app.include_router(invocations.router, prefix=API_PREFIX, tags=["Invocations"])

    return app
