"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import health, invocations
from src.config.constants import SERVICE_NAME, SERVICE_DESCRIPTION, SERVICE_VERSION, API_PREFIX
from src.exceptions import UpstreamServiceError
from src.utils.logging import get_logger

logger = get_logger(__name__)


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

    @app.exception_handler(UpstreamServiceError)
    async def upstream_service_error_handler(request: Request, exc: UpstreamServiceError) -> JSONResponse:
        """Map AWS dependency failures (Kendra/STS) to HTTP 500 for the orchestrator."""
        logger.error("Upstream %s failure on %s: %s", exc.service, request.url.path, exc.message)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Upstream service error ({exc.service}): {exc.message}"},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all so unclassified bugs return a structured 500 instead of Starlette's bare default.

        The exception message isn't included in the response body since it could
        leak internals; full detail (with traceback) goes to the logs only.
        """
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app
