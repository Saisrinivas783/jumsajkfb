"""FastAPI exception handlers for validation and HTTP errors.

Runtime errors (LLM failures, tool errors, etc.) are caught by the
orchestrator and returned as graceful responses with success=false.

These handlers cover:
- RequestValidationError: Missing required fields, invalid JSON
- StarletteHTTPException: HTTP errors (404, etc.)
- Exception: Catch-all for unexpected errors
"""

from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.utils.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers with the FastAPI app."""

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI request validation errors (missing fields, wrong types, empty values)."""
        errors = exc.errors()
        error_messages = []

        for error in errors:
            field = ".".join(str(loc) for loc in error["loc"] if loc != "body")

            if error["type"] == "missing":
                # Provide helpful context for new required fields
                if field == "context":
                    error_messages.append("'context' is required with userName, userType, source, and productId fields")
                elif field == "context.userName":
                    error_messages.append("'context.userName' is required")
                elif field == "context.userType":
                    error_messages.append("'context.userType' is required")
                elif field == "context.source":
                    error_messages.append("'context.source' is required")
                elif field == "context.productId":
                    error_messages.append("'context.productId' is required")
                else:
                    error_messages.append(f"'{field}' is required")
            elif error["type"] == "value_error":
                # Handle empty value validation errors from field validators
                error_msg = error.get("msg", "")
                if "cannot be empty" in error_msg:
                    error_messages.append(f"'{field}' cannot be empty")
                else:
                    error_messages.append(f"'{field}': {error_msg}")
            else:
                # Handle other validation errors (type mismatch, etc.)
                error_messages.append(f"'{field}': {error.get('msg', 'validation failed')}")

        # Build error message
        message = ". ".join(error_messages) if error_messages else "Request validation failed"

        logger.warning(f"Request validation failed: {message}")

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": message,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "execution_time_ms": 0.0,
                "sessionId": "",
                "responseText": "",
                "metadata": [],
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions (404, etc.)."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": str(exc.detail),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "execution_time_ms": 0.0,
                "sessionId": "",
                "responseText": "",
                "metadata": [],
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle any unhandled exceptions."""
        # Log the actual error for debugging, but don't expose it in the response
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "An unexpected error occurred. Please try again later.",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "execution_time_ms": 0.0,
                "sessionId": "",
                "responseText": "",
                "metadata": [],
            },
        )
