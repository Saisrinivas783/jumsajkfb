"""
Centralized logging configuration for the Orchestrator Agent.

Usage:
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Info message")
    logger.debug("Debug details - only visible when LOG_LEVEL=DEBUG")

Environment Variables:
    LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
"""

import logging
import sys
from typing import Optional

# Track if logging has been configured
_logging_configured = False


def configure_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the application. Should be called once at startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). If None, reads from settings.
        format_string: Custom format string. If None, uses default format.
    """
    global _logging_configured

    if _logging_configured:
        return

    # Import settings here to avoid circular imports
    from src.config.settings import orchestrator_settings

    log_level = level or orchestrator_settings.log_level
    log_format = format_string or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _logging_configured = True

    root_logger = logging.getLogger()
    root_logger.debug(f"Logging configured: level={log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Ensures logging is configured before returning logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Ensure logging is configured
    if not _logging_configured:
        configure_logging()

    return logging.getLogger(name)
