"""Logging configuration for IBT agent."""

import logging
import sys

def setup_logging():
    """Setup basic logging for IBT agent, honoring the LOG_LEVEL setting."""
    from src.config.settings import get_settings

    log_level = get_settings().log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)