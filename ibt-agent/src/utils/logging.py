"""Logging configuration for IBT agent."""

import logging
import sys

def setup_logging():
    """Setup basic logging for IBT agent."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)