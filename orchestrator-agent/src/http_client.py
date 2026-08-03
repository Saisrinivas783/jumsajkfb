"""Shared, pooled HTTP client for calling tool APIs (e.g. ibt-agent)."""

import threading
from typing import Optional

import httpx

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_http_client: Optional[httpx.Client] = None
_http_client_lock = threading.Lock()


def get_http_client() -> httpx.Client:
    """Get the singleton pooled httpx.Client, creating it if needed.

    Reusing one client across requests (instead of opening a fresh
    httpx.Client per call) keeps TCP+TLS connections to tool endpoints
    alive between requests via keep-alive, avoiding a repeated handshake
    on every tool call.

    Thread-safe via double-checked locking: `_call_tool_api` runs on
    Starlette's shared thread pool, so two threads can otherwise race
    past an unguarded `is None` check and each construct a client,
    leaking one (its pooled connections never get closed since
    `close_http_client()` only closes the survivor).
    """
    global _http_client
    if _http_client is not None:
        return _http_client
    with _http_client_lock:
        if _http_client is not None:
            return _http_client
        settings = get_settings()
        logger.info(
            f"Creating shared pooled HTTP client: "
            f"max_connections={settings.tool_http_max_connections}, "
            f"max_keepalive_connections={settings.tool_http_max_keepalive_connections}"
        )
        _http_client = httpx.Client(
            timeout=settings.tool_timeout,
            limits=httpx.Limits(
                max_connections=settings.tool_http_max_connections,
                max_keepalive_connections=settings.tool_http_max_keepalive_connections,
            ),
        )
        return _http_client


def close_http_client() -> None:
    """Close and clear the singleton pooled httpx.Client."""
    global _http_client
    if _http_client is not None:
        logger.info("Closing shared pooled HTTP client")
        _http_client.close()
        _http_client = None
