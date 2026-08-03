"""Unit tests for the shared, pooled HTTP client module."""

import httpx
import pytest

from src.http_client import get_http_client, close_http_client


class TestGetHttpClient:
    """Tests for get_http_client singleton behavior."""

    def teardown_method(self):
        close_http_client()

    def test_returns_httpx_client(self):
        client = get_http_client()
        assert isinstance(client, httpx.Client)

    def test_returns_same_instance_on_repeated_calls(self):
        first = get_http_client()
        second = get_http_client()
        assert first is second

    def test_limits_from_settings(self):
        from src.config.settings import get_settings
        settings = get_settings()

        client = get_http_client()
        # httpx 0.28.x does not expose limits via a public/`_limits` attribute
        # on Client; the effective httpcore.ConnectionPool holds them instead.
        pool = client._transport._pool

        assert pool._max_connections == settings.tool_http_max_connections
        assert pool._max_keepalive_connections == settings.tool_http_max_keepalive_connections


class TestCloseHttpClient:
    """Tests for shared client shutdown."""

    def test_close_allows_new_instance_after(self):
        first = get_http_client()
        close_http_client()
        second = get_http_client()
        assert first is not second
        close_http_client()

    def test_close_is_idempotent(self):
        get_http_client()
        close_http_client()
        close_http_client()  # must not raise
