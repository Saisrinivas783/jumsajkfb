"""Unit tests for constants module."""

from src.config.constants import (
    API_PREFIX,
    DEFAULT_CONFIDENCE,
    DEFAULT_EXECUTION_TIME,
    DEFAULT_MESSAGE,
    DEFAULT_SUCCESS,
    HEALTH_ENDPOINT,
    INVOCATIONS_ENDPOINT,
    PING_ENDPOINT,
    SERVICE_DESCRIPTION,
    SERVICE_NAME,
    SERVICE_VERSION,
    STATUS_HEALTHY,
    STATUS_OK,
)


class TestConstants:
    """Tests for application constants."""

    def test_api_constants(self):
        """Test API-related constants."""
        assert API_PREFIX == "/IbtAgent/v2"
        assert INVOCATIONS_ENDPOINT == "/invocations"
        assert PING_ENDPOINT == "/ping"
        assert HEALTH_ENDPOINT == "/health"

    def test_service_constants(self):
        """Test service-related constants."""
        assert SERVICE_NAME == "IBT Agent - Hybrid"
        assert SERVICE_VERSION == "2.0.0"
        assert "direct Kendra search" in SERVICE_DESCRIPTION

    def test_status_constants(self):
        """Test HTTP status constants."""
        assert STATUS_OK == "ok"
        assert STATUS_HEALTHY == "healthy"

    def test_default_response_constants(self):
        """Test default response constants."""
        assert DEFAULT_CONFIDENCE == 0.0
        assert DEFAULT_SUCCESS is True
        assert DEFAULT_MESSAGE == ""
        assert DEFAULT_EXECUTION_TIME == 0.0
