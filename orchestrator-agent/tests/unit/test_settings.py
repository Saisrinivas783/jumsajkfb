"""Unit tests for orchestrator configuration settings."""

from unittest.mock import patch

from src.config.settings import OrchestratorSettings, get_settings


class TestOrchestratorSettings:
    """Tests for OrchestratorSettings concurrency pool-size fields."""

    def test_concurrency_pool_defaults(self):
        """Test concurrency pool-size settings default values."""
        settings = OrchestratorSettings()

        assert settings.bedrock_max_pool_connections == 40
        assert settings.sts_max_pool_connections == 20
        assert settings.tool_http_max_connections == 40
        assert settings.tool_http_max_keepalive_connections == 20

    @patch.dict('os.environ', {
        'BEDROCK_MAX_POOL_CONNECTIONS': '60',
        'STS_MAX_POOL_CONNECTIONS': '30',
        'TOOL_HTTP_MAX_CONNECTIONS': '60',
        'TOOL_HTTP_MAX_KEEPALIVE_CONNECTIONS': '30',
    })
    def test_concurrency_pool_environment_variable_override(self):
        """Test that concurrency pool-size settings can be overridden via environment variables."""
        settings = OrchestratorSettings()

        assert settings.bedrock_max_pool_connections == 60
        assert settings.sts_max_pool_connections == 30
        assert settings.tool_http_max_connections == 60
        assert settings.tool_http_max_keepalive_connections == 30

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
