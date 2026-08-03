"""Unit tests for configuration settings."""

import pytest
from unittest.mock import patch, MagicMock
from src.config.settings import IBTSettings, get_settings

class TestIBTSettings:
    """Tests for IBTSettings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = IBTSettings()

        assert settings.aws_region == "us-east-1"
        assert settings.confidence_threshold_high == 7.0
        assert settings.confidence_threshold_low == 5.0
        assert settings.log_level == "INFO"

    @patch.dict('os.environ', {
        'AWS_REGION': 'us-west-2',
        'KENDRA_INDEX_ID': 'test-index-123'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        settings = IBTSettings()

        assert settings.aws_region == "us-west-2"
        assert settings.kendra_index_id == "test-index-123"

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance due to lru_cache
        assert settings1 is settings2

    def test_confidence_thresholds(self):
        """Test confidence threshold settings."""
        settings = IBTSettings(
            confidence_threshold_high=8.5,
            confidence_threshold_low=6.0
        )

        assert settings.confidence_threshold_high == 8.5
        assert settings.confidence_threshold_low == 6.0

    def test_dxai_settings(self):
        """Test DXAIService configuration settings."""
        settings = IBTSettings(
            dxai_base_url="https://test-dxai.com",
            dxai_timeout=60,
            dxai_max_retries=2
        )

        assert settings.dxai_base_url == "https://test-dxai.com"
        assert settings.dxai_timeout == 60
        assert settings.dxai_max_retries == 2

    def test_concurrency_pool_defaults(self):
        """Test concurrency pool-size settings default values."""
        settings = IBTSettings()

        assert settings.kendra_max_pool_connections == 40
        assert settings.sts_max_pool_connections == 20

    @patch.dict('os.environ', {
        'KENDRA_MAX_POOL_CONNECTIONS': '60',
        'STS_MAX_POOL_CONNECTIONS': '30',
    })
    def test_concurrency_pool_environment_variable_override(self):
        """Test that concurrency pool-size settings can be overridden via environment variables."""
        settings = IBTSettings()

        assert settings.kendra_max_pool_connections == 60
        assert settings.sts_max_pool_connections == 30
