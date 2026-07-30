"""Unit tests for configuration settings."""

from unittest.mock import patch

import pytest

from src.config.settings import IBTSettings, get_settings


class TestIBTSettings:
    """Tests for IBTSettings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = IBTSettings()

        assert settings.aws_region == "us-east-1"
        assert settings.kendra_index_id == ""
        assert settings.kendra_session_name == "ibt-agent-kendra"
        assert settings.kendra_role_duration == 3600
        assert settings.kendra_page_size == 10
        assert settings.confidence_threshold_high == 7.0
        assert settings.confidence_threshold_low == 5.0
        assert settings.log_level == "INFO"

    @patch.dict('os.environ', {
        'AWS_REGION': 'us-west-2',
        'KENDRA_INDEX_ID': 'test-index-123',
        'KENDRA_SESSION_NAME': 'test-kendra-session',
        'KENDRA_ROLE_DURATION': '1800',
        'KENDRA_PAGE_SIZE': '25'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        settings = IBTSettings()

        assert settings.aws_region == "us-west-2"
        assert settings.kendra_index_id == "test-index-123"
        assert settings.kendra_session_name == "test-kendra-session"
        assert settings.kendra_role_duration == 1800
        assert settings.kendra_page_size == 25

    def test_kendra_page_size_validation(self):
        """Test Kendra page size validation."""
        settings = IBTSettings(kendra_page_size=50)
        assert settings.kendra_page_size == 50

        with pytest.raises(ValueError):
            IBTSettings(kendra_page_size=0)

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
        get_settings.cache_clear()

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
