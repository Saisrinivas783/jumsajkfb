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
        assert settings.bedrock_model_id == "meta.llama3-3-70b-instruct-v1:0"
        assert settings.bedrock_temperature == 0.0
        assert settings.bedrock_max_tokens == 1024
        assert settings.confidence_threshold_high == 7.0
        assert settings.confidence_threshold_low == 5.0
        assert settings.log_level == "INFO"
    
    @patch.dict('os.environ', {
        'AWS_REGION': 'us-west-2',
        'BEDROCK_MODEL_ID': 'test-model',
        'BEDROCK_TEMPERATURE': '0.5',
        'BEDROCK_MAX_TOKENS': '2048',
        'KENDRA_INDEX_ID': 'test-index-123'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        settings = IBTSettings()
        
        assert settings.aws_region == "us-west-2"
        assert settings.bedrock_model_id == "test-model"
        assert settings.bedrock_temperature == 0.5
        assert settings.bedrock_max_tokens == 2048
        assert settings.kendra_index_id == "test-index-123"
    
    def test_temperature_validation(self):
        """Test temperature field validation."""
        # Valid temperature
        settings = IBTSettings(bedrock_temperature=0.7)
        assert settings.bedrock_temperature == 0.7
        
        # Invalid temperature (should be clamped or raise error)
        with pytest.raises(ValueError):
            IBTSettings(bedrock_temperature=1.5)
    
    def test_max_tokens_validation(self):
        """Test max_tokens field validation."""
        # Valid max_tokens
        settings = IBTSettings(bedrock_max_tokens=512)
        assert settings.bedrock_max_tokens == 512
        
        # Invalid max_tokens (should be > 0)
        with pytest.raises(ValueError):
            IBTSettings(bedrock_max_tokens=0)
    
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
    
    def test_timeout_settings(self):
        """Test timeout configuration settings."""
        settings = IBTSettings(
            bedrock_read_timeout=600,
            bedrock_connect_timeout=20,
            bedrock_max_retries=5
        )
        
        assert settings.bedrock_read_timeout == 600
        assert settings.bedrock_connect_timeout == 20
        assert settings.bedrock_max_retries == 5
    
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