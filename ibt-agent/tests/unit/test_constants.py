"""Unit tests for constants module."""

import pytest
from src.config.constants import (
    API_PREFIX, SERVICE_NAME, SERVICE_VERSION,
    DEFAULT_PAGE_SIZE,
)

class TestConstants:
    """Tests for application constants."""

    def test_api_constants(self):
        """Test API-related constants."""
        assert API_PREFIX == "/IbtAgent/v2"
        assert SERVICE_NAME == "IBT Agent - Hybrid"
        assert SERVICE_VERSION == "2.0.0"

    def test_kendra_constants(self):
        """Test Kendra-related constants."""
        assert DEFAULT_PAGE_SIZE == 10
