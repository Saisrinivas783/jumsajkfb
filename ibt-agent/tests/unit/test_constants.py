"""Unit tests for constants module."""

import pytest
from src.config.constants import (
    API_PREFIX, SERVICE_NAME, SERVICE_VERSION,
    DEFAULT_PAGE_SIZE,
    NCCT_ID_KEYS, SERVICE_NAME_KEYS,
    NO_EXCERPT_MESSAGE, PROCESSING_ERROR_MESSAGE
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
        assert isinstance(NCCT_ID_KEYS, list)
        assert isinstance(SERVICE_NAME_KEYS, list)
    
    def test_attribute_key_constants(self):
        """Test document attribute key constants."""
        assert isinstance(NCCT_ID_KEYS, list)
        assert isinstance(SERVICE_NAME_KEYS, list)
        assert "NCCT_ID" in NCCT_ID_KEYS
        assert "Service_Name" in SERVICE_NAME_KEYS
    
    def test_message_constants(self):
        """Test message constants."""
        assert isinstance(NO_EXCERPT_MESSAGE, str)
        assert isinstance(PROCESSING_ERROR_MESSAGE, str)
        assert len(NO_EXCERPT_MESSAGE) > 0
        assert len(PROCESSING_ERROR_MESSAGE) > 0
    

    
    def test_ncct_id_keys_coverage(self):
        """Test NCCT ID keys cover common variations."""
        expected_keys = ['NCCT_ID', 'ID', 'Document_ID', 'DocumentId']
        for key in expected_keys:
            assert key in NCCT_ID_KEYS
    
    def test_service_name_keys_coverage(self):
        """Test service name keys cover common variations."""
        expected_keys = ['Service_Name', 'Title', 'Name']
        for key in expected_keys:
            assert key in SERVICE_NAME_KEYS