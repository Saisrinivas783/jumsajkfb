"""Unit tests for custom exceptions."""

import pytest
from src.exceptions import IBTError

class TestIBTError:
    """Tests for base IBTError exception."""
    
    def test_ibt_error_basic(self):
        """Test basic IBTError creation."""
        error = IBTError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
    
    def test_ibt_error_inheritance(self):
        """Test IBTError inherits from Exception."""
        error = IBTError("Test")
        assert isinstance(error, Exception)
    
    def test_ibt_error_empty_message(self):
        """Test IBTError with empty message."""
        error = IBTError("")
        assert error.message == ""
        assert str(error) == ""
    
    def test_ibt_error_unicode_message(self):
        """Test IBTError with unicode characters."""
        error = IBTError("Error with unicode: 测试")
        assert "测试" in error.message
        assert "测试" in str(error)