"""Unit tests for logging utilities."""

import pytest
import logging
from unittest.mock import patch, MagicMock
from src.utils.logging import setup_logging, get_logger

class TestLogging:
    """Tests for logging utilities."""
    
    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_basic_config):
        """Test setup_logging configures logging correctly."""
        setup_logging()
        
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        
        # Check that level is set to INFO
        assert call_args[1]['level'] == logging.INFO
        
        # Check format string contains required elements
        format_str = call_args[1]['format']
        assert '%(asctime)s' in format_str
        assert '%(name)s' in format_str
        assert '%(levelname)s' in format_str
        assert '%(message)s' in format_str
        
        # Check handlers are configured
        assert 'handlers' in call_args[1]
        assert len(call_args[1]['handlers']) == 1
    
    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_get_logger_different_names(self):
        """Test get_logger returns different loggers for different names."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
        assert logger1 is not logger2
    
    def test_get_logger_same_name_returns_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("same_logger")
        logger2 = get_logger("same_logger")
        
        # Python's logging module returns the same logger instance for same name
        assert logger1 is logger2
    
    @patch('logging.getLogger')
    def test_get_logger_calls_logging_module(self, mock_get_logger):
        """Test get_logger calls logging.getLogger with correct name."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        result = get_logger("test_name")
        
        mock_get_logger.assert_called_once_with("test_name")
        assert result == mock_logger