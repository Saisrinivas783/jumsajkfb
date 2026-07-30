"""Unit tests for benefits tools."""

import pytest
from unittest.mock import patch, MagicMock
from src.tools.benefits import search_benefits

class TestSearchBenefitsTool:
    """Tests for search_benefits tool."""
    
    @patch('src.tools.benefits.get_kendra_service')
    def test_search_benefits_success(self, mock_get_service):
        mock_service = MagicMock()
        mock_service.search.return_value = {
            'success': True,
            'results': [
                {
                    'ncct_id': 'NCCT123',
                    'service_name': 'Dental Coverage',
                    'excerpt': 'Comprehensive dental benefits'
                }
            ]
        }
        mock_get_service.return_value = mock_service
        
        result = search_benefits.func("dental benefits")
        
        assert 'NCCT123' in result
        assert 'Dental Coverage' in result
        mock_service.search.assert_called_once_with("dental benefits")
    
    @patch('src.tools.benefits.get_kendra_service')
    def test_search_benefits_no_results(self, mock_get_service):
        mock_service = MagicMock()
        mock_service.search.return_value = {
            'success': True,
            'results': []
        }
        mock_get_service.return_value = mock_service
        
        result = search_benefits.func("unknown query")
        
        # Returns the FEPOC fallback message as a string
        assert isinstance(result, str)
        assert "Your search did not return any results" in result
        assert "fepblue.org/plan-brochures" in result
    
    @patch('src.tools.benefits.get_kendra_service')
    def test_search_benefits_service_error(self, mock_get_service):
        mock_service = MagicMock()
        mock_service.search.return_value = {
            'success': False,
            'error': 'Service unavailable'
        }
        mock_get_service.return_value = mock_service
        
        result = search_benefits.func("test query")
        
        assert "technical difficulties" in result.lower()
    
    @patch('src.tools.benefits.get_kendra_service')
    def test_search_benefits_exception(self, mock_get_service):
        mock_get_service.side_effect = Exception("Connection error")
        
        result = search_benefits.func("test query")
        
        assert "technical difficulties" in result.lower()
