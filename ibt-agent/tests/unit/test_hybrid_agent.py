"""Unit tests for HybridIBTAgent."""

import pytest
from unittest.mock import MagicMock, patch
from src.agent.hybrid_ibt import HybridIBTAgent

class TestHybridIBTAgent:
    """Tests for HybridIBTAgent class."""

    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_init(self, mock_settings, mock_get_kendra_service):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        mock_get_kendra_service.return_value = MagicMock()

        agent = HybridIBTAgent()
        assert agent.kendra_index_id == 'test-index'
        assert agent.aws_region == 'us-east-1'

class TestDirectKendraMode:
    """Tests for direct Kendra processing."""

    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_process_direct_kendra_success(self, mock_get_ncct_ids, mock_settings, mock_get_kendra_service):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        # Mock the function directly
        mock_get_ncct_ids.return_value = ['DENTAL_001', 'DENTAL_002']
        mock_get_kendra_service.return_value = MagicMock()

        agent = HybridIBTAgent()

        # Provide context with productId to trigger the product filtering path
        context = {'productId': '1'}
        result = agent._process_direct_kendra("dental benefits", context)

        assert result['success'] is True
        # Should return array of NCCT IDs
        response_text = result['response_text']
        assert isinstance(response_text, list)
        assert 'DENTAL_001' in response_text
        assert 'DENTAL_002' in response_text
        assert result['confidence'] == 8.0

class TestProcessQuery:
    """Tests for main process_query method."""

    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_success(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        mock_get_kendra_service.return_value = MagicMock()
        mock_direct_process.return_value = {
            "success": True,
            "response_text": "Direct response",
            "confidence": 6.0
        }

        agent = HybridIBTAgent()

        result = agent.process_query("test", "sess-001", {"productId": "1"})

        assert result['sessionId'] == 'sess-001'
        assert 'execution_time_ms' in result
        assert 'timestamp' in result
        mock_direct_process.assert_called_once()

    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_limit_exceeded(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        """Test that QueryLimitExceededError returns proper error message."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_kendra_service.return_value = MagicMock()

        from src.services.kendra_service import QueryLimitExceededError
        mock_direct_process.side_effect = QueryLimitExceededError("Kendra query limit exceeded")

        agent = HybridIBTAgent()

        result = agent.process_query("dental benefits", "sess-limit", {"productId": "1"})

        assert result['success'] is False
        assert result['sessionId'] == 'sess-limit'
        assert result['confidence'] == 0.0
        assert isinstance(result['responseText'], list)
        assert "maximum number of search requests" in result['responseText'][0]

    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_exception_handling(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        mock_get_kendra_service.return_value = MagicMock()

        from src.agent.hybrid_ibt import KendraSearchError
        mock_direct_process.side_effect = KendraSearchError("Test error")

        agent = HybridIBTAgent()

        result = agent.process_query("test", "sess-001", {"productId": "1"})

        assert result['success'] is False
        assert "technical difficulties" in result['responseText'][0]
        assert result['sessionId'] == 'sess-001'
