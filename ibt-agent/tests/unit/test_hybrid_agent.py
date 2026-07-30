"""Unit tests for HybridIBTAgent."""

import pytest
from unittest.mock import patch

from src.agent.hybrid_ibt import HybridIBTAgent, KendraSearchError, QueryProcessingError
from src.services.kendra_service import QueryLimitExceededError


class TestHybridIBTAgent:
    """Tests for HybridIBTAgent class."""

    @patch('src.agent.hybrid_ibt.get_settings')
    def test_init_configures_kendra_fields(self, mock_settings):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        agent = HybridIBTAgent()

        assert agent.kendra_index_id == 'test-index'
        assert agent.aws_region == 'us-east-1'


class TestDirectKendraProcessing:
    """Tests for direct Kendra processing."""

    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_process_direct_kendra_uses_orchestrator_product_id(self, mock_get_ncct_ids, mock_settings):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_ncct_ids.return_value = ['DENTAL_001', 'DENTAL_002']

        agent = HybridIBTAgent()
        context = {'userName': 'test_user', 'userType': 'member', 'source': 'IBTPage', 'productId': '6'}
        result = agent._process_direct_kendra("dental benefits", context)

        assert result['success'] is True
        assert result['response_text'] == ['DENTAL_001', 'DENTAL_002']
        assert result['confidence'] == 8.0
        assert result['product_id'] == '6'
        assert result['ncct_count'] == 2
        mock_get_ncct_ids.assert_called_once_with("dental benefits", '6')

    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_process_direct_kendra_trims_product_id(self, mock_get_ncct_ids, mock_settings):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_ncct_ids.return_value = ['DENTAL_001']

        agent = HybridIBTAgent()
        result = agent._process_direct_kendra("dental benefits", {'productId': ' 6 '})

        assert result['product_id'] == '6'
        mock_get_ncct_ids.assert_called_once_with("dental benefits", '6')

    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_process_direct_kendra_deduplicates_results(self, mock_get_ncct_ids, mock_settings):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_ncct_ids.return_value = ['DENTAL_001', 'DENTAL_001', 'DENTAL_002']

        agent = HybridIBTAgent()
        result = agent._process_direct_kendra("dental benefits", {'productId': '6'})

        assert result['response_text'] == ['DENTAL_001', 'DENTAL_002']
        assert result['ncct_count'] == 2

    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_process_direct_kendra_missing_product_id_fails(self, mock_get_ncct_ids, mock_settings):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        agent = HybridIBTAgent()

        with pytest.raises(QueryProcessingError, match="context.productId is required"):
            agent._process_direct_kendra("dental benefits", {'product_id': '6'})

        mock_get_ncct_ids.assert_not_called()

    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_process_direct_kendra_blank_product_id_fails(self, mock_get_ncct_ids, mock_settings):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        agent = HybridIBTAgent()

        with pytest.raises(QueryProcessingError, match="context.productId is required"):
            agent._process_direct_kendra("dental benefits", {'productId': '   '})

        mock_get_ncct_ids.assert_not_called()


class TestProcessQuery:
    """Tests for main process_query method."""

    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_uses_direct_kendra(self, mock_settings, mock_direct_process):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_direct_process.return_value = {
            "success": True,
            "response_text": ['DENTAL_001'],
            "confidence": 8.0
        }
        context = {'userName': 'test_user', 'userType': 'member', 'source': 'IBTPage', 'productId': '6'}

        agent = HybridIBTAgent()
        result = agent.process_query("test", "sess-001", context)

        assert result['mode'] == 'direct_kendra'
        assert result['sessionId'] == 'sess-001'
        assert result['responseText'] == ['DENTAL_001']
        assert 'execution_time_ms' in result
        assert 'timestamp' in result
        mock_direct_process.assert_called_once_with("test", context)

    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_limit_exceeded(self, mock_settings, mock_direct_process):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_direct_process.side_effect = QueryLimitExceededError("Kendra query limit exceeded")
        context = {'productId': '6'}

        agent = HybridIBTAgent()
        result = agent.process_query("dental benefits", "sess-limit", context)

        assert result['success'] is False
        assert result['mode'] == 'direct_kendra'
        assert result['sessionId'] == 'sess-limit'
        assert result['confidence'] == 0.0
        assert isinstance(result['responseText'], list)
        assert "maximum number of search requests" in result['responseText'][0]

    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_exception_handling(self, mock_settings, mock_direct_process):
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_direct_process.side_effect = KendraSearchError("Test error")
        context = {'productId': '6'}

        agent = HybridIBTAgent()
        result = agent.process_query("test", "sess-001", context)

        assert result['success'] is False
        assert result['mode'] == 'direct_kendra'
        assert "technical difficulties" in result['responseText'][0]
        assert result['sessionId'] == 'sess-001'
