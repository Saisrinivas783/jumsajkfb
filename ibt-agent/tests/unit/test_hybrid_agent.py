"""Unit tests for HybridIBTAgent."""

import pytest
from unittest.mock import MagicMock, patch
from src.agent.hybrid_ibt import HybridIBTAgent

class TestHybridIBTAgent:
    """Tests for HybridIBTAgent class."""
    
    @patch.dict('os.environ', {'USE_LLM': 'true'})
    @patch('src.agent.hybrid_ibt.LLM_AVAILABLE', True)
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_init_with_llm_mode(self, mock_settings, mock_get_kendra_service):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        
        agent = HybridIBTAgent()
        assert agent.use_llm is True
        assert agent.kendra_index_id == 'test-index'
        assert agent.aws_region == 'us-east-1'
    
    @patch.dict('os.environ', {'USE_LLM': 'false'})
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_init_with_direct_mode(self, mock_settings, mock_get_kendra_service):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        
        agent = HybridIBTAgent()
        assert agent.use_llm is False
    
    @patch('src.agent.hybrid_ibt.LLM_AVAILABLE', True)
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_set_mode(self, mock_settings, mock_get_kendra_service):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        
        agent = HybridIBTAgent()
        agent.set_mode(False)
        assert agent.use_llm is False
        
        agent.set_mode(True)
        assert agent.use_llm is True
    
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_get_mode_info(self, mock_settings, mock_get_kendra_service):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-west-2'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        
        agent = HybridIBTAgent()
        info = agent.get_mode_info()
        assert info['kendra_index_id'] == 'test-index'
        assert info['aws_region'] == 'us-west-2'
        assert 'current_mode' in info
        assert 'using_kendra_role_assumption' in info
        assert 'using_bedrock_role_assumption' in info

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
        agent.use_llm = False
        
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
    

    


class TestLLMMode:
    """Tests for LLM-enhanced processing."""
    
    @patch('src.agent.hybrid_ibt.HybridIBTAgent.agent_executor')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_with_llm_success(self, mock_settings, mock_get_kendra_service, mock_executor):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        mock_executor.invoke.return_value = {
            "output": "Here are your benefits: <a href='NCCT123'>Dental Coverage</a>"
        }
        
        agent = HybridIBTAgent()
        result = agent._process_with_llm("dental benefits")
        
        assert result['success'] is True
        assert result['confidence'] == 8.0
        assert 'NCCT123' in result['response_text']
    
    @patch('src.agent.hybrid_ibt.HybridIBTAgent.agent_executor')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_with_llm_context(self, mock_settings, mock_get_kendra_service, mock_executor):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        mock_executor.invoke.return_value = {"output": "Response"}
        
        agent = HybridIBTAgent()
        context = {"userName": "John", "userType": "member"}
        agent._process_with_llm("test query", context)
        
        call_args = mock_executor.invoke.call_args[0][0]
        assert "John" in call_args["input"]
        assert "member" in call_args["input"]

class TestProcessQuery:
    """Tests for main process_query method."""
    
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_with_llm')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_llm_mode(self, mock_settings, mock_get_kendra_service, mock_llm_process):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        mock_llm_process.return_value = {
            "success": True,
            "response_text": "LLM response",
            "confidence": 8.0
        }
        
        agent = HybridIBTAgent()
        agent.use_llm = True
        
        result = agent.process_query("test", "sess-001")
        
        assert result['mode'] == 'llm_enhanced'
        assert result['sessionId'] == 'sess-001'
        assert 'execution_time_ms' in result
        assert 'timestamp' in result
        mock_llm_process.assert_called_once()
    
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_direct_mode(self, mock_settings, mock_get_kendra_service, mock_direct_process):
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
        agent.use_llm = False
        
        result = agent.process_query("test", "sess-001")
        
        assert result['mode'] == 'direct_kendra'
        assert result['sessionId'] == 'sess-001'
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
        agent.use_llm = False
        
        result = agent.process_query("dental benefits", "sess-limit")
        
        assert result['success'] is False
        assert result['sessionId'] == 'sess-limit'
        assert result['confidence'] == 0.0
        assert isinstance(result['responseText'], list)
        assert "maximum number of search requests" in result['responseText'][0]
    
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_exception_handling(self, mock_settings, mock_get_kendra_service):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        mock_get_kendra_service.return_value = MagicMock()
        
        agent = HybridIBTAgent()
        agent.use_llm = True  # Force LLM mode for this test
        
        # Import the specific exception that will be caught
        from src.agent.hybrid_ibt import KendraSearchError
        
        with patch.object(agent, '_process_with_llm', side_effect=KendraSearchError("Test error")):
            result = agent.process_query("test", "sess-001")
            
            assert result['success'] is False
            assert "technical difficulties" in result['responseText'][0]
            assert result['sessionId'] == 'sess-001'