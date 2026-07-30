"""Test cases for direct Kendra integration."""

from unittest.mock import MagicMock, patch

from src.agent.hybrid_ibt import HybridIBTAgent


class TestDirectModeIntegration:
    """Test integration with direct mode processing."""

    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_direct_mode_with_enhanced_kendra(self, mock_get_ncct_ids, mock_settings, mock_get_kendra_service):
        """Test direct mode integration with Kendra NCCT ID results."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_ncct_ids.return_value = ['DENTAL_001', 'DENTAL_002']
        mock_get_kendra_service.return_value = MagicMock()

        agent = HybridIBTAgent()
        context = {'productId': '6'}
        result = agent._process_direct_kendra("dental benefits", context)

        assert result['success'] is True
        assert isinstance(result['response_text'], list)
        assert result['response_text'] == ['DENTAL_001', 'DENTAL_002']
        assert result['confidence'] == 8.0
        assert result['product_id'] == '6'
