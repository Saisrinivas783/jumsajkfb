"""Test cases for enhanced Kendra service with comprehensive data extraction."""

import pytest
from unittest.mock import MagicMock, patch, call
from src.services.kendra_service import KendraService, get_kendra_service

class TestEnhancedKendraService:
    """Test cases for enhanced Kendra service functionality."""
    
    @patch('src.services.kendra_service.get_settings')
    def test_format_results_comprehensive_extraction(self, mock_settings):
        """Test that _format_results extracts all possible data from Kendra items."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        service = KendraService()
        
        # Mock Kendra response with comprehensive data
        mock_items = [
            {
                'DocumentTitle': {'Text': 'Dental Benefits Plan'},
                'DocumentExcerpt': {'Text': 'Comprehensive dental coverage including preventive care'},
                'DocumentURI': 'https://example.com/dental-plan.pdf',
                'Type': 'DOCUMENT',
                'ScoreAttributes': {'ScoreConfidence': 'HIGH'},
                'DocumentAttributes': [
                    {
                        'Key': 'NCCT_ID',
                        'Value': {'StringValue': 'DENTAL_001'}
                    },
                    {
                        'Key': 'Service_Name',
                        'Value': {'StringValue': 'Dental Preventive Care'}
                    },
                    {
                        'Key': 'Category',
                        'Value': {'StringValue': 'Health Benefits'}
                    },
                    {
                        'Key': 'Cost',
                        'Value': {'LongValue': 150}
                    }
                ]
            },
            {
                # Item with minimal data
                'DocumentTitle': {'Text': 'Vision Plan'},
                'Type': 'DOCUMENT',
                'DocumentAttributes': []
            },
            {
                # Item with different attribute structure
                'DocumentExcerpt': {'Text': 'Medical coverage details'},
                'Type': 'DOCUMENT',
                'DocumentAttributes': [
                    {
                        'Key': 'ID',
                        'Value': {'StringValue': 'MED_001'}
                    },
                    {
                        'Key': 'Title',
                        'Value': {'StringValue': 'Medical Benefits'}
                    }
                ]
            }
        ]
        
        results = service._format_results(mock_items)
        
        # Verify all items are included
        assert len(results) == 3
        
        # Verify first item with comprehensive data
        first_result = results[0]
        assert first_result['ncct_id'] == 'DENTAL_001'
        assert first_result['service_name'] == 'Dental Preventive Care'
        assert first_result['excerpt'] == 'Comprehensive dental coverage including preventive care'
        assert first_result['confidence_score'] == 'HIGH'
        assert first_result['document_uri'] == 'https://example.com/dental-plan.pdf'
        assert first_result['item_type'] == 'DOCUMENT'
        assert 'Category' in first_result['all_attributes']
        assert first_result['all_attributes']['Cost'] == '150'
        
        # Verify second item with minimal data
        second_result = results[1]
        assert second_result['service_name'] == 'Vision Plan'
        assert second_result['ncct_id'] == 'DOC_2'
        assert second_result['excerpt'] == 'No excerpt available'
        
        # Verify third item with different structure
        third_result = results[2]
        assert third_result['ncct_id'] == 'MED_001'
        assert third_result['service_name'] == 'Medical Benefits'
        assert third_result['excerpt'] == 'Medical coverage details'
    
    @patch('src.services.kendra_service.get_settings')
    def test_format_results_error_handling(self, mock_settings):
        """Test that _format_results handles malformed items gracefully."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        service = KendraService()
        
        # Mock items with various error conditions
        mock_items = [
            {
                'DocumentTitle': {'Text': 'Valid Item'},
                'Type': 'DOCUMENT'
            },
            {
                # Malformed item that will cause processing error
                'DocumentAttributes': [
                    {
                        'Key': 'BadAttribute',
                        'Value': None  # This will cause an error
                    }
                ]
            },
            None  # Completely invalid item
        ]
        
        results = service._format_results(mock_items)
        
        # Should still return results for all items
        assert len(results) == 3
        
        # First item should be processed normally
        assert results[0]['service_name'] == 'Valid Item'
        
        # Second item should be processed with enhanced ID generation
        assert results[1]['ncct_id'] == 'DOC_2'  # Updated expectation
        assert results[1]['service_name'] == 'Document 2'
        
        # Third item should have error handling
        assert results[2]['ncct_id'] == 'ERR_3'
        assert 'error' in results[2]


class TestDirectModeIntegration:
    """Test integration with direct mode processing."""
    
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_direct_mode_with_enhanced_kendra(self, mock_get_ncct_ids, mock_settings, mock_get_kendra_service):
        """Test direct mode integration with enhanced Kendra service."""
        from src.agent.hybrid_ibt import HybridIBTAgent
        
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        # Mock the function directly
        mock_get_ncct_ids.return_value = ['DENTAL_001', 'DENTAL_002']
        mock_get_kendra_service.return_value = MagicMock()
        
        agent = HybridIBTAgent()
        agent.use_llm = False
        
        # Provide context with productId
        context = {'productId': '1'}
        result = agent._process_direct_kendra("dental benefits", context)
        
        assert result['success'] is True
        # Should return array of NCCT IDs
        assert isinstance(result['response_text'], list)
        assert result['response_text'] == ['DENTAL_001', 'DENTAL_002']
        assert result['confidence'] == 8.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])