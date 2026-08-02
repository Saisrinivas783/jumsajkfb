"""Integration tests for enhanced Kendra service functionality."""

import pytest
from unittest.mock import MagicMock, patch
from src.services.kendra_service import KendraService
from src.agent.hybrid_ibt import HybridIBTAgent

class TestEnhancedKendraIntegration:
    """Integration tests for enhanced Kendra service with real-world scenarios."""
    
    @patch('src.services.kendra_service.get_settings')
    def test_comprehensive_data_extraction_integration(self, mock_settings):
        """Test end-to-end data extraction with various document structures."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        service = KendraService()
        mock_client = MagicMock()
        service._client = mock_client
        
        # Mock comprehensive Kendra response with various document types
        mock_response = {
            'ResultItems': [
                {
                    'DocumentTitle': {'Text': 'FEHB Blue Cross Blue Shield Focus Plan'},
                    'DocumentExcerpt': {'Text': 'Federal Employee Health Benefits plan offering comprehensive medical, dental, and vision coverage including x-ray and diagnostic imaging services.'},
                    'DocumentURI': 'https://fehb.gov/plans/blue-focus.pdf',
                    'Type': 'DOCUMENT',
                    'ScoreAttributes': {'ScoreConfidence': 'VERY_HIGH'},
                    'DocumentAttributes': [
                        {'Key': 'NCCT_ID', 'Value': {'StringValue': 'FEHB_BLUE_001'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'FEHB Blue Focus Plan'}},
                        {'Key': 'Plan_Type', 'Value': {'StringValue': 'Federal Employee Health Benefits'}},
                        {'Key': 'Coverage_Types', 'Value': {'StringListValue': ['Medical', 'Dental', 'Vision', 'Prescription']}},
                        {'Key': 'Annual_Cost', 'Value': {'LongValue': 2400}},
                        {'Key': 'Effective_Date', 'Value': {'DateValue': '2024-01-01'}}
                    ]
                },
                {
                    'DocumentTitle': {'Text': 'X-Ray and Diagnostic Imaging Coverage'},
                    'DocumentExcerpt': {'Text': 'Comprehensive coverage for x-ray services, CT scans, MRI, and other diagnostic imaging procedures under the Blue Focus plan.'},
                    'Type': 'DOCUMENT',
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'},
                    'DocumentAttributes': [
                        {'Key': 'NCCT_ID', 'Value': {'StringValue': 'XRAY_IMG_001'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Diagnostic Imaging Services'}},
                        {'Key': 'Coverage_Percentage', 'Value': {'LongValue': 80}},
                        {'Key': 'Copay_Amount', 'Value': {'LongValue': 25}}
                    ]
                },
                {
                    # Document with minimal structure
                    'DocumentTitle': {'Text': 'Preventive Care Benefits'},
                    'Type': 'DOCUMENT',
                    'DocumentAttributes': [
                        {'Key': 'ID', 'Value': {'StringValue': 'PREV_001'}}
                    ]
                },
                {
                    # Document with different attribute names
                    'DocumentExcerpt': {'Text': 'Abdominal health services including gastroenterology consultations and procedures.'},
                    'Type': 'DOCUMENT',
                    'DocumentAttributes': [
                        {'Key': 'Document_ID', 'Value': {'StringValue': 'ABDOM_001'}},
                        {'Key': 'Title', 'Value': {'StringValue': 'Abdominal Health Services'}},
                        {'Key': 'Specialty', 'Value': {'StringValue': 'Gastroenterology'}}
                    ]
                }
            ]
        }
        
        mock_client.query.return_value = mock_response
        
        result = service.search("blue focus x-ray fehb abdominal health benefits")
        
        # Verify comprehensive extraction
        assert result['success'] is True
        assert len(result['results']) == 4
        
        # Verify first result (comprehensive data)
        first_result = result['results'][0]
        assert first_result['ncct_id'] == 'FEHB_BLUE_001'
        assert first_result['service_name'] == 'FEHB Blue Focus Plan'
        assert 'Federal Employee Health Benefits' in first_result['excerpt']
        assert first_result['confidence_score'] == 'VERY_HIGH'
        assert first_result['all_attributes']['Plan_Type'] == 'Federal Employee Health Benefits'
        assert first_result['all_attributes']['Coverage_Types'] == 'Medical, Dental, Vision, Prescription'
        assert first_result['all_attributes']['Annual_Cost'] == '2400'
        
        # Verify second result (x-ray specific)
        second_result = result['results'][1]
        assert second_result['ncct_id'] == 'XRAY_IMG_001'
        assert second_result['service_name'] == 'Diagnostic Imaging Services'
        assert 'x-ray services' in second_result['excerpt']
        
        # Verify third result (minimal data)
        third_result = result['results'][2]
        assert third_result['ncct_id'] == 'PREV_001'
        assert third_result['service_name'] == 'Preventive Care Benefits'
        
        # Verify fourth result (different attribute structure)
        fourth_result = result['results'][3]
        assert fourth_result['ncct_id'] == 'ABDOM_001'
        assert fourth_result['service_name'] == 'Abdominal Health Services'
        assert 'gastroenterology' in fourth_result['excerpt'].lower()
    

    
    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    def test_direct_mode_with_real_kendra_data(self, mock_get_ncct_ids, mock_get_kendra_service, mock_settings):
        """Test direct mode processing with realistic Kendra data."""
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        
        # Mock the function directly
        mock_get_ncct_ids.return_value = ['DENTAL_001', 'DENTAL_002']
        mock_get_kendra_service.return_value = MagicMock()
        
        agent = HybridIBTAgent()

        # Provide context with productId
        context = {'productId': '1'}
        result = agent._process_direct_kendra("What are my benefits for blue focus x-ray fehb?", context)
        
        # Verify response contains actual NCCT IDs as array
        assert result['success'] is True
        assert result['confidence'] == 8.0  # High confidence for direct Kendra results
        
        response_text = result['response_text']
        assert isinstance(response_text, list)
        # Should return array of NCCT IDs
        assert 'DENTAL_001' in response_text
        assert 'DENTAL_002' in response_text
    

    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])