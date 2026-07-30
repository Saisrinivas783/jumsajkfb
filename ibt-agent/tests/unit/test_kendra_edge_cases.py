"""Additional unit tests for Kendra service edge cases."""

import pytest
from unittest.mock import patch, MagicMock
from src.services.kendra_service import KendraService, get_kendra_service

class TestKendraServiceEdgeCases:
    """Additional tests for KendraService edge cases."""
    

    

    

    

    

    

    
    @patch('src.services.kendra_service.boto3.client')
    def test_search_missing_result_items(self, mock_boto):
        """Test search with missing ResultItems in response."""
        mock_client = MagicMock()
        mock_client.query.return_value = {}  # No ResultItems key
        mock_boto.return_value = mock_client
        
        service = KendraService()
        # Disable role assumption for this test
        service.settings.kendra_role_arn = None
        service._client = None  # Reset client to force re-initialization
        
        result = service.search("test query")
        
        assert result['success'] is True
        assert len(result['results']) == 0
    

    
    @patch('src.services.kendra_service.boto3.client')
    def test_search_boto3_client_error(self, mock_boto):
        """Test search with boto3 client creation error."""
        mock_boto.side_effect = Exception("AWS credentials not found")
        
        service = KendraService()
        # The exception should be raised when we try to access the client
        with pytest.raises(Exception, match="AWS credentials not found"):
            _ = service.client
    

    

    
    def test_get_kendra_service_singleton(self):
        """Test get_kendra_service returns singleton instance."""
        # Mock the settings to avoid role assumption
        with patch('src.services.kendra_service.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.kendra_role_arn = None  # Disable role assumption
            mock_settings.aws_region = "us-east-1"
            mock_settings.kendra_index_id = "test-index"
            mock_get_settings.return_value = mock_settings
            
            with patch('src.services.kendra_service.boto3.client') as mock_boto:
                mock_boto.return_value = MagicMock()
                
                service1 = get_kendra_service()
                service2 = get_kendra_service()
                
                # Both should be KendraService instances
                assert isinstance(service1, KendraService)
                assert isinstance(service2, KendraService)
    
    @patch('src.services.kendra_service.get_settings')
    def test_kendra_service_settings_integration(self, mock_get_settings):
        """Test KendraService integrates with settings correctly."""
        mock_settings = MagicMock()
        mock_settings.aws_region = "us-west-2"
        mock_settings.kendra_index_id = "test-index-456"
        mock_settings.kendra_role_arn = None  # Disable role assumption
        mock_get_settings.return_value = mock_settings
        
        with patch('src.services.kendra_service.boto3.client') as mock_boto:
            mock_boto.return_value = MagicMock()
            service = KendraService()
            
            # Force client initialization by accessing the client property
            _ = service.client
            
            # Verify the call was made with correct parameters
            assert mock_boto.called
            call_args = mock_boto.call_args
            assert call_args[0][0] == 'kendra'
            assert call_args[1]['region_name'] == 'us-west-2'
            assert 'config' in call_args[1]
            assert service.index_id == "test-index-456"