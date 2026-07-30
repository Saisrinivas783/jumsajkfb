"""Unit tests for Kendra service."""

import pytest
from unittest.mock import patch, MagicMock
from src.services.kendra_service import KendraService, get_ncct_ids_by_product
from src.config.constants import DEFAULT_PAGE_SIZE, NO_EXCERPT_MESSAGE

class TestKendraService:
    """Tests for KendraService class."""
    
    @patch('src.services.kendra_service.boto3.client')
    def test_init(self, mock_boto):
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        assert service.index_id == 'test-index'
        
        # Force client initialization by accessing the client property
        _ = service.client
        
        # Verify the client was created
        assert mock_boto.called
    
    @patch('boto3.client')
    def test_search_success(self, mock_boto, mock_kendra_response):
        mock_client = MagicMock()
        mock_client.query.return_value = mock_kendra_response
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        result = service.search('dental benefits')
        
        assert result['success'] is True
        assert len(result['results']) == 1
        assert result['results'][0]['ncct_id'] == 'NCCT123'
        assert result['results'][0]['service_name'] == 'Dental Coverage'
        
        # Verify query was called with correct parameters
        mock_client.query.assert_called_once_with(
            IndexId='test-index',
            QueryText='dental benefits',
            PageSize=DEFAULT_PAGE_SIZE
        )
    

    

    
    @patch('src.services.kendra_service.get_settings')
    def test_format_results_with_all_attributes(self, mock_settings):
        mock_settings.return_value = MagicMock(kendra_index_id='test-index', aws_region='us-east-1')
        service = KendraService()
        
        raw_items = [
            {
                'DocumentAttributes': [
                    {'Key': 'NCCT_ID', 'Value': {'StringValue': 'NCCT123'}},
                    {'Key': 'Service_Name', 'Value': {'StringValue': 'Dental'}}
                ],
                'DocumentExcerpt': {'Text': 'Dental coverage info'},
                'ScoreAttributes': {'ScoreConfidence': 'HIGH'}
            }
        ]
        
        results = service._format_results(raw_items)
        
        assert len(results) == 1
        assert results[0]['ncct_id'] == 'NCCT123'
        assert results[0]['service_name'] == 'Dental'
        assert results[0]['excerpt'] == 'Dental coverage info'
        assert results[0]['confidence_score'] == 'HIGH'
    

    
    @patch('src.services.kendra_service.get_settings')
    def test_format_results_no_excerpt_uses_constant(self, mock_settings):
        mock_settings.return_value = MagicMock(kendra_index_id='test-index', aws_region='us-east-1')
        service = KendraService()
        
        raw_items = [
            {
                'DocumentAttributes': [
                    {'Key': 'NCCT_ID', 'Value': {'StringValue': 'NCCT456'}},
                    {'Key': 'Service_Name', 'Value': {'StringValue': 'Vision'}}
                ],
                'ScoreAttributes': {'ScoreConfidence': 'LOW'}
            }
        ]
        
        results = service._format_results(raw_items)
        
        assert len(results) == 1
        assert results[0]['excerpt'] == NO_EXCERPT_MESSAGE
    
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_success(self, mock_boto):
        """Test get_ncct_ids_by_product returns only NCCT IDs with AttributeFilter."""
        mock_client = MagicMock()
        mock_client.query.return_value = {
            'ResultItems': [
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT123'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Dental'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'VERY_HIGH'}
                },
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT456'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Vision'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'}
                },
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT789'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Low Confidence'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'MEDIUM'}
                }
            ]
        }
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        ncct_ids = service.get_ncct_ids_by_product('dental benefits', '1')
        
        assert ncct_ids == ['NCCT123', 'NCCT456', 'NCCT789']  # VERY_HIGH, HIGH, and MEDIUM confidence
        
        # Verify query was called with correct parameters including AttributeFilter
        expected_filter = {
            "AndAllFilters": [
                {
                    "EqualsTo": {
                        "Key": "plan",
                        "Value": {
                            "StringValue": "standard/basic"
                        }
                    }
                },
                {
                    "EqualsTo": {
                        "Key": "brochure",
                        "Value": {
                            "StringValue": "fehb"
                        }
                    }
                }
            ]
        }
        
        mock_client.query.assert_called_once_with(
            IndexId='test-index',
            QueryText='dental benefits',
            PageSize=DEFAULT_PAGE_SIZE,
            RequestedDocumentAttributes=['NCCTID'],
            AttributeFilter=expected_filter
        )
    
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_no_product_filter(self, mock_boto):
        """Test get_ncct_ids_by_product without product filter."""
        mock_client = MagicMock()
        mock_client.query.return_value = {
            'ResultItems': [
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT789'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'}
                }
            ]
        }
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        ncct_ids = service.get_ncct_ids_by_product('general query', None)
        
        assert ncct_ids == ['NCCT789']
        
        # Verify no AttributeFilter was used
        mock_client.query.assert_called_once_with(
            IndexId='test-index',
            QueryText='general query',
            PageSize=DEFAULT_PAGE_SIZE,
            RequestedDocumentAttributes=['NCCTID']
        )
    

    @patch('src.services.kendra_service.get_settings')
    def test_build_attribute_filter(self, mock_settings):
        """Test _build_attribute_filter creates correct filter structure."""
        mock_settings.return_value = MagicMock(kendra_index_id='test-index', aws_region='us-east-1')
        service = KendraService()
        
        product_config = {"plan": "standard/basic", "brochure": "pshb"}
        filter_result = service._build_attribute_filter(product_config)
        
        expected = {
            "AndAllFilters": [
                {
                    "EqualsTo": {
                        "Key": "plan",
                        "Value": {
                            "StringValue": "standard/basic"
                        }
                    }
                },
                {
                    "EqualsTo": {
                        "Key": "brochure",
                        "Value": {
                            "StringValue": "pshb"
                        }
                    }
                }
            ]
        }
        
        assert filter_result == expected
    
    @patch('src.services.kendra_service.get_settings')
    def test_build_attribute_filter_none_config(self, mock_settings):
        """Test _build_attribute_filter returns None for empty config."""
        mock_settings.return_value = MagicMock(kendra_index_id='test-index', aws_region='us-east-1')
        service = KendraService()
        
        filter_result = service._build_attribute_filter(None)
        assert filter_result is None
        
        filter_result = service._build_attribute_filter({})
        assert filter_result is None

    @patch('boto3.client')
    def test_get_ncct_ids_by_product_empty_results(self, mock_boto):
        """Test get_ncct_ids_by_product returns empty list when no results."""
        mock_client = MagicMock()
        mock_client.query.return_value = {'ResultItems': []}
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        ncct_ids = service.get_ncct_ids_by_product('unknown query', '1')
        
        assert ncct_ids == []
    
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_exception_handling(self, mock_boto):
        """Test get_ncct_ids_by_product raises exception on error."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception('Kendra error')
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        
        with pytest.raises(RuntimeError, match="Kendra search failed for product 1: Kendra error"):
            service.get_ncct_ids_by_product('test query', '1')
    
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_throttling_raises_query_limit_exceeded(self, mock_boto):
        """Test ThrottlingException raises QueryLimitExceededError."""
        from botocore.exceptions import ClientError
        from src.services.kendra_service import QueryLimitExceededError
        
        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'Query'
        )
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        
        with pytest.raises(QueryLimitExceededError, match="Kendra query limit exceeded"):
            service.get_ncct_ids_by_product('dental benefits', '1')
    
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_client_error_non_throttling(self, mock_boto):
        """Test non-throttling ClientError raises RuntimeError."""
        from botocore.exceptions import ClientError
        
        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'Invalid query'}},
            'Query'
        )
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        
        with pytest.raises(RuntimeError, match="Kendra search failed for product 1: ValidationException"):
            service.get_ncct_ids_by_product('dental benefits', '1')
    
    @patch('src.services.kendra_service.get_kendra_service')
    def test_get_ncct_ids_by_product_function(self, mock_get_service):
        """Test the convenience function get_ncct_ids_by_product."""
        mock_service = MagicMock()
        mock_service.get_ncct_ids_by_product.return_value = ['NCCT123', 'NCCT456']
        mock_get_service.return_value = mock_service
        
        result = get_ncct_ids_by_product('dental benefits', '1')
        
        assert result == ['NCCT123', 'NCCT456']
        mock_service.get_ncct_ids_by_product.assert_called_once_with('dental benefits', '1')