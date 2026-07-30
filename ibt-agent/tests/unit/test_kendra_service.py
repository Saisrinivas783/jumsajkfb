"""Unit tests for Kendra service."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.services.kendra_service import KendraService, QueryLimitExceededError, get_ncct_ids_by_product


class TestKendraService:
    """Tests for KendraService class."""

    @patch('src.services.kendra_service.boto3.client')
    def test_init(self, mock_boto):
        mock_client = MagicMock()
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')
        assert service.index_id == 'test-index'

        _ = service.client

        assert mock_boto.called

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
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}},
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'VERY_HIGH'},
                },
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT456'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Vision'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}},
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'},
                },
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT789'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Low Confidence'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}},
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'MEDIUM'},
                },
            ]
        }
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')
        ncct_ids = service.get_ncct_ids_by_product('dental benefits', '1')

        assert ncct_ids == ['NCCT123', 'NCCT456', 'NCCT789']

        expected_filter = {
            "AndAllFilters": [
                {"EqualsTo": {"Key": "plan", "Value": {"StringValue": "standard/basic"}}},
                {"EqualsTo": {"Key": "brochure", "Value": {"StringValue": "fehb"}}},
            ]
        }

        mock_client.query.assert_called_once_with(
            IndexId='test-index',
            QueryText='dental benefits',
            PageSize=service.settings.kendra_page_size,
            RequestedDocumentAttributes=['NCCTID'],
            AttributeFilter=expected_filter,
        )

    @patch('boto3.client')
    def test_get_ncct_ids_by_product_no_product_filter(self, mock_boto):
        """Test get_ncct_ids_by_product without product filter."""
        mock_client = MagicMock()
        mock_client.query.return_value = {
            'ResultItems': [
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT789'}},
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'},
                }
            ]
        }
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')
        ncct_ids = service.get_ncct_ids_by_product('general query', None)

        assert ncct_ids == ['NCCT789']

        mock_client.query.assert_called_once_with(
            IndexId='test-index',
            QueryText='general query',
            PageSize=service.settings.kendra_page_size,
            RequestedDocumentAttributes=['NCCTID'],
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
                {"EqualsTo": {"Key": "plan", "Value": {"StringValue": "standard/basic"}}},
                {"EqualsTo": {"Key": "brochure", "Value": {"StringValue": "pshb"}}},
            ]
        }

        assert filter_result == expected

    @patch('src.services.kendra_service.get_settings')
    def test_build_attribute_filter_none_config(self, mock_settings):
        """Test _build_attribute_filter returns None for empty config."""
        mock_settings.return_value = MagicMock(kendra_index_id='test-index', aws_region='us-east-1')
        service = KendraService()

        assert service._build_attribute_filter(None) is None
        assert service._build_attribute_filter({}) is None

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
        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'Query',
        )
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')

        with pytest.raises(QueryLimitExceededError, match="Kendra query limit exceeded"):
            service.get_ncct_ids_by_product('dental benefits', '1')

    @patch('boto3.client')
    def test_get_ncct_ids_by_product_client_error_non_throttling(self, mock_boto):
        """Test non-throttling ClientError raises RuntimeError."""
        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'Invalid query'}},
            'Query',
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
