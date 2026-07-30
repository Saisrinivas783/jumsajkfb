"""Additional unit tests for Kendra service edge cases."""

from unittest.mock import MagicMock, patch

from src.services.kendra_service import KendraService, get_kendra_service


class TestKendraServiceEdgeCases:
    """Additional tests for KendraService edge cases."""

    @patch('src.services.kendra_service.boto3.client')
    def test_client_creation_error(self, mock_boto):
        """Test Kendra client creation error propagation."""
        mock_boto.side_effect = Exception("AWS credentials not found")

        service = KendraService()

        with patch.object(service.settings, 'kendra_role_arn', None):
            try:
                _ = service.client
            except Exception as exc:
                assert str(exc) == "AWS credentials not found"
            else:
                raise AssertionError("Expected client creation to fail")

    def test_get_kendra_service_singleton(self):
        """Test get_kendra_service returns singleton instance."""
        with patch('src.services.kendra_service.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.kendra_role_arn = None
            mock_settings.aws_region = "us-east-1"
            mock_settings.kendra_index_id = "test-index"
            mock_get_settings.return_value = mock_settings

            with patch('src.services.kendra_service.boto3.client') as mock_boto:
                mock_boto.return_value = MagicMock()
                get_kendra_service.cache_clear()

                service1 = get_kendra_service()
                service2 = get_kendra_service()

                assert service1 is service2
                assert isinstance(service1, KendraService)
                assert isinstance(service2, KendraService)
                get_kendra_service.cache_clear()

    @patch('src.services.kendra_service.get_settings')
    def test_kendra_service_settings_integration(self, mock_get_settings):
        """Test KendraService integrates with settings correctly."""
        mock_settings = MagicMock()
        mock_settings.aws_region = "us-west-2"
        mock_settings.kendra_index_id = "test-index-456"
        mock_settings.kendra_role_arn = None
        mock_get_settings.return_value = mock_settings

        with patch('src.services.kendra_service.boto3.client') as mock_boto:
            mock_boto.return_value = MagicMock()
            service = KendraService()

            _ = service.client

            assert mock_boto.called
            call_args = mock_boto.call_args
            assert call_args[0][0] == 'kendra'
            assert call_args[1]['region_name'] == 'us-west-2'
            assert 'config' in call_args[1]
            assert service.index_id == "test-index-456"
