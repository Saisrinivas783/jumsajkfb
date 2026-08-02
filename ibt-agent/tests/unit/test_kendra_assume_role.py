"""
Unit tests for KendraService's use of the shared AssumedRoleClientFactory.

STS retry/backoff behavior itself is covered in tests/unit/test_aws/test_assume_role.py;
these tests cover KendraService's integration with it (factory wiring, client
caching, boto config, and the credential refresh worker lifecycle).
"""

from unittest.mock import Mock, patch, MagicMock

import pytest

from src.aws.assume_role import CredentialsError
from src.services.kendra_service import KendraService


class TestKendraAssumeRole:
    """Test KendraService's integration with AssumedRoleClientFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kendra_service = KendraService('test-index', 'us-east-1')

    def test_get_kendra_client_no_arn_uses_default_credentials(self):
        """No role ARN configured -> default-credentials boto3 client, not the factory."""
        self.kendra_service.settings.kendra_role_arn = None

        with patch('src.services.kendra_service.boto3.client') as mock_boto_client:
            mock_kendra = Mock()
            mock_boto_client.return_value = mock_kendra

            client = self.kendra_service._get_kendra_client()

            call_args = mock_boto_client.call_args
            assert call_args[0][0] == 'kendra'
            assert call_args[1]['region_name'] == 'us-east-1'
            assert client is mock_kendra

    def test_get_kendra_client_with_role_assumption(self):
        """A configured role ARN routes client construction through the factory."""
        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'

        with patch('src.services.kendra_service.AssumedRoleClientFactory') as mock_factory_cls:
            mock_factory = Mock()
            mock_kendra = Mock()
            mock_factory.client.return_value = mock_kendra
            mock_factory_cls.return_value = mock_factory

            client = self.kendra_service._get_kendra_client()

            mock_factory_cls.assert_called_once_with(
                role_arn=self.kendra_service.settings.kendra_role_arn,
                session_name=self.kendra_service.settings.kendra_session_name,
                duration_seconds=self.kendra_service.settings.kendra_role_duration,
                region_name='us-east-1',
                method='kendra-assume-role',
            )
            mock_factory.client.assert_called_once()
            assert mock_factory.client.call_args[0][0] == 'kendra'
            assert client is mock_kendra

    def test_get_kendra_client_role_assumption_failure_propagates(self):
        """A factory that can't build a session raises CredentialsError, not a silent fallback."""
        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'

        with patch('src.services.kendra_service.AssumedRoleClientFactory') as mock_factory_cls:
            mock_factory = Mock()
            mock_factory.client.side_effect = CredentialsError('Role assumption failed: AccessDenied')
            mock_factory_cls.return_value = mock_factory

            with pytest.raises(CredentialsError, match='AccessDenied'):
                self.kendra_service._get_kendra_client()

    def test_client_caching(self):
        """Test that Kendra client is cached after first creation."""
        self.kendra_service.settings.kendra_role_arn = None

        with patch('src.services.kendra_service.boto3.client') as mock_boto_client:
            mock_kendra = Mock()
            mock_boto_client.return_value = mock_kendra

            client1 = self.kendra_service._get_kendra_client()
            client2 = self.kendra_service._get_kendra_client()

            mock_boto_client.assert_called_once()
            assert client1 is client2 is mock_kendra

    def test_start_credential_refresh_starts_single_daemon_thread(self):
        """Background refresh should run as a single daemon worker per instance."""
        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'

        with patch('src.services.kendra_service.CredentialRefreshWorker') as mock_worker_cls:
            mock_worker = Mock()
            mock_worker_cls.return_value = mock_worker

            self.kendra_service.start_credential_refresh()
            self.kendra_service.start_credential_refresh()

            mock_worker_cls.assert_called_once()
            assert mock_worker.start.call_count == 2

    def test_stop_credential_refresh_without_start_is_a_no_op(self):
        """Stopping before starting shouldn't raise."""
        self.kendra_service.stop_credential_refresh()

    def test_get_kendra_client_without_role_assumption(self):
        """Test Kendra client creation without role assumption."""
        mock_kendra = Mock()

        with patch('src.services.kendra_service.boto3.client', return_value=mock_kendra) as mock_boto_client:
            self.kendra_service.settings.kendra_role_arn = None
            self.kendra_service._client = None

            client = self.kendra_service._get_kendra_client()

            call_args = mock_boto_client.call_args
            assert call_args[0][0] == 'kendra'
            assert call_args[1]['region_name'] == 'us-east-1'
            assert 'config' in call_args[1]

            assert client == mock_kendra

    def test_get_ncct_ids_by_product_with_assume_role(self):
        """Test product-filtered Kendra query with assume role, via the factory."""
        mock_kendra = Mock()
        mock_kendra.query.return_value = {
            'ResultItems': [
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT123'}},
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'},
                }
            ]
        }

        with patch('src.services.kendra_service.AssumedRoleClientFactory') as mock_factory_cls:
            mock_factory = Mock()
            mock_factory.client.return_value = mock_kendra
            mock_factory_cls.return_value = mock_factory

            self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'
            self.kendra_service._client = None

            result = self.kendra_service.get_ncct_ids_by_product('test query', '1')

            mock_factory_cls.assert_called_once()
            assert result == ['NCCT123']

    def test_boto_config_creation(self):
        """Test boto configuration creation."""
        config = self.kendra_service._get_boto_config()

        assert config is not None
        assert hasattr(config, 'read_timeout')
        assert hasattr(config, 'connect_timeout')
        assert config.max_pool_connections == self.kendra_service.settings.kendra_max_pool_connections
