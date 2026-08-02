"""
Unit tests for Kendra assume role functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from src.services.kendra_service import KendraService
from src.config.settings import get_settings


class TestKendraAssumeRole:
    """Test assume role functionality in KendraService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kendra_service = KendraService('test-index', 'us-east-1')

    @patch('src.services.kendra_service.boto3.client')
    def test_assume_kendra_role_success(self, mock_boto_client):
        """Test successful Kendra role assumption."""
        # Mock STS client and response
        mock_sts = Mock()
        mock_boto_client.return_value = mock_sts
        
        mock_response = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': '2024-01-01T12:00:00Z'
            }
        }
        mock_sts.assume_role.return_value = mock_response
        
        # Set up role ARN
        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'
        
        # Test role assumption
        credentials = self.kendra_service._assume_kendra_role()
        
        # Verify STS call
        mock_sts.assume_role.assert_called_once_with(
            RoleArn='arn:aws:iam::054940911799:role/ibt-ai-index-role',
            RoleSessionName='ibt-agent-kendra',
            DurationSeconds=3600
        )
        
        # Verify returned credentials
        assert credentials['aws_access_key_id'] == 'ASIA123456789'
        assert credentials['aws_secret_access_key'] == 'secret123'
        assert credentials['aws_session_token'] == 'token123'

    def test_assume_kendra_role_no_arn(self):
        """Test role assumption when no ARN is configured."""
        self.kendra_service.settings.kendra_role_arn = None
        
        with pytest.raises(ValueError, match="KENDRA_ROLE_ARN is not configured"):
            self.kendra_service._assume_kendra_role()

    @patch('src.services.kendra_service.boto3.client')
    def test_assume_kendra_role_access_denied(self, mock_boto_client):
        """Test role assumption failure due to access denied."""
        # Mock STS client with access denied error
        mock_sts = Mock()
        mock_boto_client.return_value = mock_sts
        
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_sts.assume_role.side_effect = ClientError(error_response, 'AssumeRole')
        
        # Set up role ARN
        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'
        
        # Test role assumption failure
        with pytest.raises(RuntimeError, match="Role assumption failed: AccessDenied"):
            self.kendra_service._assume_kendra_role()

    @patch('src.services.kendra_service.boto3.client')
    def test_get_kendra_client_with_role_assumption(self, mock_boto_client):
        """Test Kendra client creation with role assumption."""
        # Mock STS client
        mock_sts = Mock()
        mock_kendra = Mock()
        
        def boto_client_side_effect(service_name, **kwargs):
            if service_name == 'sts':
                return mock_sts
            elif service_name == 'kendra':
                return mock_kendra
            return Mock()
        
        mock_boto_client.side_effect = boto_client_side_effect
        
        # Mock successful role assumption
        mock_response = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': '2024-01-01T12:00:00Z'
            }
        }
        mock_sts.assume_role.return_value = mock_response
        
        # Set up role ARN
        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'
        
        # Reset client to force re-initialization
        self.kendra_service._client = None
        
        # Test client creation
        client = self.kendra_service._get_kendra_client()
        
        # Verify role assumption was called
        mock_sts.assume_role.assert_called_once()
        
        # Verify Kendra client was created with assumed credentials
        kendra_calls = [call for call in mock_boto_client.call_args_list if call[0][0] == 'kendra']
        assert len(kendra_calls) > 0
        kendra_call = kendra_calls[0]
        assert kendra_call[1]['aws_access_key_id'] == 'ASIA123456789'
        assert kendra_call[1]['aws_secret_access_key'] == 'secret123'
        assert kendra_call[1]['aws_session_token'] == 'token123'
        
        assert client == mock_kendra

    @patch('src.services.kendra_service.boto3.client')
    def test_get_kendra_client_without_role_assumption(self, mock_boto_client):
        """Test Kendra client creation without role assumption."""
        mock_kendra = Mock()
        mock_boto_client.return_value = mock_kendra
        
        # No role ARN configured
        self.kendra_service.settings.kendra_role_arn = None
        
        # Reset client to force re-initialization
        self.kendra_service._client = None
        
        # Test client creation
        client = self.kendra_service._get_kendra_client()
        
        # Verify the call was made with correct parameters
        call_args = mock_boto_client.call_args
        assert call_args[0][0] == 'kendra'
        assert call_args[1]['region_name'] == 'us-east-1'
        assert 'config' in call_args[1]
        
        assert client == mock_kendra

    @patch('src.services.kendra_service.boto3.client')
    def test_client_caching(self, mock_boto_client):
        """Test that Kendra client is cached after first creation."""
        mock_kendra = Mock()
        mock_boto_client.return_value = mock_kendra
        
        # No role ARN configured for simplicity
        self.kendra_service.settings.kendra_role_arn = None
        
        # Reset client
        self.kendra_service._client = None
        
        # Get client twice
        client1 = self.kendra_service._get_kendra_client()
        client2 = self.kendra_service._get_kendra_client()
        
        # Verify boto3.client was called only once
        mock_boto_client.assert_called_once()
        
        # Verify same client instance returned
        assert client1 == client2 == mock_kendra

    def test_boto_config_creation(self):
        """Test boto configuration creation."""
        config = self.kendra_service._get_boto_config()

        assert config is not None
        # Verify config has expected timeout settings
        assert hasattr(config, 'read_timeout')
        assert hasattr(config, 'connect_timeout')