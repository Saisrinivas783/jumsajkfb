"""
Unit tests for Bedrock assume role functionality in ChatModels.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from src.llm.client import ChatModels
from src.config.settings import OrchestratorSettings


class TestBedrockAssumeRole:
    """Test assume role functionality in ChatModels."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chat_models = ChatModels()

    @patch('src.llm.client.boto3.client')
    def test_assume_bedrock_role_success(self, mock_boto_client):
        """Test successful Bedrock role assumption."""
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
        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
        
        # Test role assumption
        credentials = self.chat_models._assume_bedrock_role()
        
        # Verify STS call - use the actual session name from settings
        mock_sts.assume_role.assert_called_once_with(
            RoleArn='arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role',
            RoleSessionName=self.chat_models.settings.bedrock_session_name,
            DurationSeconds=3600
        )
        
        # Verify returned credentials
        assert credentials['aws_access_key_id'] == 'ASIA123456789'
        assert credentials['aws_secret_access_key'] == 'secret123'
        assert credentials['aws_session_token'] == 'token123'

    def test_assume_bedrock_role_no_arn(self):
        """Test role assumption when no ARN is configured."""
        self.chat_models.settings.bedrock_role_arn = None
        
        with pytest.raises(ValueError, match="BEDROCK_ROLE_ARN is not configured"):
            self.chat_models._assume_bedrock_role()

    @patch('src.llm.client.boto3.client')
    def test_assume_bedrock_role_access_denied(self, mock_boto_client):
        """Test role assumption failure due to access denied."""
        # Mock STS client with access denied error
        mock_sts = Mock()
        mock_boto_client.return_value = mock_sts
        
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_sts.assume_role.side_effect = ClientError(error_response, 'AssumeRole')
        
        # Set up role ARN
        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
        
        # Test role assumption failure
        with pytest.raises(RuntimeError, match="Role assumption failed: AccessDenied"):
            self.chat_models._assume_bedrock_role()

    @patch('src.llm.client.boto3.client')
    def test_assume_bedrock_role_unexpected_error(self, mock_boto_client):
        """Test role assumption failure due to unexpected error."""
        # Mock STS client with unexpected error
        mock_sts = Mock()
        mock_boto_client.return_value = mock_sts
        
        mock_sts.assume_role.side_effect = Exception('Unexpected error')
        
        # Set up role ARN
        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
        
        # Test role assumption failure
        with pytest.raises(RuntimeError, match="Role assumption failed: Unexpected error"):
            self.chat_models._assume_bedrock_role()

    @patch('src.llm.client.boto3.client')
    def test_get_bedrock_client_with_role_assumption(self, mock_boto_client):
        """Test Bedrock client creation with role assumption."""
        # Mock STS client
        mock_sts = Mock()
        mock_bedrock = Mock()
        
        def boto_client_side_effect(service_name, **kwargs):
            if service_name == 'sts':
                return mock_sts
            elif service_name == 'bedrock-runtime':
                return mock_bedrock
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
        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
        
        # Reset client to force re-initialization
        self.chat_models._client = None
        
        # Test client creation
        client = self.chat_models._get_bedrock_client()
        
        # Verify role assumption was called
        mock_sts.assume_role.assert_called_once()
        
        # Verify Bedrock client was created with assumed credentials
        bedrock_calls = [call for call in mock_boto_client.call_args_list if call[1].get('service_name') == 'bedrock-runtime']
        assert len(bedrock_calls) > 0
        bedrock_call = bedrock_calls[0]
        assert bedrock_call[1]['aws_access_key_id'] == 'ASIA123456789'
        assert bedrock_call[1]['aws_secret_access_key'] == 'secret123'
        assert bedrock_call[1]['aws_session_token'] == 'token123'
        
        assert client == mock_bedrock

    @patch('src.llm.client.boto3.client')
    def test_get_bedrock_client_without_role_assumption(self, mock_boto_client):
        """Test Bedrock client creation without role assumption."""
        mock_bedrock = Mock()
        mock_boto_client.return_value = mock_bedrock
        
        # No role ARN configured
        self.chat_models.settings.bedrock_role_arn = None
        
        # Reset client to force re-initialization
        self.chat_models._client = None
        
        # Test client creation
        client = self.chat_models._get_bedrock_client()
        
        # Verify the call was made with correct parameters
        call_args = mock_boto_client.call_args
        assert call_args[1]['service_name'] == 'bedrock-runtime'
        assert call_args[1]['region_name'] == self.chat_models.settings.aws_region
        assert 'config' in call_args[1]
        
        assert client == mock_bedrock

    @patch('src.llm.client.boto3.client')
    def test_client_caching(self, mock_boto_client):
        """Test that Bedrock client is cached after first creation."""
        mock_bedrock = Mock()
        mock_boto_client.return_value = mock_bedrock
        
        # No role ARN configured for simplicity
        self.chat_models.settings.bedrock_role_arn = None
        
        # Reset client
        self.chat_models._client = None
        
        # Get client twice
        client1 = self.chat_models._get_bedrock_client()
        client2 = self.chat_models._get_bedrock_client()
        
        # Verify boto3.client was called only once
        mock_boto_client.assert_called_once()
        
        # Verify same client instance returned
        assert client1 == client2 == mock_bedrock

    @patch('src.llm.client.boto3.client')
    @patch('src.llm.client.ChatBedrockConverse')
    def test_bedrock_model_with_assume_role(self, mock_chat_bedrock, mock_boto_client):
        """Test bedrock_model creation with assume role."""
        # Mock STS and Bedrock clients
        mock_sts = Mock()
        mock_bedrock = Mock()
        mock_model = Mock()
        
        def boto_client_side_effect(service_name, **kwargs):
            if service_name == 'sts':
                return mock_sts
            elif service_name == 'bedrock-runtime':
                return mock_bedrock
            return Mock()
        
        mock_boto_client.side_effect = boto_client_side_effect
        mock_chat_bedrock.return_value = mock_model
        
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
        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
        
        # Reset client
        self.chat_models._client = None
        
        # Test model creation
        model = self.chat_models.bedrock_model()
        
        # Verify role assumption occurred
        mock_sts.assume_role.assert_called_once()
        
        # Verify ChatBedrockConverse was called with the assumed role client
        mock_chat_bedrock.assert_called_once()
        call_kwargs = mock_chat_bedrock.call_args[1]
        assert call_kwargs['client'] == mock_bedrock
        
        assert model == mock_model

    def test_settings_configuration(self):
        """Test that settings are properly configured for role assumption."""
        settings = OrchestratorSettings()
        
        # Test actual values from settings (may be defaults or environment overrides)
        # The test should work with whatever the settings actually contain
        assert settings.bedrock_role_duration == 3600
        # Session name can be either default or environment override
        assert settings.bedrock_session_name in ["orchestrator-agent-bedrock", "orchestrator-agent-bedrock-session"]
        
        # Test with environment variables
        with patch.dict('os.environ', {
            'BEDROCK_ROLE_ARN': 'arn:aws:iam::123456789:role/test-role',
            'BEDROCK_SESSION_NAME': 'custom-session',
            'BEDROCK_ROLE_DURATION': '7200'
        }):
            settings = OrchestratorSettings()
            assert settings.bedrock_role_arn == 'arn:aws:iam::123456789:role/test-role'
            assert settings.bedrock_session_name == 'custom-session'
            assert settings.bedrock_role_duration == 7200

    def test_boto_config_creation(self):
        """Test boto configuration creation."""
        config = self.chat_models._get_boto_config()
        
        assert config is not None
        # Verify config has expected timeout and retry settings
        assert hasattr(config, 'read_timeout')
        assert hasattr(config, 'connect_timeout')

    @patch('src.llm.client.boto3.client')
    def test_credentials_storage(self, mock_boto_client):
        """Test that assumed credentials are stored properly."""
        # Mock STS client
        mock_sts = Mock()
        mock_bedrock = Mock()
        
        def boto_client_side_effect(service_name, **kwargs):
            if service_name == 'sts':
                return mock_sts
            elif service_name == 'bedrock-runtime':
                return mock_bedrock
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
        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
        
        # Reset client and credentials
        self.chat_models._client = None
        self.chat_models._assumed_credentials = None
        
        # Test client creation
        self.chat_models._get_bedrock_client()
        
        # Verify credentials are stored
        assert self.chat_models._assumed_credentials is not None
        assert self.chat_models._assumed_credentials['aws_access_key_id'] == 'ASIA123456789'
        assert self.chat_models._assumed_credentials['aws_secret_access_key'] == 'secret123'
        assert self.chat_models._assumed_credentials['aws_session_token'] == 'token123'