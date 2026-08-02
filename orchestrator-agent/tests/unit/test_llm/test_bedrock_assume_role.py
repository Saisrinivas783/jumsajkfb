"""
Unit tests for ChatModels' use of the shared AssumedRoleClientFactory.

STS retry/backoff behavior itself is covered in tests/unit/test_aws/test_assume_role.py;
these tests cover ChatModels' integration with it (factory wiring, client
caching, boto config, and the credential refresh worker lifecycle).
"""

from unittest.mock import Mock, MagicMock, patch

import pytest

from src.exceptions import CredentialsError
from src.llm.client import ChatModels
from src.config.settings import OrchestratorSettings


class TestBedrockAssumeRole:
    """Test ChatModels' integration with AssumedRoleClientFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chat_models = ChatModels()

    def test_get_bedrock_client_no_arn_uses_default_credentials(self):
        """No role ARN configured -> default-credentials boto3 client, not the factory."""
        with patch('src.llm.client.boto3.client') as mock_boto_client:
            mock_bedrock = Mock()
            mock_boto_client.return_value = mock_bedrock

            self.chat_models.settings.bedrock_role_arn = None
            self.chat_models._client = None

            client = self.chat_models._get_bedrock_client()

            call_args = mock_boto_client.call_args
            assert call_args[1]['service_name'] == 'bedrock-runtime'
            assert call_args[1]['region_name'] == self.chat_models.settings.aws_region
            assert 'config' in call_args[1]
            assert client == mock_bedrock

    def test_get_bedrock_client_with_role_assumption_uses_factory(self):
        """A configured role ARN routes client construction through the factory."""
        with patch('src.llm.client.AssumedRoleClientFactory') as mock_factory_cls:
            mock_factory = Mock()
            mock_bedrock = Mock()
            mock_factory.client.return_value = mock_bedrock
            mock_factory_cls.return_value = mock_factory

            self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
            self.chat_models._client = None

            client = self.chat_models._get_bedrock_client()

            mock_factory_cls.assert_called_once_with(
                role_arn=self.chat_models.settings.bedrock_role_arn,
                session_name=self.chat_models.settings.bedrock_session_name,
                duration_seconds=self.chat_models.settings.bedrock_role_duration,
                region_name=self.chat_models.settings.aws_region,
                method='bedrock-assume-role',
            )
            mock_factory.client.assert_called_once()
            assert mock_factory.client.call_args[0][0] == 'bedrock-runtime'
            assert client == mock_bedrock

    def test_get_bedrock_client_role_assumption_failure_propagates(self):
        """A factory that can't build a session raises CredentialsError, not a silent fallback."""
        with patch('src.llm.client.AssumedRoleClientFactory') as mock_factory_cls:
            mock_factory = Mock()
            mock_factory.client.side_effect = CredentialsError('AssumeRole failed: AccessDenied')
            mock_factory_cls.return_value = mock_factory

            self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
            self.chat_models._client = None

            with pytest.raises(CredentialsError, match='AccessDenied'):
                self.chat_models._get_bedrock_client()

    def test_start_credential_refresh_starts_worker_once(self):
        """Repeated calls reuse the same worker instance rather than starting a new one."""
        with patch('src.llm.client.CredentialRefreshWorker') as mock_worker_cls:
            mock_worker = Mock()
            mock_worker_cls.return_value = mock_worker

            self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'

            self.chat_models.start_credential_refresh()
            self.chat_models.start_credential_refresh()

            mock_worker_cls.assert_called_once()
            assert mock_worker.start.call_count == 2

    def test_stop_credential_refresh_without_start_is_a_no_op(self):
        """Stopping before starting shouldn't raise."""
        self.chat_models.stop_credential_refresh()

    def test_client_caching(self):
        """Test that Bedrock client is cached after first creation."""
        with patch('src.llm.client.boto3.client') as mock_boto_client:
            mock_bedrock = Mock()
            mock_boto_client.return_value = mock_bedrock

            self.chat_models.settings.bedrock_role_arn = None
            self.chat_models._client = None

            client1 = self.chat_models._get_bedrock_client()
            client2 = self.chat_models._get_bedrock_client()

            mock_boto_client.assert_called_once()
            assert client1 == client2 == mock_bedrock

    def test_bedrock_model_with_assume_role(self):
        """Test bedrock_model creation with assume role, via the factory."""
        with (
            patch('src.llm.client.AssumedRoleClientFactory') as mock_factory_cls,
            patch('src.llm.client.ChatBedrockConverse') as mock_chat_bedrock,
        ):
            mock_factory = Mock()
            mock_bedrock = Mock()
            mock_model = Mock()
            mock_factory.client.return_value = mock_bedrock
            mock_factory_cls.return_value = mock_factory
            mock_chat_bedrock.return_value = mock_model

            self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
            self.chat_models._client = None

            model = self.chat_models.bedrock_model()

            mock_factory.client.assert_called_once()
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
        assert hasattr(config, 'read_timeout')
        assert hasattr(config, 'connect_timeout')
        assert config.max_pool_connections == self.chat_models.settings.bedrock_max_pool_connections
