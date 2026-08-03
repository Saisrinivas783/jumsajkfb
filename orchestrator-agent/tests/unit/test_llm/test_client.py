"""Unit tests for LLM client (ChatModels)."""

import pytest
from unittest.mock import MagicMock, patch
from botocore.config import Config


class TestChatModelsInit:
    """Tests for ChatModels initialization."""

    def test_init_sets_settings(self):
        from src.llm.client import ChatModels
        cm = ChatModels()
        assert cm.settings is not None
        assert cm._client is None

    def test_get_boto_config_returns_config(self):
        from src.llm.client import ChatModels
        cm = ChatModels()
        config = cm._get_boto_config()
        assert isinstance(config, Config)

    @patch("src.llm.client.boto3.client")
    def test_get_boto_config_uses_bedrock_max_pool_connections(self, mock_boto):
        from src.llm.client import ChatModels
        cm = ChatModels()
        config = cm._get_boto_config()
        assert config.max_pool_connections == cm.settings.bedrock_max_pool_connections


class TestGetBedrockClient:
    """Tests for _get_bedrock_client method."""

    @patch("src.llm.client.boto3.client")
    def test_creates_client_on_first_call(self, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()

        cm = ChatModels()
        # Disable role assumption for this test to avoid STS calls
        cm.settings.bedrock_role_arn = None
        client = cm._get_bedrock_client()

        mock_boto.assert_called_once()
        assert client is not None

    @patch("src.llm.client.boto3.client")
    def test_returns_cached_client_on_second_call(self, mock_boto):
        """Line 71: return self._client (cached path)."""
        from src.llm.client import ChatModels
        mock_client = MagicMock()
        mock_boto.return_value = mock_client

        cm = ChatModels()
        # Disable role assumption for this test to avoid STS calls
        cm.settings.bedrock_role_arn = None
        first = cm._get_bedrock_client()
        second = cm._get_bedrock_client()

        assert mock_boto.call_count == 1
        assert first is second

    @patch("src.llm.client.boto3.client")
    def test_client_configured_with_correct_service(self, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()

        cm = ChatModels()
        cm._get_bedrock_client()

        call_kwargs = mock_boto.call_args
        assert call_kwargs[1]["service_name"] == "bedrock-runtime"


class TestBedrockModel:
    """Tests for bedrock_model method."""

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_bedrock_model_default_params(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        result = cm.bedrock_model()

        mock_cbc.assert_called_once()
        assert result is not None

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_bedrock_model_custom_model_id(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.bedrock_model(model_id="custom-model")

        kwargs = mock_cbc.call_args[1]
        assert kwargs["model"] == "custom-model"

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_bedrock_model_custom_temperature(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.bedrock_model(temperature=0.5)

        kwargs = mock_cbc.call_args[1]
        assert kwargs["temperature"] == 0.5

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_bedrock_model_custom_max_tokens(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.bedrock_model(max_tokens=512)

        kwargs = mock_cbc.call_args[1]
        assert kwargs["max_tokens"] == 512


class TestBedrockModelWithExtendedThinking:
    """Tests for bedrock_model_with_extended_thinking method (lines 156-182)."""

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_extended_thinking_uses_temperature_1(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.bedrock_model_with_extended_thinking()

        kwargs = mock_cbc.call_args[1]
        assert kwargs["temperature"] == 1

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_extended_thinking_includes_thinking_config(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.bedrock_model_with_extended_thinking()

        kwargs = mock_cbc.call_args[1]
        thinking = kwargs["additional_model_request_fields"]["thinking"]
        assert thinking["type"] == "enabled"
        assert "budget_tokens" in thinking

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_extended_thinking_custom_budget(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.bedrock_model_with_extended_thinking(budget_tokens=3000, max_tokens=6000)

        kwargs = mock_cbc.call_args[1]
        assert kwargs["max_tokens"] == 6000
        assert kwargs["additional_model_request_fields"]["thinking"]["budget_tokens"] == 3000

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_extended_thinking_custom_model_id(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.bedrock_model_with_extended_thinking(model_id="claude-3-5-sonnet")

        kwargs = mock_cbc.call_args[1]
        assert kwargs["model"] == "claude-3-5-sonnet"


class TestGetModel:
    """Tests for get_model method."""

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_get_model_standard_when_extended_thinking_disabled(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.extended_thinking_enabled = False
        cm.get_model()

        kwargs = mock_cbc.call_args[1]
        # Standard model should NOT use temperature=1 (extended thinking)
        assert kwargs.get("temperature") != 1

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_get_model_extended_thinking_when_enabled(self, mock_cbc, mock_boto):
        """Line 203: extended thinking branch of get_model()."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.extended_thinking_enabled = True
        cm.get_model()

        kwargs = mock_cbc.call_args[1]
        assert kwargs["temperature"] == 1  # Extended thinking requires temp=1


class TestBedrockModelWithGuardrails:
    """Tests for bedrock_model_with_guardrails method."""

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_raises_when_guardrail_id_not_set(self, mock_cbc, mock_boto):
        """Raises ValueError when AWS_BEDROCK_GUARDRAIL_ID is not configured."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = None

        with pytest.raises(ValueError, match="AWS_BEDROCK_GUARDRAIL_ID"):
            cm.bedrock_model_with_guardrails()

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_guardrail_config_passed_to_model(self, mock_cbc, mock_boto):
        """guardrail_config dict is passed to ChatBedrockConverse."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        cm.bedrock_model_with_guardrails()

        kwargs = mock_cbc.call_args[1]
        assert "guardrail_config" in kwargs

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_guardrail_config_contains_identifier(self, mock_cbc, mock_boto):
        """guardrailIdentifier in config matches the setting."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-xyz"
        cm.bedrock_model_with_guardrails()

        kwargs = mock_cbc.call_args[1]
        assert kwargs["guardrail_config"]["guardrailIdentifier"] == "guardrail-xyz"

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_guardrail_config_contains_version(self, mock_cbc, mock_boto):
        """guardrailVersion in config matches the setting."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        cm.settings.bedrock_guardrail_version = "2"
        cm.bedrock_model_with_guardrails()

        kwargs = mock_cbc.call_args[1]
        assert kwargs["guardrail_config"]["guardrailVersion"] == "2"

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_guardrail_config_trace_enabled(self, mock_cbc, mock_boto):
        """trace is set to 'enabled' in guardrail config."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        cm.bedrock_model_with_guardrails()

        kwargs = mock_cbc.call_args[1]
        assert kwargs["guardrail_config"]["trace"] == "enabled"

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_uses_same_region_as_bedrock_model(self, mock_cbc, mock_boto):
        """Uses the same region_name as the standard bedrock_model."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        cm.bedrock_model_with_guardrails()

        kwargs = mock_cbc.call_args[1]
        assert kwargs["region_name"] == cm.settings.aws_region

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_custom_model_id_is_used(self, mock_cbc, mock_boto):
        """Custom model_id overrides the default."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        cm.bedrock_model_with_guardrails(model_id="custom-model-id")

        kwargs = mock_cbc.call_args[1]
        assert kwargs["model"] == "custom-model-id"

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_default_model_id_used_when_not_specified(self, mock_cbc, mock_boto):
        """Default model ID from settings is used when not overridden."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        cm.bedrock_model_with_guardrails()

        kwargs = mock_cbc.call_args[1]
        assert kwargs["model"] == cm.settings.bedrock_model_id

    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    def test_custom_temperature_is_used(self, mock_cbc, mock_boto):
        """Custom temperature overrides the settings default."""
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        cm.bedrock_model_with_guardrails(temperature=0.7)

        kwargs = mock_cbc.call_args[1]
        assert kwargs["temperature"] == 0.7


class TestAssumeBedrockRole:
    """Tests for _assume_bedrock_role STS client configuration."""

    @patch("src.llm.client.boto3.client")
    def test_sts_client_uses_sts_max_pool_connections(self, mock_boto):
        from src.llm.client import ChatModels

        mock_sts = MagicMock()
        mock_boto.return_value = mock_sts
        mock_sts.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': '2024-01-01T12:00:00Z'
            }
        }

        cm = ChatModels()
        cm.settings.bedrock_role_arn = 'arn:aws:iam::054940911799:role/orchestrator-bedrock-role'

        cm._assume_bedrock_role()

        sts_call = mock_boto.call_args
        assert sts_call[0][0] == 'sts'
        assert 'config' in sts_call[1]
        assert sts_call[1]['config'].max_pool_connections == cm.settings.sts_max_pool_connections

    @patch("src.llm.client.boto3.client")
    def test_concurrent_get_bedrock_client_calls_assume_role_once(self, mock_boto):
        """Test that concurrent calls to _get_bedrock_client with expired credentials
        only trigger one assume_role call, not one per caller."""
        from concurrent.futures import ThreadPoolExecutor
        import time as time_module
        from src.llm.client import ChatModels

        mock_sts = MagicMock()
        mock_bedrock = MagicMock()

        def boto_client_side_effect(service_name=None, **kwargs):
            if service_name == 'sts':
                time_module.sleep(0.05)  # widen the race window
                return mock_sts
            return mock_bedrock

        mock_boto.side_effect = boto_client_side_effect
        from datetime import datetime, timezone, timedelta
        mock_sts.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': datetime.now(timezone.utc) + timedelta(hours=1)
            }
        }

        cm = ChatModels()
        cm.settings.bedrock_role_arn = 'arn:aws:iam::054940911799:role/orchestrator-bedrock-role'

        with ThreadPoolExecutor(max_workers=10) as pool:
            clients = list(pool.map(lambda _: cm._get_bedrock_client(), range(10)))

        assert mock_sts.assume_role.call_count == 1
        assert all(c is mock_bedrock for c in clients)


class TestGetChatModels:
    """Tests for get_chat_models singleton."""

    def test_get_chat_models_returns_same_instance(self):
        import src.llm.client as mod
        mod._chat_models = None

        from src.llm.client import get_chat_models, ChatModels
        a = get_chat_models()
        b = get_chat_models()

        assert a is b
        assert isinstance(a, ChatModels)
        mod._chat_models = None

    def test_get_chat_models_creates_on_first_call(self):
        import src.llm.client as mod
        mod._chat_models = None

        from src.llm.client import get_chat_models, ChatModels
        result = get_chat_models()

        assert isinstance(result, ChatModels)
        mod._chat_models = None


