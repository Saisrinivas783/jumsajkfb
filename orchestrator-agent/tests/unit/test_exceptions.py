"""Unit tests for exception hierarchy."""

import pytest
from src.exceptions import (
    OrchestratorError,
    ToolUnavailableError,
    ToolTimeoutError,
    LLMFailureError,
    GuardrailError,
    ConfigurationError,
    ToolRegistryError,
)


class TestOrchestratorError:
    def test_message_stored(self):
        err = OrchestratorError("Something went wrong")
        assert err.message == "Something went wrong"
        assert str(err) == "Something went wrong"

    def test_details_default_empty(self):
        err = OrchestratorError("msg")
        assert err.details == {}

    def test_details_stored(self):
        err = OrchestratorError("msg", details={"key": "val"})
        assert err.details["key"] == "val"


class TestToolUnavailableError:
    def test_message_includes_tool_name(self):
        err = ToolUnavailableError("IBTAgent")
        assert "IBTAgent" in str(err)

    def test_custom_message(self):
        err = ToolUnavailableError("IBTAgent", "Service is down")
        assert "Service is down" in str(err)

    def test_is_orchestrator_error(self):
        assert isinstance(ToolUnavailableError("X"), OrchestratorError)


class TestToolTimeoutError:
    def test_message_includes_tool_name_and_timeout(self):
        err = ToolTimeoutError("IBTAgent", 30.0)
        assert "IBTAgent" in str(err)
        assert "30" in str(err)

    def test_details_stored(self):
        err = ToolTimeoutError("IBTAgent", 15.0)
        assert err.details["tool_name"] == "IBTAgent"
        assert err.details["timeout"] == 15.0



class TestLLMFailureError:
    def test_default_message(self):
        err = LLMFailureError()
        assert "LLM failure" in str(err)

    def test_custom_message(self):
        err = LLMFailureError("Bedrock unreachable")
        assert "Bedrock unreachable" in str(err)


class TestGuardrailError:
    def test_default_message(self):
        err = GuardrailError()
        assert "Guardrail check failed" in str(err)

    def test_custom_message(self):
        err = GuardrailError("Bedrock guardrail unreachable")
        assert "Bedrock guardrail unreachable" in str(err)

    def test_is_orchestrator_error(self):
        assert isinstance(GuardrailError(), OrchestratorError)


class TestConfigurationError:
    def test_message_stored(self):
        err = ConfigurationError("Bad config")
        assert "Bad config" in str(err)

    def test_config_key_stored(self):
        err = ConfigurationError("Bad config", config_key="AWS_REGION")
        assert err.details["config_key"] == "AWS_REGION"


class TestToolRegistryError:
    def test_is_configuration_error(self):
        err = ToolRegistryError("YAML parse error")
        assert isinstance(err, ConfigurationError)

    def test_message_stored(self):
        err = ToolRegistryError("Missing tools key")
        assert "Missing tools key" in str(err)


