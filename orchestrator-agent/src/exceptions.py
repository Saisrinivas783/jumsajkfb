"""Exception hierarchy for the Orchestrator Agent."""

from typing import Any


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ToolUnavailableError(OrchestratorError):
    """Raised when a tool is down or unreachable."""

    def __init__(self, tool_name: str, message: str = "Tool unavailable") -> None:
        super().__init__(f"{message}: {tool_name}", {"tool_name": tool_name})


class ToolTimeoutError(OrchestratorError):
    """Raised when a tool response times out."""

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            {"tool_name": tool_name, "timeout": timeout_seconds},
        )



class LLMFailureError(OrchestratorError):
    """Raised when the LLM service is unavailable or returns an error."""

    def __init__(self, message: str = "LLM failure") -> None:
        super().__init__(message)


class GuardrailError(OrchestratorError):
    """Raised when the AWS Bedrock Guardrail service is unavailable or returns an error."""

    def __init__(self, message: str = "Guardrail check failed") -> None:
        super().__init__(message)


class CredentialsError(OrchestratorError):
    """Raised when AWS STS role assumption fails after exhausting retries."""

    def __init__(self, message: str = "Credentials unavailable") -> None:
        super().__init__(message)


class ConfigurationError(OrchestratorError):
    """Raised for invalid configuration."""

    def __init__(self, message: str, config_key: str | None = None) -> None:
        super().__init__(message, {"config_key": config_key})


class ToolRegistryError(ConfigurationError):
    """Raised for tool registry loading or parsing errors."""
