"""
Application settings using pydantic-settings for type-safe configuration.

Environment variables can be set directly or via .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorSettings(BaseSettings):
    """
    Orchestrator Agent configuration settings.

    All settings can be overridden via environment variables.
    Environment variables should be prefixed based on the setting group.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AWS Configuration
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock and other services"
    )

    # Bedrock Role Configuration (for assume role)
    bedrock_role_arn: Optional[str] = Field(
        default=None,
        description="IAM role ARN for Bedrock access (env: BEDROCK_ROLE_ARN)"
    )
    bedrock_session_name: str = Field(
        default="orchestrator-agent-bedrock",
        description="Session name for role assumption"
    )
    bedrock_role_duration: int = Field(
        default=3600,
        description="Role session duration in seconds (1 hour)"
    )

    # Bedrock LLM Configuration
    # bedrock_model_id: str = Field(
    #     default="us.anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (commented out)
    #     description="Default Bedrock model ID"
    # )
    bedrock_model_id: str = Field(
        default="us.meta.llama3-3-70b-instruct-v1:0",
        description="Default Bedrock model ID"
    )
    bedrock_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.0 = deterministic, 1.0 = creative)"
    )
    bedrock_max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum tokens in LLM response"
    )

    # Timeout Configuration
    bedrock_read_timeout: int = Field(
        default=300,
        description="Read timeout in seconds for Bedrock API calls"
    )
    bedrock_connect_timeout: int = Field(
        default=10,
        description="Connection timeout in seconds for Bedrock API calls"
    )
    bedrock_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for Bedrock API calls"
    )
    bedrock_max_pool_connections: int = Field(
        default=50,
        gt=0,
        description="Maximum pooled HTTP connections for the Bedrock boto client"
    )

    # Extended Thinking Configuration (for Claude 3.5+ models)
    extended_thinking_enabled: bool = Field(
        default=False,
        description="Enable extended thinking for complex reasoning tasks"
    )
    extended_thinking_max_tokens: int = Field(
        default=16000,
        description="Maximum tokens when extended thinking is enabled"
    )
    extended_thinking_budget_tokens: int = Field(
        default=10000,
        description="Token budget for thinking process"
    )

    # Guardrails Configuration
    aws_bedrock_guardrail_id: Optional[str] = Field(
        default=None,
        description="AWS Bedrock Guardrail identifier (env: AWS_BEDROCK_GUARDRAIL_ID)"
    )
    bedrock_guardrail_version: str = Field(
        default="1",
        description="AWS Bedrock Guardrail version"
    )

    # Guard Rails Configuration
    confidence_threshold_high: float = Field(
        default=7.0,
        description="Minimum confidence to execute tool"
    )
    confidence_threshold_low: float = Field(
        default=5.0,
        description="Minimum confidence for clarification (below this = fallback)"
    )

    # Tool Execution Configuration
    tool_timeout: int = Field(
        default=30,
        description="Timeout in seconds for tool API calls"
    )
    tool_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for tool API calls"
    )

    # Session Configuration
    session_ttl_seconds: int = Field(
        default=1800,
        description="Session time-to-live in seconds (default: 30 minutes)"
    )
    max_conversation_history: int = Field(
        default=20,
        description="Maximum number of messages to retain in conversation history"
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Registry Configuration
    registry_path: str = Field(
        default="src/tools/definitions/tools.yaml",
        description="Path to the tool registry YAML file"
    )

    # Query Configuration
    use_reformulated_query: bool = Field(
        default=True,
        description="If True, send the LLM-reformulated query to tools; if False, send the original user query"
    )



@lru_cache
def get_settings() -> OrchestratorSettings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return OrchestratorSettings()


# Convenience alias for importing
orchestrator_settings = get_settings()
