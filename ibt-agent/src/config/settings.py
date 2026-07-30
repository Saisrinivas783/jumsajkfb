"""
Application settings using pydantic-settings for type-safe configuration.

Environment variables can be set directly or via .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IBTSettings(BaseSettings):
    """
    IBT Agent configuration settings.

    All settings can be overridden via environment variables.
    Environment variables should be prefixed based on the setting group.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Tool Registry
    registry_path: str = Field(
        default="src/tools/definitions/tools.yaml",
        description="Path to tools registry YAML file"
    )

    # AWS Configuration
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock and other services"
    )
    kendra_index_id: str = Field(
        default="",
        description="AWS Kendra index ID for semantic search"
    )

    # Kendra Role Configuration
    kendra_role_arn: Optional[str] = Field(
        default=None,
        description="IAM role ARN for Kendra access (env: KENDRA_ROLE_ARN)"
    )
    kendra_session_name: str = Field(
        default="ibt-agent-kendra",
        description="Session name for role assumption"
    )
    kendra_role_duration: int = Field(
        default=3600,
        description="Role session duration in seconds (1 hour)"
    )

    # Kendra Search Configuration
    kendra_page_size: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Number of documents to retrieve from Kendra per query"
    )

    # Bedrock LLM Configuration
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

    # Bedrock Role Configuration
    bedrock_session_name: str = Field(
        default="ibt-agent-bedrock",
        description="Session name for Bedrock role assumption"
    )
    bedrock_role_duration: int = Field(
        default=3600,
        description="Bedrock role session duration in seconds (1 hour)"
    )

    # Bedrock Timeout Configuration
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

    # DXAIService Configuration
    dxai_base_url: str = Field(
        default="https://dxai-service.internal",
        description="Base URL for DXAIService"
    )
    dxai_timeout: int = Field(
        default=30,
        description="Timeout in seconds for DXAIService calls"
    )
    dxai_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for DXAIService calls"
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

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


@lru_cache
def get_settings() -> IBTSettings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return IBTSettings()


# Convenience alias for importing
ibt_settings = get_settings()