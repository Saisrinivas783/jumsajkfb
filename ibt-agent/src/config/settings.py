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

    # Concurrency Configuration
    kendra_max_pool_connections: int = Field(
        default=40,
        gt=0,
        description="Max HTTP connection pool size for the Kendra boto3 client"
    )
    sts_max_pool_connections: int = Field(
        default=20,
        gt=0,
        description="Max HTTP connection pool size for the STS boto3 client (Kendra role assumption)"
    )

    # Kendra Client Timeout/Retry Configuration
    kendra_read_timeout: int = Field(
        default=300,
        gt=0,
        description="Read timeout in seconds for the Kendra boto3 client"
    )
    kendra_connect_timeout: int = Field(
        default=10,
        gt=0,
        description="Connect timeout in seconds for the Kendra boto3 client"
    )
    kendra_max_retries: int = Field(
        default=3,
        gt=0,
        description="Maximum retry attempts for Kendra API calls"
    )

    # STS Client Timeout/Retry Configuration (Kendra role assumption)
    sts_connect_timeout: int = Field(
        default=5,
        gt=0,
        description="Connect timeout in seconds for the STS boto3 client"
    )
    sts_read_timeout: int = Field(
        default=10,
        gt=0,
        description="Read timeout in seconds for the STS boto3 client"
    )
    sts_max_retries: int = Field(
        default=2,
        gt=0,
        description="Maximum retry attempts for STS role-assumption calls"
    )

    # Assumed-role credential refresh buffer
    credentials_refresh_buffer_minutes: int = Field(
        default=5,
        gt=0,
        description="Refresh assumed Kendra role credentials this many minutes before expiration"
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
