import threading
from functools import lru_cache
from typing import Optional

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from src.aws.assume_role import AssumedRoleClientFactory, CredentialRefreshWorker
from src.config.settings import orchestrator_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChatModels:
    """Factory for AWS Bedrock chat models with timeout/retry configuration and role assumption."""

    def __init__(self):
        self.settings = orchestrator_settings
        self._client: Optional[boto3.client] = None
        self._client_lock = threading.Lock()
        self._assume_role_factory: Optional[AssumedRoleClientFactory] = None
        self._refresh_worker: Optional[CredentialRefreshWorker] = None

    def _get_boto_config(self) -> Config:
        return Config(
            read_timeout=self.settings.bedrock_read_timeout,
            connect_timeout=self.settings.bedrock_connect_timeout,
            max_pool_connections=self.settings.bedrock_max_pool_connections,
            retries={"max_attempts": self.settings.bedrock_max_retries, "mode": "adaptive"},
        )

    def _get_assume_role_factory(self) -> AssumedRoleClientFactory:
        """Return the Bedrock role factory with botocore-managed refresh."""
        if not self.settings.bedrock_role_arn:
            raise ValueError("BEDROCK_ROLE_ARN is not configured")

        if self._assume_role_factory is None:
            self._assume_role_factory = AssumedRoleClientFactory(
                role_arn=self.settings.bedrock_role_arn,
                session_name=self.settings.bedrock_session_name,
                duration_seconds=self.settings.bedrock_role_duration,
                region_name=self.settings.aws_region,
                method="bedrock-assume-role",
            )
        return self._assume_role_factory

    def _get_bedrock_client(self) -> boto3.client:
        """Get the Bedrock client, built once and cached.

        Credentials refresh themselves in place via botocore's
        ``RefreshableCredentials`` (see ``AssumedRoleClientFactory``), so the
        client object never needs to be rebuilt once constructed.
        """
        if self._client is not None:
            return self._client

        with self._client_lock:
            if self._client is not None:
                return self._client

            logger.info(f"Initializing Bedrock client: region={self.settings.aws_region}")

            if self.settings.bedrock_role_arn:
                logger.info("Using refreshable role assumption for Bedrock access")
                self._client = self._get_assume_role_factory().client(
                    "bedrock-runtime", config=self._get_boto_config()
                )
            else:
                logger.info("Using default AWS credentials for Bedrock access")
                self._client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.settings.aws_region,
                    config=self._get_boto_config(),
                )

        return self._client

    def warm_credentials(self) -> None:
        """Build the Bedrock client once, synchronously. Call at startup, off the event loop."""
        self._get_bedrock_client()

    def start_credential_refresh(self) -> None:
        """Start the background worker that proactively refreshes assumed-role credentials."""
        if not self.settings.bedrock_role_arn:
            return
        if self._refresh_worker is None:
            self._refresh_worker = CredentialRefreshWorker(
                self._get_assume_role_factory(), name="bedrock-credential-refresh"
            )
        self._refresh_worker.start()

    def stop_credential_refresh(self) -> None:
        """Stop the background credential refresh worker."""
        if self._refresh_worker is not None:
            self._refresh_worker.stop()

    def bedrock_model(
        self,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatBedrockConverse:
        """Return a standard ChatBedrockConverse model."""
        resolved_model = model_id or self.settings.bedrock_model_id
        logger.info(f"Creating LLM: {resolved_model}")
        return ChatBedrockConverse(
            client=self._get_bedrock_client(),
            model=resolved_model,
            region_name=self.settings.aws_region,
            temperature=temperature if temperature is not None else self.settings.bedrock_temperature,
            max_tokens=max_tokens or self.settings.bedrock_max_tokens,
            **kwargs,
        )

    def bedrock_model_with_extended_thinking(
        self,
        model_id: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatBedrockConverse:
        """Return a ChatBedrockConverse model with extended thinking enabled (Claude 3.5+)."""
        resolved_model = model_id or self.settings.bedrock_model_id
        thinking_budget = budget_tokens or self.settings.extended_thinking_budget_tokens
        response_max_tokens = max_tokens or self.settings.extended_thinking_max_tokens
        logger.info(f"Creating LLM (extended thinking): {resolved_model}, budget_tokens={thinking_budget}")
        return ChatBedrockConverse(
            client=self._get_bedrock_client(),
            model=resolved_model,
            region_name=self.settings.aws_region,
            temperature=1,  # Required for extended thinking
            max_tokens=response_max_tokens,
            additional_model_request_fields={
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            },
            **kwargs,
        )

    def bedrock_model_with_guardrails(
        self,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatBedrockConverse:
        """Return a ChatBedrockConverse model with AWS Bedrock Guardrails enabled."""
        guardrail_id = self.settings.aws_bedrock_guardrail_id
        if not guardrail_id:
            raise ValueError("AWS_BEDROCK_GUARDRAIL_ID environment variable is not set")
        resolved_model = model_id or self.settings.bedrock_model_id
        logger.info(f"Creating LLM with guardrails: {resolved_model}, guardrail={guardrail_id}")
        return ChatBedrockConverse(
            client=self._get_bedrock_client(),
            model=resolved_model,
            region_name=self.settings.aws_region,
            temperature=temperature if temperature is not None else self.settings.bedrock_temperature,
            max_tokens=max_tokens or self.settings.bedrock_max_tokens,
            guardrail_config={
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": self.settings.bedrock_guardrail_version,
                "trace": "enabled",
            },
            **kwargs,
        )

    def apply_guardrail(self, text: str, source: str = "INPUT") -> dict:
        """Evaluate content against AWS Bedrock Guardrails without an LLM call."""
        guardrail_id = self.settings.aws_bedrock_guardrail_id
        if not guardrail_id:
            raise ValueError("AWS_BEDROCK_GUARDRAIL_ID environment variable is not set")
        client = self._get_bedrock_client()
        return client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=self.settings.bedrock_guardrail_version,
            source=source,
            content=[{"text": {"text": text}}],
        )

    def get_model(self, model_id: Optional[str] = None, **kwargs) -> BaseChatModel:
        """Return the appropriate model based on settings (standard or extended thinking)."""
        if self.settings.extended_thinking_enabled:
            return self.bedrock_model_with_extended_thinking(model_id=model_id, **kwargs)
        return self.bedrock_model(model_id=model_id, **kwargs)


@lru_cache(maxsize=1)
def get_chat_models() -> ChatModels:
    """Return the singleton ChatModels instance."""
    return ChatModels()
