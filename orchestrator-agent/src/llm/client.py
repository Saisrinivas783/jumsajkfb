from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import orchestrator_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChatModels:
    """Factory for AWS Bedrock chat models with timeout/retry configuration and role assumption."""

    CREDENTIALS_REFRESH_BUFFER = timedelta(minutes=5)

    def __init__(self):
        self.settings = orchestrator_settings
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None

    def _get_boto_config(self) -> Config:
        return Config(
            read_timeout=self.settings.bedrock_read_timeout,
            connect_timeout=self.settings.bedrock_connect_timeout,
            retries={"max_attempts": self.settings.bedrock_max_retries, "mode": "adaptive"},
        )

    def _credentials_expired(self) -> bool:
        """Check if assumed credentials are expired or about to expire."""
        if self._credentials_expiration is None:
            return True
        return datetime.now(timezone.utc) >= self._credentials_expiration - self.CREDENTIALS_REFRESH_BUFFER

    def _assume_bedrock_role(self) -> dict:
        """Assume the Bedrock role and return temporary credentials."""
        if not self.settings.bedrock_role_arn:
            raise ValueError("BEDROCK_ROLE_ARN is not configured")

        try:
            logger.info(f"Assuming Bedrock role: {self.settings.bedrock_role_arn}")
            
            sts_client = boto3.client('sts', region_name=self.settings.aws_region)
            
            response = sts_client.assume_role(
                RoleArn=self.settings.bedrock_role_arn,
                RoleSessionName=self.settings.bedrock_session_name,
                DurationSeconds=self.settings.bedrock_role_duration
            )
            
            credentials = response['Credentials']
            self._credentials_expiration = credentials['Expiration']
            logger.info(f"Successfully assumed Bedrock role. Session expires at: {credentials['Expiration']}")
            
            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to assume Bedrock role {self.settings.bedrock_role_arn}: {error_code}")
            raise RuntimeError(f"Role assumption failed: {error_code}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RuntimeError(f"Role assumption failed: {str(e)}") from e

    def _get_bedrock_client(self) -> boto3.client:
        if self._client is not None and (not self.settings.bedrock_role_arn or not self._credentials_expired()):
            return self._client

        if self._client is not None and self.settings.bedrock_role_arn and self._credentials_expired():
            logger.info("Assumed role credentials expired or expiring soon, refreshing...")

        logger.info(f"Initializing Bedrock client: region={self.settings.aws_region}")
        
        # Check if role assumption is configured
        if self.settings.bedrock_role_arn:
            logger.info("Using role assumption for Bedrock access")
            credentials = self._assume_bedrock_role()
            self._assumed_credentials = credentials
            
            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.settings.aws_region,
                config=self._get_boto_config(),
                **credentials
            )
        else:
            logger.info("Using default AWS credentials for Bedrock access")
            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.settings.aws_region,
                config=self._get_boto_config(),
            )
        
        return self._client

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


_chat_models: Optional[ChatModels] = None


def get_chat_models() -> ChatModels:
    """Return the singleton ChatModels instance."""
    global _chat_models
    if _chat_models is None:
        _chat_models = ChatModels()
    return _chat_models
