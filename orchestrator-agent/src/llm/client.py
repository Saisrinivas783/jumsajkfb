import random
import threading
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import orchestrator_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RoleAssumptionError(RuntimeError):
    """Raised when STS role assumption fails."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class ChatModels:
    """Factory for AWS Bedrock chat models with timeout/retry configuration and role assumption."""

    CREDENTIALS_REFRESH_BUFFER = timedelta(minutes=5)
    BACKGROUND_REFRESH_LEAD_TIME = timedelta(seconds=60)
    BACKGROUND_REFRESH_JITTER = timedelta(seconds=60)
    BACKGROUND_REFRESH_RETRY_DELAY = timedelta(seconds=30)
    TRANSIENT_STS_ERROR_CODES = {
        "InternalError",
        "InternalFailure",
        "PriorRequestNotComplete",
        "RequestLimitExceeded",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
    }

    def __init__(self):
        self.settings = orchestrator_settings
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None
        self._refresh_lock = threading.Lock()
        self._refresh_thread_lock = threading.Lock()
        self._refresh_stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None

    def _get_boto_config(self) -> Config:
        return Config(
            read_timeout=self.settings.bedrock_read_timeout,
            connect_timeout=self.settings.bedrock_connect_timeout,
            retries={"max_attempts": self.settings.bedrock_max_retries, "mode": "adaptive"},
        )

    def _get_sts_config(self) -> Config:
        return Config(retries={"max_attempts": 3, "mode": "adaptive"})

    def _normalize_expiration(self, expiration) -> datetime:
        """Normalize boto/test credential expirations to timezone-aware UTC datetimes."""
        if isinstance(expiration, datetime):
            normalized = expiration
        elif isinstance(expiration, str):
            normalized = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
        else:
            raise ValueError(f"Unsupported credential expiration type: {type(expiration)!r}")

        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc)

    def _credentials_expired(self) -> bool:
        """Check if assumed credentials are expired or about to expire."""
        if self._credentials_expiration is None:
            return True
        return datetime.now(timezone.utc) >= self._credentials_expiration - self.CREDENTIALS_REFRESH_BUFFER

    def _credentials_still_valid(self) -> bool:
        """Check whether the cached assumed-role credentials can still serve traffic."""
        return (
            self._credentials_expiration is not None
            and datetime.now(timezone.utc) < self._credentials_expiration
        )

    def _background_refresh_delay_seconds(self) -> float:
        """Return delay until the next proactive refresh, jittered per process."""
        jitter_seconds = random.uniform(0, self.BACKGROUND_REFRESH_JITTER.total_seconds())

        if self._credentials_expiration is None:
            return jitter_seconds

        refresh_at = (
            self._credentials_expiration
            - self.CREDENTIALS_REFRESH_BUFFER
            - self.BACKGROUND_REFRESH_LEAD_TIME
            - timedelta(seconds=jitter_seconds)
        )
        return max(0.0, (refresh_at - datetime.now(timezone.utc)).total_seconds())

    def _background_retry_delay_seconds(self) -> float:
        """Return a jittered retry delay after a failed proactive refresh."""
        return (
            self.BACKGROUND_REFRESH_RETRY_DELAY.total_seconds()
            + random.uniform(0, self.BACKGROUND_REFRESH_JITTER.total_seconds())
        )

    def _is_transient_sts_error(self, error_code: str) -> bool:
        return error_code in self.TRANSIENT_STS_ERROR_CODES

    def _assume_bedrock_role(self) -> dict:
        """Assume the Bedrock role and return temporary credentials."""
        if not self.settings.bedrock_role_arn:
            raise ValueError("BEDROCK_ROLE_ARN is not configured")

        try:
            logger.info(f"Assuming Bedrock role: {self.settings.bedrock_role_arn}")

            sts_client = boto3.client('sts', region_name=self.settings.aws_region, config=self._get_sts_config())

            response = sts_client.assume_role(
                RoleArn=self.settings.bedrock_role_arn,
                RoleSessionName=self.settings.bedrock_session_name,
                DurationSeconds=self.settings.bedrock_role_duration
            )

            credentials = response['Credentials']
            self._credentials_expiration = self._normalize_expiration(credentials['Expiration'])
            logger.info(f"Successfully assumed Bedrock role. Session expires at: {credentials['Expiration']}")

            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to assume Bedrock role {self.settings.bedrock_role_arn}: {error_code}")
            raise RoleAssumptionError(
                f"Role assumption failed: {error_code}",
                transient=self._is_transient_sts_error(error_code),
            ) from e
        except BotoCoreError as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RoleAssumptionError(f"Role assumption failed: {str(e)}", transient=True) from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RoleAssumptionError(f"Role assumption failed: {str(e)}", transient=False) from e

    def _get_bedrock_client(self) -> boto3.client:
        if self._client is not None and (not self.settings.bedrock_role_arn or not self._credentials_expired()):
            return self._client

        if self._client is not None and self.settings.bedrock_role_arn and self._credentials_expired():
            logger.info("Assumed role credentials expired or expiring soon, refreshing...")

        return self.refresh_credentials()

    def refresh_credentials(self, force: bool = False) -> boto3.client:
        """Refresh the cached Bedrock client under a lock and return a usable client."""
        with self._refresh_lock:
            if (
                self._client is not None
                and not force
                and (not self.settings.bedrock_role_arn or not self._credentials_expired())
            ):
                return self._client

            logger.info(f"Initializing Bedrock client: region={self.settings.aws_region}")

            if self.settings.bedrock_role_arn:
                logger.info("Using role assumption for Bedrock access")
                try:
                    credentials = self._assume_bedrock_role()
                    self._assumed_credentials = credentials

                    self._client = boto3.client(
                        service_name="bedrock-runtime",
                        region_name=self.settings.aws_region,
                        config=self._get_boto_config(),
                        **credentials
                    )
                except RoleAssumptionError as e:
                    if e.transient and self._client is not None and self._credentials_still_valid():
                        logger.warning(
                            "Transient Bedrock role refresh failed; continuing with cached client "
                            "until credentials expire: %s",
                            e,
                        )
                        return self._client
                    raise
            else:
                logger.info("Using default AWS credentials for Bedrock access")
                self._client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.settings.aws_region,
                    config=self._get_boto_config(),
                )

            return self._client

    def start_credential_refresh(self) -> None:
        """Start one daemon credential refresh worker for this process."""
        if not self.settings.bedrock_role_arn:
            return

        with self._refresh_thread_lock:
            if self._refresh_thread is not None and self._refresh_thread.is_alive():
                return

            self._refresh_stop_event.clear()
            self._refresh_thread = threading.Thread(
                target=self._credential_refresh_loop,
                name="bedrock-credential-refresh",
                daemon=True,
            )
            self._refresh_thread.start()
            logger.info("Started Bedrock credential refresh worker")

    def stop_credential_refresh(self) -> None:
        """Stop the daemon credential refresh worker."""
        with self._refresh_thread_lock:
            thread = self._refresh_thread
            if thread is None:
                return

            self._refresh_stop_event.set()

        thread.join(timeout=1.0)

        with self._refresh_thread_lock:
            if self._refresh_thread is thread:
                self._refresh_thread = None

    def _credential_refresh_loop(self) -> None:
        """Refresh credentials before request-time refresh buffer opens."""
        while not self._refresh_stop_event.is_set():
            delay_seconds = self._background_refresh_delay_seconds()
            if self._refresh_stop_event.wait(delay_seconds):
                return

            previous_expiration = self._credentials_expiration
            try:
                self.refresh_credentials(force=True)
                if previous_expiration is not None and self._credentials_expiration == previous_expiration:
                    self._refresh_stop_event.wait(self._background_retry_delay_seconds())
            except Exception as e:
                logger.warning("Background Bedrock credential refresh failed: %s", e)
                self._refresh_stop_event.wait(self._background_retry_delay_seconds())

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
