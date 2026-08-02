"""Refreshable AssumeRole-backed boto3 client factory.

Centralizes botocore's ``RefreshableCredentials`` wiring so any AWS-backed
client (Kendra, Bedrock, ...) gets STS role assumption with correct
concurrent-refresh semantics without hand-rolling expiry tracking: botocore's
own advisory/mandatory refresh windows and internal locking already handle
that (non-blocking refresh attempts outside the mandatory window, automatic
tolerance of a failed refresh as long as the cached credentials haven't
actually expired yet).

``refresh_metadata()`` is the only network call in this module (STS AssumeRole)
and is retried (3 attempts, exponential backoff) since botocore invokes it on
every credential refresh — a single transient STS throttle or network blip
should not fail every in-flight request that happens to need fresh credentials
at that moment.

``CredentialRefreshWorker`` is an optional daemon thread that periodically
touches a factory's credentials so botocore's advisory-window refresh runs
proactively, off the request path, instead of only at request-signing time.
"""

from __future__ import annotations

import random
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import botocore.session
from botocore.credentials import RefreshableCredentials
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CredentialsError(RuntimeError):
    """Raised when AWS STS role assumption fails after exhausting retries."""


class AssumedRoleClientFactory:
    """Create boto3 clients whose STS credentials refresh via botocore."""

    def __init__(
        self,
        *,
        role_arn: str,
        session_name: str,
        duration_seconds: int,
        region_name: str,
        method: str,
    ) -> None:
        if not role_arn:
            raise ValueError("role_arn is required")
        self.role_arn = role_arn
        self.session_name = session_name
        self.duration_seconds = duration_seconds
        self.region_name = region_name
        self.method = method
        self._session: boto3.Session | None = None
        self._lock = threading.Lock()

    def client(self, service_name: str, **kwargs: Any) -> Any:
        """Return a service client backed by one refreshable boto3 session."""
        return self.session().client(
            service_name=service_name,
            region_name=kwargs.pop("region_name", self.region_name),
            **kwargs,
        )

    def session(self) -> boto3.Session:
        """Return the cached refreshable boto3 session, building it once."""
        if self._session is not None:
            return self._session

        with self._lock:
            if self._session is None:
                self._session = self._build_session()
            return self._session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def refresh_metadata(self) -> dict[str, str]:
        """Assume the configured role and return RefreshableCredentials metadata.

        This is the callback botocore's ``RefreshableCredentials`` invokes on every
        refresh (first build and every subsequent expiry). Retried up to 3x with
        exponential backoff (1-8s) so a transient STS throttle or network blip
        doesn't fail every in-flight request that happens to need fresh
        credentials at that moment.
        """
        logger.info("Assuming role for %s: %s", self.method, self.role_arn)
        sts_client = boto3.client("sts", region_name=self.region_name)

        try:
            response = sts_client.assume_role(
                RoleArn=self.role_arn,
                RoleSessionName=self.session_name,
                DurationSeconds=self.duration_seconds,
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            logger.error("AssumeRole failed for %s: %s", self.method, error_code)
            raise CredentialsError(f"Role assumption failed: {error_code}") from exc
        except Exception as exc:
            logger.error("AssumeRole failed for %s: %s", self.method, exc)
            raise CredentialsError(
                f"Role assumption failed: {type(exc).__name__}"
            ) from exc

        credentials = response["Credentials"]
        expiry_time = self._expiry_time(credentials["Expiration"])
        logger.info(
            "Assumed role for %s; credentials expire at %s",
            self.method,
            expiry_time,
        )
        return {
            "access_key": credentials["AccessKeyId"],
            "secret_key": credentials["SecretAccessKey"],
            "token": credentials["SessionToken"],
            "expiry_time": expiry_time,
        }

    def _build_session(self) -> boto3.Session:
        metadata = self.refresh_metadata()
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=metadata,
            refresh_using=self.refresh_metadata,
            method=self.method,
        )

        botocore_session = botocore.session.get_session()
        botocore_session._credentials = refreshable_credentials
        botocore_session.set_config_variable("region", self.region_name)
        return boto3.Session(botocore_session=botocore_session)

    @staticmethod
    def _expiry_time(value: datetime | str) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return value


class CredentialRefreshWorker:
    """Proactively touches a factory's credentials so refresh happens off the request path.

    Cheap by design: touching credentials that aren't yet due for refresh is a
    local check only (no network call), so a short poll interval just gives
    botocore's advisory window (15 min by default) many chances to refresh
    proactively before its mandatory window (10 min by default) would ever
    block a request.
    """

    def __init__(
        self,
        factory: AssumedRoleClientFactory,
        name: str,
        poll_interval_seconds: float = 120.0,
        jitter_seconds: float = 30.0,
    ) -> None:
        self._factory = factory
        self._name = name
        self._poll_interval_seconds = poll_interval_seconds
        self._jitter_seconds = jitter_seconds
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def warm(self) -> None:
        """Synchronously build the session/credentials once. Call at startup."""
        self._touch()

    def start(self) -> None:
        """Start the daemon refresh-touch thread (idempotent)."""
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, name=self._name, daemon=True
            )
            self._thread.start()
            logger.info("Started credential refresh worker: %s", self._name)

    def stop(self) -> None:
        """Signal the worker to stop and wait briefly for it to exit."""
        with self._thread_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout=5.0)
        with self._thread_lock:
            if self._thread is thread:
                self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            delay = self._poll_interval_seconds + random.uniform(0, self._jitter_seconds)
            if self._stop_event.wait(delay):
                return
            try:
                self._touch()
            except Exception as e:
                logger.warning(
                    "Background credential touch failed for %s: %s", self._name, e
                )

    def _touch(self) -> None:
        """Read frozen credentials, letting botocore refresh proactively if due."""
        self._factory.session().get_credentials().get_frozen_credentials()
