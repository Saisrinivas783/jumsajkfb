"""Direct AWS Kendra integration without fallback logic."""

import random
import threading
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import List, Dict, Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from botocore.config import Config
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QueryLimitExceededError(Exception):
    """Raised when Kendra query limit is exceeded."""
    pass


class RoleAssumptionError(RuntimeError):
    """Raised when STS role assumption fails."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


# Product to Plan/Brochure mapping
PRODUCT_MAPPING = {
    "1": {"plan": "standard/basic", "brochure": "fehb"},
    "4": {"plan": "standard/basic", "brochure": "fehb"},
    "6": {"plan": "blue focus", "brochure": "fehb"},
    "7": {"plan": "standard/basic", "brochure": "pshb"},
    "8": {"plan": "standard/basic", "brochure": "pshb"},
    "9": {"plan": "blue focus", "brochure": "pshb"},
}

class KendraService:
    """Direct AWS Kendra semantic search client without fallback logic."""

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

    def __init__(self, index_id: Optional[str] = None, region: Optional[str] = None):
        self.settings = get_settings()
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None
        self._refresh_lock = threading.Lock()
        self._refresh_thread_lock = threading.Lock()
        self._refresh_stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None

        if index_id and region:
            self.index_id = index_id
            self.region = region
        else:
            self.index_id = self.settings.kendra_index_id
            self.region = self.settings.aws_region

    def _get_boto_config(self) -> Config:
        """Get boto3 configuration with timeout and retry settings."""
        return Config(
            read_timeout=300,
            connect_timeout=10,
            retries={"max_attempts": 3, "mode": "adaptive"},
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

    def _is_transient_sts_error(self, error_code: str) -> bool:
        return error_code in self.TRANSIENT_STS_ERROR_CODES

    def _assume_kendra_role(self) -> Dict[str, str]:
        """Assume the Kendra role and return temporary credentials."""
        if not self.settings.kendra_role_arn:
            raise ValueError("KENDRA_ROLE_ARN is not configured")

        try:
            logger.info(f"Assuming Kendra role: {self.settings.kendra_role_arn}")

            sts_client = boto3.client('sts', region_name=self.region, config=self._get_sts_config())

            response = sts_client.assume_role(
                RoleArn=self.settings.kendra_role_arn,
                RoleSessionName=self.settings.kendra_session_name,
                DurationSeconds=self.settings.kendra_role_duration
            )

            credentials = response['Credentials']
            self._credentials_expiration = self._normalize_expiration(credentials['Expiration'])
            logger.info(f"Successfully assumed Kendra role. Session expires at: {credentials['Expiration']}")

            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error'].get('Message', '')
            logger.error(f"Failed to assume Kendra role {self.settings.kendra_role_arn}: {error_code} - {error_msg}")
            raise RoleAssumptionError(
                f"Role assumption failed: {error_code} - {error_msg}",
                transient=self._is_transient_sts_error(error_code),
            ) from e
        except BotoCoreError as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RoleAssumptionError(f"Role assumption failed: {str(e)}", transient=True) from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RoleAssumptionError(f"Role assumption failed: {str(e)}", transient=False) from e

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

    def _get_kendra_client(self) -> boto3.client:
        """Get Kendra client with appropriate credentials."""
        if self._client is not None and (not self.settings.kendra_role_arn or not self._credentials_expired()):
            return self._client

        if self._client is not None and self.settings.kendra_role_arn and self._credentials_expired():
            logger.info("Assumed role credentials expired or expiring soon, refreshing...")

        return self.refresh_credentials()

    def refresh_credentials(self, force: bool = False) -> boto3.client:
        """Refresh the cached Kendra client under a lock and return a usable client."""
        with self._refresh_lock:
            if (
                self._client is not None
                and not force
                and (not self.settings.kendra_role_arn or not self._credentials_expired())
            ):
                return self._client

            logger.info(f"Initializing Kendra client: region={self.region}, index_id={self.index_id}")

            if self.settings.kendra_role_arn:
                logger.info("Using role assumption for Kendra access")
                try:
                    credentials = self._assume_kendra_role()
                    self._assumed_credentials = credentials

                    self._client = boto3.client(
                        'kendra',
                        region_name=self.region,
                        config=self._get_boto_config(),
                        **credentials
                    )
                except RoleAssumptionError as e:
                    if e.transient and self._client is not None and self._credentials_still_valid():
                        logger.warning(
                            "Transient Kendra role refresh failed; continuing with cached client "
                            "until credentials expire: %s",
                            e,
                        )
                        return self._client
                    raise
            else:
                logger.info("Using default AWS credentials for Kendra access")
                self._client = boto3.client(
                    'kendra',
                    region_name=self.region,
                    config=self._get_boto_config()
                )

            return self._client

    def start_credential_refresh(self) -> None:
        """Start one daemon credential refresh worker for this process."""
        if not self.settings.kendra_role_arn:
            return

        with self._refresh_thread_lock:
            if self._refresh_thread is not None and self._refresh_thread.is_alive():
                return

            self._refresh_stop_event.clear()
            self._refresh_thread = threading.Thread(
                target=self._credential_refresh_loop,
                name="kendra-credential-refresh",
                daemon=True,
            )
            self._refresh_thread.start()
            logger.info("Started Kendra credential refresh worker")

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
                logger.warning("Background Kendra credential refresh failed: %s", e)
                self._refresh_stop_event.wait(self._background_retry_delay_seconds())

    @property
    def client(self) -> boto3.client:
        """Get the Kendra client (lazy initialization)."""
        return self._get_kendra_client()

    def _build_attribute_filter(self, product_config: Dict[str, str]) -> Dict[str, Any]:
        """Build Kendra AttributeFilter for product filtering."""
        if not product_config:
            return None

        return {
            "AndAllFilters": [
                {
                    "EqualsTo": {
                        "Key": "plan",
                        "Value": {
                            "StringValue": product_config['plan']
                        }
                    }
                },
                {
                    "EqualsTo": {
                        "Key": "brochure",
                        "Value": {
                            "StringValue": product_config['brochure']
                        }
                    }
                }
            ]
        }

    def get_ncct_ids_by_product(self, query: str, product_id: str = None) -> List[str]:
        """Search Kendra with product filter and return only NCCT IDs."""
        try:
            logger.info(f"Getting NCCT IDs for product {product_id} with query: {query[:50]}...")

            product_config = None
            attribute_filter = None

            if product_id and product_id in PRODUCT_MAPPING:
                product_config = PRODUCT_MAPPING[product_id]
                attribute_filter = self._build_attribute_filter(product_config)
                logger.info(f"Product {product_id} requires plan: {product_config['plan']}, brochure: {product_config['brochure']}")

            query_params = {
                'IndexId': self.index_id,
                'QueryText': query,
                'PageSize': self.settings.kendra_page_size,
                'RequestedDocumentAttributes': ['NCCTID']
            }

            if attribute_filter:
                query_params['AttributeFilter'] = attribute_filter

            response = self.client.query(**query_params)
            items = response.get('ResultItems', [])

            if not items:
                logger.info("No results found")
                return []

            ncct_ids = [
                attr['Value']['StringValue']
                for item in items
                for attr in item.get('DocumentAttributes', [])
                if (attr['Key'] == 'NCCTID' and attr.get('Value', {}).get('StringValue') and
                    item.get('ScoreAttributes', {}).get('ScoreConfidence') in ['VERY_HIGH', 'HIGH', 'MEDIUM'])
            ]

            logger.info(f"Extracted {len(ncct_ids)} NCCT IDs for product {product_id}")
            return ncct_ids

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning(f"Query limit exceeded for product {product_id}")
                raise QueryLimitExceededError("Kendra query limit exceeded") from e
            logger.error(f"Kendra API error for product {product_id}: {error_code} - {str(e)}")
            raise RuntimeError(f"Kendra search failed for product {product_id}: {error_code}") from e
        except QueryLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Error extracting NCCT IDs for product {product_id}: {str(e)}")
            raise RuntimeError(f"Kendra search failed for product {product_id}: {str(e)}") from e


@lru_cache(maxsize=1)
def get_kendra_service() -> KendraService:
    """Get singleton KendraService instance."""
    return KendraService()

def get_ncct_ids_by_product(query: str, product_id: str = None) -> List[str]:
    """Get NCCT IDs from Kendra search filtered by product."""
    service = get_kendra_service()
    return service.get_ncct_ids_by_product(query, product_id)
