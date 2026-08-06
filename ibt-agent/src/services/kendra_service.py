"""Direct AWS Kendra integration without fallback logic."""

import boto3
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from botocore.exceptions import ClientError
from botocore.config import Config
from src.config.settings import get_settings
from src.exceptions import UpstreamServiceError
from src.utils.logging import get_logger

logger = get_logger(__name__)


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

    def __init__(self, index_id: Optional[str] = None, region: Optional[str] = None):
        self.settings = get_settings()
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None
        self._client_lock = threading.Lock()

        if index_id and region:
            self.index_id = index_id
            self.region = region
        else:
            self.index_id = self.settings.kendra_index_id
            self.region = self.settings.aws_region

    def _get_boto_config(self) -> Config:
        """Get boto3 configuration with timeout, retry, and pool settings."""
        return Config(
            read_timeout=self.settings.kendra_read_timeout,
            connect_timeout=self.settings.kendra_connect_timeout,
            retries={"max_attempts": self.settings.kendra_max_retries, "mode": "standard"},
            max_pool_connections=self.settings.kendra_max_pool_connections,
        )

    def _get_sts_boto_config(self) -> Config:
        """Get boto3 configuration for the STS client used in role assumption."""
        return Config(
            max_pool_connections=self.settings.sts_max_pool_connections,
            connect_timeout=self.settings.sts_connect_timeout,
            read_timeout=self.settings.sts_read_timeout,
            retries={"max_attempts": self.settings.sts_max_retries, "mode": "standard"},
        )

    def _assume_kendra_role(self) -> Tuple[Dict[str, str], datetime]:
        """Assume the Kendra role and return temporary credentials and their expiration.

        Does not set self._credentials_expiration directly. The caller
        (_refresh_client_locked) assigns self._client before publishing the
        new expiration, so an unlocked reader never observes a fresh
        expiration paired with the stale, about-to-be-replaced client.
        """
        if not self.settings.kendra_role_arn:
            raise ValueError("KENDRA_ROLE_ARN is not configured")

        try:
            logger.info(f"Assuming Kendra role: {self.settings.kendra_role_arn}")

            sts_start = time.perf_counter()
            sts_client = boto3.client('sts', region_name=self.region, config=self._get_sts_boto_config())

            response = sts_client.assume_role(
                RoleArn=self.settings.kendra_role_arn,
                RoleSessionName=self.settings.kendra_session_name,
                DurationSeconds=self.settings.kendra_role_duration
            )
            sts_elapsed_ms = (time.perf_counter() - sts_start) * 1000

            credentials = response['Credentials']
            expiration = credentials['Expiration']
            logger.info(f"Successfully assumed Kendra role in {sts_elapsed_ms:.2f}ms. Session expires at: {expiration}")

            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }, expiration

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to assume Kendra role {self.settings.kendra_role_arn}: {error_code} - {error_msg}")
            raise UpstreamServiceError("kendra", f"Role assumption failed: {error_code} - {error_msg}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise UpstreamServiceError("kendra", f"Role assumption failed: {str(e)}") from e
    
    def _credentials_expired(self) -> bool:
        """Check if assumed credentials are expired or about to expire."""
        if self._credentials_expiration is None:
            return True
        buffer = timedelta(minutes=self.settings.credentials_refresh_buffer_minutes)
        return datetime.now(timezone.utc) >= self._credentials_expiration - buffer

    def _refresh_client_locked(self) -> boto3.client:
        """Refresh (or create) the Kendra client. Must be called while holding self._client_lock.

        Role-assumption failures are not caught here and propagate to the
        caller — there is no silent fallback to default AWS credentials.
        A broken KENDRA_ROLE_ARN should fail loudly rather than quietly
        serving Kendra results under a different (possibly wrong) identity.
        """
        logger.info(f"Initializing Kendra client: region={self.region}, index_id={self.index_id}")
        refresh_start = time.perf_counter()

        if self.settings.kendra_role_arn:
            logger.info("Using role assumption for Kendra access")
            credentials, expiration = self._assume_kendra_role()
            self._assumed_credentials = credentials

            # Assign the new client before publishing the new expiration.
            # If a reader takes the unlocked fast path in between, it sees
            # either the old client + old (still-expired) expiration, or
            # the new client + new expiration — never new expiration paired
            # with the stale client.
            client_create_start = time.perf_counter()
            self._client = boto3.client(
                'kendra',
                region_name=self.region,
                config=self._get_boto_config(),
                **credentials
            )
            client_create_elapsed_ms = (time.perf_counter() - client_create_start) * 1000
            self._credentials_expiration = expiration
            logger.info(f"boto3 Kendra client construction took {client_create_elapsed_ms:.2f}ms")
        else:
            logger.info("Using default AWS credentials for Kendra access")
            client_create_start = time.perf_counter()
            self._client = boto3.client(
                'kendra',
                region_name=self.region,
                config=self._get_boto_config()
            )
            client_create_elapsed_ms = (time.perf_counter() - client_create_start) * 1000
            logger.info(f"boto3 Kendra client construction took {client_create_elapsed_ms:.2f}ms")

        total_refresh_elapsed_ms = (time.perf_counter() - refresh_start) * 1000
        logger.info(f"Kendra client refresh (total) took {total_refresh_elapsed_ms:.2f}ms")
        return self._client

    def _get_kendra_client(self) -> boto3.client:
        """Get Kendra client with appropriate credentials (thread-safe)."""
        # Fast path: valid cached client, no lock needed.
        if self._client is not None and (not self.settings.kendra_role_arn or not self._credentials_expired()):
            return self._client

        lock_wait_start = time.perf_counter()
        with self._client_lock:
            lock_wait_elapsed_ms = (time.perf_counter() - lock_wait_start) * 1000
            if lock_wait_elapsed_ms > 50:
                logger.info(f"Waited {lock_wait_elapsed_ms:.2f}ms to acquire Kendra client lock (contention from concurrent refresh)")

            # Re-check inside the lock: another thread may have just refreshed it.
            if self._client is not None and (not self.settings.kendra_role_arn or not self._credentials_expired()):
                return self._client

            if self._client is not None and self.settings.kendra_role_arn and self._credentials_expired():
                logger.info("Assumed role credentials expired or expiring soon, refreshing...")

            return self._refresh_client_locked()
    
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
        method_start = time.perf_counter()
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

            client_start = time.perf_counter()
            client = self.client
            client_elapsed_ms = (time.perf_counter() - client_start) * 1000

            query_start = time.perf_counter()
            response = client.query(**query_params)
            query_elapsed_ms = (time.perf_counter() - query_start) * 1000

            logger.info(
                f"Kendra timing for product {product_id}: "
                f"client_acquire={client_elapsed_ms:.2f}ms, query_call={query_elapsed_ms:.2f}ms"
            )

            items = response.get('ResultItems', [])

            if not items:
                logger.info("No results found")
                return []

            extract_start = time.perf_counter()
            ncct_ids = [
                attr['Value']['StringValue']
                for item in items
                for attr in item.get('DocumentAttributes', [])
                if (attr['Key'] == 'NCCTID' and attr.get('Value', {}).get('StringValue') and
                    item.get('ScoreAttributes', {}).get('ScoreConfidence') in ['VERY_HIGH', 'HIGH', 'MEDIUM'])
            ]
            extract_elapsed_ms = (time.perf_counter() - extract_start) * 1000
            total_elapsed_ms = (time.perf_counter() - method_start) * 1000

            logger.info(
                f"Extracted {len(ncct_ids)} NCCT IDs for product {product_id} "
                f"(extract={extract_elapsed_ms:.2f}ms, total={total_elapsed_ms:.2f}ms)"
            )
            return ncct_ids

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning(f"Query limit exceeded for product {product_id}")
            else:
                logger.error(f"Kendra API error for product {product_id}: {error_code} - {str(e)}")
            raise UpstreamServiceError("kendra", f"Kendra search failed for product {product_id}: {error_code}") from e
        except Exception as e:
            logger.error(f"Error extracting NCCT IDs for product {product_id}: {str(e)}")
            raise UpstreamServiceError("kendra", f"Kendra search failed for product {product_id}: {str(e)}") from e
    
_kendra_service: Optional[KendraService] = None
_kendra_service_lock = threading.Lock()

def get_kendra_service() -> KendraService:
    """Get singleton KendraService instance (thread-safe)."""
    global _kendra_service
    if _kendra_service is not None:
        return _kendra_service

    with _kendra_service_lock:
        if _kendra_service is None:
            _kendra_service = KendraService()
        return _kendra_service

def get_ncct_ids_by_product(query: str, product_id: str = None) -> List[str]:
    """Get NCCT IDs from Kendra search filtered by product."""
    service = get_kendra_service()
    return service.get_ncct_ids_by_product(query, product_id)
