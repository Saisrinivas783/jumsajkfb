"""Direct AWS Kendra integration without fallback logic."""

import threading
from functools import lru_cache
from typing import List, Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

from src.aws.assume_role import AssumedRoleClientFactory, CredentialRefreshWorker
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QueryLimitExceededError(Exception):
    """Raised when Kendra query limit is exceeded."""
    pass


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
        self._client_lock = threading.Lock()
        self._assume_role_factory: Optional[AssumedRoleClientFactory] = None
        self._refresh_worker: Optional[CredentialRefreshWorker] = None

        if index_id and region:
            self.index_id = index_id
            self.region = region
        else:
            self.index_id = self.settings.kendra_index_id
            self.region = self.settings.aws_region

    def _get_boto_config(self) -> Config:
        """Get boto3 configuration with timeout, retry, and pool settings."""
        return Config(
            read_timeout=300,
            connect_timeout=10,
            max_pool_connections=self.settings.kendra_max_pool_connections,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

    def _get_assume_role_factory(self) -> AssumedRoleClientFactory:
        """Return the Kendra role factory with botocore-managed refresh."""
        if not self.settings.kendra_role_arn:
            raise ValueError("KENDRA_ROLE_ARN is not configured")

        if self._assume_role_factory is None:
            self._assume_role_factory = AssumedRoleClientFactory(
                role_arn=self.settings.kendra_role_arn,
                session_name=self.settings.kendra_session_name,
                duration_seconds=self.settings.kendra_role_duration,
                region_name=self.region,
                method="kendra-assume-role",
            )
        return self._assume_role_factory

    def _get_kendra_client(self) -> boto3.client:
        """Get the Kendra client, built once and cached.

        Credentials refresh themselves in place via botocore's
        ``RefreshableCredentials`` (see ``AssumedRoleClientFactory``), so the
        client object never needs to be rebuilt once constructed.
        """
        if self._client is not None:
            return self._client

        with self._client_lock:
            if self._client is not None:
                return self._client

            logger.info(f"Initializing Kendra client: region={self.region}, index_id={self.index_id}")

            if self.settings.kendra_role_arn:
                logger.info("Using refreshable role assumption for Kendra access")
                self._client = self._get_assume_role_factory().client(
                    "kendra", config=self._get_boto_config()
                )
            else:
                logger.info("Using default AWS credentials for Kendra access")
                self._client = boto3.client(
                    "kendra", region_name=self.region, config=self._get_boto_config()
                )

        return self._client

    @property
    def client(self) -> boto3.client:
        """Get the Kendra client (lazy initialization)."""
        return self._get_kendra_client()

    def warm_credentials(self) -> None:
        """Build the Kendra client once, synchronously. Call at startup, off the event loop."""
        self._get_kendra_client()

    def start_credential_refresh(self) -> None:
        """Start the background worker that proactively refreshes assumed-role credentials."""
        if not self.settings.kendra_role_arn:
            return
        if self._refresh_worker is None:
            self._refresh_worker = CredentialRefreshWorker(
                self._get_assume_role_factory(), name="kendra-credential-refresh"
            )
        self._refresh_worker.start()

    def stop_credential_refresh(self) -> None:
        """Stop the background credential refresh worker."""
        if self._refresh_worker is not None:
            self._refresh_worker.stop()

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
