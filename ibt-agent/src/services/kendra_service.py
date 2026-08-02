"""Direct AWS Kendra integration without fallback logic."""

import boto3
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError
from botocore.config import Config
from src.config.settings import get_settings
from src.config.constants import (
    DEFAULT_PAGE_SIZE,
    NCCT_ID_KEYS, SERVICE_NAME_KEYS,
    NO_EXCERPT_MESSAGE, PROCESSING_ERROR_MESSAGE
)
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
    
    CREDENTIALS_REFRESH_BUFFER = timedelta(minutes=5)

    def __init__(self, index_id: Optional[str] = None, region: Optional[str] = None):
        self.settings = get_settings()
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None
        
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
    
    def _assume_kendra_role(self) -> Dict[str, str]:
        """Assume the Kendra role and return temporary credentials."""
        if not self.settings.kendra_role_arn:
            raise ValueError("KENDRA_ROLE_ARN is not configured")
        
        try:
            logger.info(f"Assuming Kendra role: {self.settings.kendra_role_arn}")
            
            sts_client = boto3.client('sts', region_name=self.region)
            
            response = sts_client.assume_role(
                RoleArn=self.settings.kendra_role_arn,
                RoleSessionName=self.settings.kendra_session_name,
                DurationSeconds=self.settings.kendra_role_duration
            )
            
            credentials = response['Credentials']
            self._credentials_expiration = credentials['Expiration']
            logger.info(f"Successfully assumed Kendra role. Session expires at: {credentials['Expiration']}")
            
            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to assume Kendra role {self.settings.kendra_role_arn}: {error_code} - {error_msg}")
            raise RuntimeError(f"Role assumption failed: {error_code} - {error_msg}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RuntimeError(f"Role assumption failed: {str(e)}") from e
    
    def _credentials_expired(self) -> bool:
        """Check if assumed credentials are expired or about to expire."""
        if self._credentials_expiration is None:
            return True
        return datetime.now(timezone.utc) >= self._credentials_expiration - self.CREDENTIALS_REFRESH_BUFFER

    def _get_kendra_client(self) -> boto3.client:
        """Get Kendra client with appropriate credentials."""
        if self._client is not None and (not self.settings.kendra_role_arn or not self._credentials_expired()):
            return self._client
        
        if self._client is not None and self.settings.kendra_role_arn and self._credentials_expired():
            logger.info("Assumed role credentials expired or expiring soon, refreshing...")
        
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
            except Exception as e:
                logger.error(f"Role assumption failed, falling back to default credentials: {e}")
                
                self._client = boto3.client(
                    'kendra',
                    region_name=self.region,
                    config=self._get_boto_config()
                )
        else:
            logger.info("Using default AWS credentials for Kendra access")
            self._client = boto3.client(
                'kendra',
                region_name=self.region,
                config=self._get_boto_config()
            )
        
        return self._client
    
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
    
    def _format_results(self, items: List[Dict]) -> List[Dict[str, Any]]:
        """Format Kendra results - accept ANY data structure."""
        results = []
        logger.info(f"Processing {len(items)} raw Kendra items")
        
        for i, item in enumerate(items):
            try:
                logger.info(f"Processing item {i+1}: {list(item.keys())}")
                
                title = None
                excerpt = None
                ncct_id = None
                service_name = None
                
                if item.get('DocumentTitle', {}).get('Text'):
                    title = item['DocumentTitle']['Text']
                    logger.info(f"Found title: {title}")
                
                if item.get('DocumentExcerpt', {}).get('Text'):
                    excerpt = item['DocumentExcerpt']['Text']
                    logger.info(f"Found excerpt: {excerpt[:100]}...")
                
                attributes = {}
                for attr in item.get('DocumentAttributes', []):
                    try:
                        key = attr.get('Key', '')
                        value = attr.get('Value', {})
                        
                        if 'StringValue' in value:
                            attributes[key] = value['StringValue']
                        elif 'StringListValue' in value:
                            attributes[key] = ', '.join(value['StringListValue'])
                        elif 'LongValue' in value:
                            attributes[key] = str(value['LongValue'])
                        elif 'DateValue' in value:
                            attributes[key] = str(value['DateValue'])
                            
                        logger.info(f"Attribute {key}: {attributes.get(key, 'N/A')}")
                    except Exception as e:
                        logger.warning(f"Error processing attribute: {e}")
                
                ncct_id = None
                for key in NCCT_ID_KEYS:
                    if attributes.get(key):
                        ncct_id = attributes[key]
                        break
                if not ncct_id:
                    ncct_id = item.get('DocumentId') or f'DOC_{i+1}'
                
                service_name = None
                for key in SERVICE_NAME_KEYS:
                    if attributes.get(key):
                        service_name = attributes[key]
                        break
                if not service_name:
                    service_name = title or f'Document {i+1}'
                
                confidence = item.get('ScoreAttributes', {}).get('ScoreConfidence')
                
                result = {
                    'ncct_id': ncct_id,
                    'service_name': service_name,
                    'excerpt': excerpt or NO_EXCERPT_MESSAGE,
                    'confidence_score': confidence,
                    'all_attributes': attributes,
                    'document_uri': item.get('DocumentURI', ''),
                    'item_type': item.get('Type', 'DOCUMENT')
                }
                
                results.append(result)
                logger.info(f"Added result: {service_name} (ID: {ncct_id})")
                
            except Exception as e:
                logger.error(f"Error processing item {i+1}: {e}")
                results.append({
                    'ncct_id': f'ERR_{i+1}',
                    'service_name': f'Kendra Document {i+1}',
                    'excerpt': PROCESSING_ERROR_MESSAGE,
                    'confidence_score': 'LOW',
                    'error': str(e)
                })
        
        logger.info(f"Successfully formatted {len(results)} results from {len(items)} raw items")
        return results

_kendra_service = None

def get_kendra_service() -> KendraService:
    """Get singleton KendraService instance."""
    global _kendra_service
    if _kendra_service is None:
        _kendra_service = KendraService()
    return _kendra_service

def get_ncct_ids_by_product(query: str, product_id: str = None) -> List[str]:
    """Get NCCT IDs from Kendra search filtered by product."""
    service = get_kendra_service()
    return service.get_ncct_ids_by_product(query, product_id)
