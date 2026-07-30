import os
import time
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Dict, Any, Optional
from src.tools.benefits import search_benefits
from src.config.messages import get_message
from src.config.settings import get_settings
from src.services.kendra_service import get_ncct_ids_by_product, QueryLimitExceededError
from src.utils.logging import get_logger

# Conditional imports for LLM functionality
try:
    from langchain.agents import AgentExecutor
    from langchain.agents import create_agent
    from langchain_aws import ChatBedrock
    from langchain.prompts import ChatPromptTemplate
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"LLM dependencies not available: {e}")

logger = get_logger(__name__)

# Custom exceptions
class KendraSearchError(Exception):
    """Raised when Kendra search operations fail."""
    pass

class QueryProcessingError(Exception):
    """Raised when query processing fails."""
    pass

class HybridIBTAgent:
    def __init__(self):
        self.settings = get_settings()
        self.kendra_index_id = self.settings.kendra_index_id
        self.aws_region = self.settings.aws_region
        
        self._llm = None
        self._bedrock_client = None
        self._assumed_credentials = None
        self._agent_executor = None
        self.use_llm = os.getenv('USE_LLM', 'true').lower() == 'true' and LLM_AVAILABLE
    
    def _get_boto_config(self) -> Config:
        """Get boto3 configuration with timeout and retry settings."""
        return Config(
            read_timeout=self.settings.bedrock_read_timeout,
            connect_timeout=self.settings.bedrock_connect_timeout,
            retries={"max_attempts": self.settings.bedrock_max_retries, "mode": "adaptive"},
        )

    def _assume_bedrock_role(self) -> dict:
        """Assume the Bedrock role and return temporary credentials."""
        bedrock_role_arn = os.getenv('BEDROCK_ROLE_ARN')
        if not bedrock_role_arn:
            raise ValueError("BEDROCK_ROLE_ARN environment variable is not set")

        try:
            logger.info(f"Assuming Bedrock role: {bedrock_role_arn}")
            
            # Create STS client with default credentials
            sts_client = boto3.client('sts', region_name=self.settings.aws_region)
            
            # Assume the role
            response = sts_client.assume_role(
                RoleArn=bedrock_role_arn,
                RoleSessionName=self.settings.bedrock_session_name,
                DurationSeconds=self.settings.bedrock_role_duration
            )
            
            credentials = response['Credentials']
            logger.info(f"Successfully assumed Bedrock role. Session expires at: {credentials['Expiration']}")
            
            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to assume Bedrock role {bedrock_role_arn}: {error_code}")
            raise RuntimeError(f"Role assumption failed: {error_code}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RuntimeError(f"Role assumption failed: {str(e)}") from e

    def _get_bedrock_client(self) -> boto3.client:
        """Get or create a Bedrock client with optional role assumption."""
        if self._bedrock_client is not None:
            return self._bedrock_client

        logger.info(f"Initializing Bedrock client: region={self.settings.aws_region}")
        
        # Check if role assumption is configured
        bedrock_role_arn = os.getenv('BEDROCK_ROLE_ARN')
        if bedrock_role_arn:
            logger.info("Using role assumption for Bedrock access")
            credentials = self._assume_bedrock_role()
            self._assumed_credentials = credentials
            
            self._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.settings.aws_region,
                config=self._get_boto_config(),
                **credentials
            )
        else:
            logger.info("Using default AWS credentials for Bedrock access")
            self._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.settings.aws_region,
                config=self._get_boto_config(),
            )
        
        return self._bedrock_client
    
    @property
    def llm(self):
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM dependencies not available")
        if self._llm is None:
            logger.info(f"Creating LLM: {self.settings.bedrock_model_id}")
            self._llm = ChatBedrock(
                client=self._get_bedrock_client(),
                model_id=self.settings.bedrock_model_id,
                region_name=self.settings.aws_region,
                model_kwargs={
                    'temperature': self.settings.bedrock_temperature,
                    'max_tokens': self.settings.bedrock_max_tokens
                }
            )
        return self._llm
    
    @property
    def agent_executor(self):
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM dependencies not available")
        if self._agent_executor is None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an insurance benefits assistant. Use search_benefits tool to find information. Create HTML links as: <a href='{{NCCT_ID}}'>{{Service_Name}}</a>"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
            agent = create_agent(self.llm, [search_benefits], prompt)
            self._agent_executor = AgentExecutor(agent=agent, tools=[search_benefits], verbose=True)
        return self._agent_executor
    
    def process_query(self, user_prompt: str, session_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # Log context information
        if context:
            logger.info(f"Processing query with context: userName={context.get('userName')}, userType={context.get('userType')}, productId={context.get('productId')}, source={context.get('source')}")
        else:
            logger.info("Processing query without context")
        
        try:
            if self.use_llm:
                result = self._process_with_llm(user_prompt, context)
            else:
                result = self._process_direct_kendra(user_prompt, context)
            
            execution_time = (time.time() - start_time) * 1000
            
            response = {
                "sessionId": session_id,
                "confidence": result.get("confidence", 0.0),
                "responseText": result.get("response_text", ""),
                "success": result.get("success", False),
                "execution_time_ms": round(execution_time, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "llm_enhanced" if self.use_llm else "direct_kendra"
            }
            logger.info("IBT Agent response: session_id=%s, mode=%s, success=%s, confidence=%s, responseText=%s",
                        response["sessionId"], response["mode"], response["success"], response["confidence"], response["responseText"])
            return response
            
        except QueryLimitExceededError:
            logger.warning(f"Query limit exceeded for session {session_id}")
            response = {
                "sessionId": session_id,
                "confidence": 0.0,
                "responseText": get_message("query_limit_exceeded"),
                "success": False,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "llm_enhanced" if self.use_llm else "direct_kendra"
            }
            logger.info("IBT Agent query limit response: session_id=%s, responseText=%s",
                        response["sessionId"], response["responseText"])
            return response
        except (KendraSearchError, RuntimeError) as e:
            logger.error(f"Query processing error: {str(e)}")
            response = {
                "sessionId": session_id,
                "confidence": 0.0,
                "responseText": get_message("service_unavailable"),
                "success": False,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "llm_enhanced" if self.use_llm else "direct_kendra"
            }
            logger.info("IBT Agent error response: session_id=%s, mode=%s, success=%s, responseText=%s",
                        response["sessionId"], response["mode"], response["success"], response["responseText"])
            return response
    
    def _process_with_llm(self, user_prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        enhanced_prompt = user_prompt
        if context and (context.get('userName') or context.get('userType')):
            enhanced_prompt = f"User: {context.get('userName', '')} ({context.get('userType', '')})\nQuery: {user_prompt}"
            logger.info(f"Enhanced LLM prompt with user context: {enhanced_prompt[:100]}...")
        
        result = self.agent_executor.invoke({"input": enhanced_prompt})
        return {"success": True, "response_text": result.get("output", ""), "confidence": 8.0}
    
    def _process_direct_kendra(self, user_prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        # Extract product ID from context with fallback to default
        product_id = "1"  # Default to FEHB Standard plan
        if context:
            # Handle both camelCase (from orchestrator) and snake_case (from tests)
            product_id = context.get('productId') or context.get('product_id') or "1"
            logger.info(f"Using product ID from context: {product_id}")
        else:
            logger.warning(f"No product ID in context, using default: {product_id}")
        
        logger.info(f"Processing direct Kendra query with product ID: {product_id}")
        
        # Always use product-filtered NCCT ID extraction
        ncct_ids = get_ncct_ids_by_product(user_prompt, str(product_id))
        logger.info(f"Product {product_id} filtering applied, found {len(ncct_ids)} results")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ncct_ids = []
        for ncct_id in ncct_ids:
            if ncct_id not in seen:
                seen.add(ncct_id)
                unique_ncct_ids.append(ncct_id)
        
        if not unique_ncct_ids:
            logger.info("No NCCT IDs found for the query and product combination")
            return {
                "success": True,
                "response_text": get_message("no_results_found"),
                "confidence": 0.0,
                "product_id": product_id,
                "ncct_count": 0
            }
        
        logger.info(f"Found {len(unique_ncct_ids)} unique NCCT IDs: {unique_ncct_ids}")
        
        return {
            "success": True,
            "response_text": unique_ncct_ids,
            "confidence": 8.0,  # High confidence for direct Kendra results
            "product_id": product_id,
            "ncct_count": len(unique_ncct_ids)
        }
    
    def set_mode(self, use_llm: bool):
        if use_llm and not LLM_AVAILABLE:
            logger.warning("Cannot enable LLM mode: LLM dependencies not available")
            self.use_llm = False
        else:
            self.use_llm = use_llm
        logger.info(f"Switched to {'LLM-enhanced' if self.use_llm else 'direct Kendra'} mode")
    
    def get_mode_info(self) -> Dict[str, Any]:
        bedrock_role_arn = os.getenv('BEDROCK_ROLE_ARN')
        return {
            "current_mode": "llm_enhanced" if self.use_llm else "direct_kendra",
            "kendra_index_id": self.kendra_index_id,
            "aws_region": self.aws_region,
            "kendra_role_arn": self.settings.kendra_role_arn,
            "bedrock_role_arn": bedrock_role_arn,
            "using_kendra_role_assumption": bool(self.settings.kendra_role_arn),
            "using_bedrock_role_assumption": bool(bedrock_role_arn)
        }