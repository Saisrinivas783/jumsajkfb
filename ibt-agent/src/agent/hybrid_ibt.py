import time
from typing import Dict, Any
from src.config.messages import get_message
from src.config.settings import get_settings
from src.services.kendra_service import get_ncct_ids_by_product, QueryLimitExceededError
from src.utils.logging import get_logger

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

    def process_query(self, user_prompt: str, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        logger.info(f"Processing query with context: userName={context.get('userName')}, userType={context.get('userType')}, productId={context.get('productId')}, source={context.get('source')}")

        try:
            result = self._process_direct_kendra(user_prompt, context)

            execution_time = (time.time() - start_time) * 1000

            response = {
                "sessionId": session_id,
                "confidence": result.get("confidence", 0.0),
                "responseText": result.get("response_text", ""),
                "success": result.get("success", False),
                "execution_time_ms": round(execution_time, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            logger.info("IBT Agent response: session_id=%s, success=%s, confidence=%s, responseText=%s",
                        response["sessionId"], response["success"], response["confidence"], response["responseText"])
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
            }
            logger.info("IBT Agent error response: session_id=%s, success=%s, responseText=%s",
                        response["sessionId"], response["success"], response["responseText"])
            return response

    def _process_direct_kendra(self, user_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        product_id = context['productId']

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
