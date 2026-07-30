import time
from typing import Any, Dict

from src.config.messages import get_message
from src.config.settings import get_settings
from src.services.kendra_service import QueryLimitExceededError, get_ncct_ids_by_product
from src.utils.logging import get_logger

logger = get_logger(__name__)


class KendraSearchError(Exception):
    """Raised when Kendra search operations fail."""
    pass


class QueryProcessingError(Exception):
    """Raised when query processing fails."""
    pass


class HybridIBTAgent:
    """IBT agent that resolves benefit queries through direct Kendra search."""

    def __init__(self):
        self.settings = get_settings()
        self.kendra_index_id = self.settings.kendra_index_id
        self.aws_region = self.settings.aws_region

    def process_query(self, user_prompt: str, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        if context:
            logger.info(
                "Processing query with context: userName=%s, userType=%s, productId=%s, source=%s",
                context.get('userName'),
                context.get('userType'),
                context.get('productId'),
                context.get('source'),
            )
        else:
            logger.info("Processing query without context")

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
                "mode": "direct_kendra",
            }
            logger.info(
                "IBT Agent response: session_id=%s, mode=%s, success=%s, confidence=%s, responseText=%s",
                response["sessionId"],
                response["mode"],
                response["success"],
                response["confidence"],
                response["responseText"],
            )
            return response

        except QueryLimitExceededError:
            logger.warning("Query limit exceeded for session %s", session_id)
            response = {
                "sessionId": session_id,
                "confidence": 0.0,
                "responseText": get_message("query_limit_exceeded"),
                "success": False,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "direct_kendra",
            }
            logger.info(
                "IBT Agent query limit response: session_id=%s, responseText=%s",
                response["sessionId"],
                response["responseText"],
            )
            return response
        except (KendraSearchError, RuntimeError) as e:
            logger.error("Query processing error: %s", str(e))
            response = {
                "sessionId": session_id,
                "confidence": 0.0,
                "responseText": get_message("service_unavailable"),
                "success": False,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "direct_kendra",
            }
            logger.info(
                "IBT Agent error response: session_id=%s, mode=%s, success=%s, responseText=%s",
                response["sessionId"],
                response["mode"],
                response["success"],
                response["responseText"],
            )
            return response

    def _get_required_product_id(self, context: Dict[str, Any]) -> str:
        """Extract the orchestrator-provided context.productId without a default fallback."""
        if not context:
            raise QueryProcessingError("context.productId is required")

        product_id = context.get('productId')
        if not isinstance(product_id, str) or not product_id.strip():
            raise QueryProcessingError("context.productId is required")

        return product_id.strip()

    def _process_direct_kendra(self, user_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        product_id = self._get_required_product_id(context)
        logger.info("Using product ID from context: %s", product_id)
        logger.info("Processing direct Kendra query with product ID: %s", product_id)

        ncct_ids = get_ncct_ids_by_product(user_prompt, str(product_id))
        logger.info("Product %s filtering applied, found %s results", product_id, len(ncct_ids))

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
                "ncct_count": 0,
            }

        logger.info("Found %s unique NCCT IDs: %s", len(unique_ncct_ids), unique_ncct_ids)

        return {
            "success": True,
            "response_text": unique_ncct_ids,
            "confidence": 8.0,
            "product_id": product_id,
            "ncct_count": len(unique_ncct_ids),
        }
