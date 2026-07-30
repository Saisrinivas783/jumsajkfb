"""Langchain tools for IBT agent."""

from langchain.tools import tool
from typing import List, Dict, Any
from src.services.kendra_service import get_kendra_service
from src.config.messages import get_message
from src.utils.logging import get_logger

logger = get_logger(__name__)

@tool
def search_benefits(query: str) -> str:
    """Search for insurance benefits and coverage information.
    
    Args:
        query: User's benefits inquiry
        
    Returns:
        HTML formatted links to relevant benefits
    """
    logger.info(f"Searching benefits for query: {query[:50]}...")
    
    try:
        kendra_service = get_kendra_service()
        search_result = kendra_service.search(query)
        
        if not search_result.get('success', False):
            logger.error(f"Service error: {search_result.get('error', 'Unknown error')}")
            msg = get_message("service_unavailable")
            return msg[0] if isinstance(msg, list) else msg
        
        if not search_result.get('results'):
            logger.warning("No results found for query")
            msg = get_message("no_results_found")
            return msg[0] if isinstance(msg, list) else msg
        
        results = search_result['results']
        links = []
        for result in results:
            ncct_id = result['ncct_id']
            service_name = result['service_name']
            if ncct_id:
                links.append(f"<a href='{ncct_id}'>{service_name}</a>")
            else:
                links.append(service_name)
        
        logger.info(f"Found {len(results)} benefits results")
        return "Here are the relevant benefits: " + ", ".join(links)
        
    except Exception as e:
        logger.error(f"Benefits search failed: {str(e)}")
        msg = get_message("service_unavailable")
        return msg[0] if isinstance(msg, list) else msg
