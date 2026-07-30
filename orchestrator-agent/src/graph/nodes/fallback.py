"""Fallback node - handles non-tool responses."""

from typing import Any

from src.schemas.state import OrchestratorState
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Pre-defined response messages
# --- Original messages (commented out for revert) ---
# FALLBACK_MESSAGES = {
#     "no_tool_found": "I'm sorry, I couldn't find the right resource to help with your question. Please try rephrasing your query or contact our support team for assistance.",
#     "low_confidence": "I'm not entirely sure I understand your question. Could you please provide more details or rephrase your request?",
#     "service_unavailable": "I'm currently experiencing technical difficulties. Please try again in a few moments or contact support if the issue persists.",
# }
# --- End original messages ---

_FALLBACK_MESSAGE = [
    "Your search did not return any results. Please refer to the "
    "<a href='https://www.fepblue.org/plan-brochures' target='_blank'>Blue Cross and Blue Shield Service Benefit Plan brochure</a> "
    "or to the "
    "<a href='https://www.fepblue.org/plan-summaries' target='_blank'>Health Plan Summaries</a>."
]

FALLBACK_MESSAGES = {
    "no_tool_found": _FALLBACK_MESSAGE,
    "low_confidence": _FALLBACK_MESSAGE,
    "service_unavailable": _FALLBACK_MESSAGE,
}


def fallback_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Handles fallback scenarios when no tool can be executed.

    Routes here when:
    - Tool execution failed (routed from post_tool_router)
    - NO_TOOL selected (including out-of-scope queries)
    - Low confidence score
    """
    logger.info("→ fallback")

    # Tool execution failed — routed here from post_tool_router
    if state.error is not None:
        logger.info(f"← fallback: service_unavailable (error_type={state.error.error_type})")
        return {"final_answer": FALLBACK_MESSAGES["service_unavailable"]}

    if not state.selected_tool:
        logger.debug("No tool in state - using service_unavailable message")
        logger.info("← fallback: service_unavailable")
        return {"final_answer": FALLBACK_MESSAGES["service_unavailable"]}

    tool_name = state.selected_tool.tool_name
    confidence = state.selected_tool.confidence

    logger.debug(f"Fallback context: tool={tool_name}, confidence={confidence:.1f}")

    # No tool match (includes out-of-scope and conversational queries)
    if tool_name == "NO_TOOL":
        logger.info("← fallback: no_tool_found")
        return {"final_answer": FALLBACK_MESSAGES["no_tool_found"]}

    # Low confidence
    logger.info(f"← fallback: low_confidence ({confidence:.1f})")
    return {"final_answer": FALLBACK_MESSAGES["low_confidence"]}