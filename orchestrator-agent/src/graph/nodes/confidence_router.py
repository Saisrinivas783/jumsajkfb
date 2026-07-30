"""Confidence-based routing node.

Routes to specific tool nodes or fallback based on confidence scores.
This enables explicit tool nodes in the graph for better traceability.
"""

from langgraph.graph import END

from src.config.settings import orchestrator_settings
from src.schemas.state import OrchestratorState
from src.utils.logging import get_logger

logger = get_logger(__name__)


def guard_rails_router(state: OrchestratorState) -> str:
    """
    Routes to specific tool node or fallback.

    Returns:
        - Tool name (e.g., "IBTAgent") for high confidence matches
        - "fallback" for low confidence or no tool match

    This routing function returns the actual tool name, which maps directly
    to a tool-specific node in the graph. This provides:
    - Explicit visibility of which tool runs in LangGraph traces
    - Direct routing without intermediate generic executors
    - Automatic support for new tools via registry
    """
    logger.debug("→ guard_rails_router")

    if not state.selected_tool:
        logger.info("Guard Rails: No tool selected → fallback")
        return "fallback"

    tool_name = state.selected_tool.tool_name
    confidence = state.selected_tool.confidence

    logger.info(f"Guard Rails - Tool: {tool_name}, Confidence: {confidence:.1f}")

    # Route to fallback for NO_TOOL
    if tool_name == "NO_TOOL":
        logger.info("Guard Rails: No tool match → fallback")
        return "fallback"

    # Route to fallback for low confidence
    threshold = orchestrator_settings.confidence_threshold_high
    if confidence < threshold:
        logger.info(f"Guard Rails: Confidence {confidence:.1f} < {threshold} → fallback")
        return "fallback"

    # Return tool name as route target (maps to tool-specific node)
    logger.info(f"Guard Rails: PASSED → {tool_name}")
    return tool_name


def post_tool_router(state: OrchestratorState) -> str:
    """Route to fallback if tool failed, otherwise end."""
    if state.tool_result and not state.tool_result.success:
        logger.info("post_tool_router: tool failed → fallback")
        return "fallback"
    return END