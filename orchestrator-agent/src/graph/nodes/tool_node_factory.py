"""Factory for creating tool-specific graph nodes.

This module provides a factory function that creates dedicated LangGraph nodes
for each tool in the registry. This enables:
- Explicit tool nodes in the graph (e.g., "IBTAgent" instead of "tool_executor")
- Better traceability in LangGraph traces
- Independent testing of each tool node
- Automatic registration of new tools from YAML config
"""

from typing import Any, Callable

import httpx

from src.schemas.state import OrchestratorState
from src.schemas.tools import ToolResult, ErrorInfo
from src.schemas.registry import ToolDefinition
from src.schemas.api import AgentMetadata, MetadataItem
from src.config.settings import get_settings
from src.exceptions import ToolTimeoutError, ToolUnavailableError
from src.http_client import get_http_client
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ============================================================
# MOCK RESPONSES
# Comment out this block when switching to real HTTP calls.
# Each key is the tool name (must match tools.yaml).
# ============================================================
MOCK_RESPONSES: dict[str, dict] = {
    "IBTAgent": {
        "responseText": (
            "Based on your insurance plan, here are your dental benefits:\n\n"
            "- **Preventive Care** (cleanings, exams, X-rays): Covered at 100% in-network, no deductible.\n"
            "- **Basic Services** (fillings, extractions): Covered at 80% after $50 deductible.\n"
            "- **Major Services** (crowns, root canals): Covered at 50% after deductible.\n"
            "- **Orthodontia**: Covered at 50% up to $1,500 lifetime maximum.\n\n"
            "Annual maximum benefit: $2,000 per member. Network: Delta Dental PPO."
        ),
        "metadata": [
            {
                "agent": "IBTAgent",
                "data": [
                    {"key": "source", "value": "benefits_catalog_v3"},
                    {"key": "plan_type", "value": "PPO"},
                    {"key": "mock", "value": True},
                ],
            }
        ],
    },
}
# ============================================================


def create_tool_node(tool_def: ToolDefinition) -> Callable[[OrchestratorState], dict[str, Any]]:
    """
    Factory function that creates a node for a specific tool.

    Args:
        tool_def: Tool definition from registry

    Returns:
        A node function that executes this specific tool
    """
    tool_name = tool_def.name
    endpoint = str(tool_def.endpoint)

    def tool_node(state: OrchestratorState) -> dict[str, Any]:
        """Execute the tool and update state with results."""
        logger.info(f"→ {tool_name}_node")

        # Determine which query to send based on USE_REFORMULATED_QUERY setting
        settings = get_settings()
        reformulated = state.selected_tool.reformulated_query if state.selected_tool else None
        if settings.use_reformulated_query and reformulated:
            effective_query = reformulated
            logger.info(f"Using reformulated query: {effective_query!r} (original: {state.query!r})")
        else:
            effective_query = state.query
            if not settings.use_reformulated_query:
                logger.info(f"USE_REFORMULATED_QUERY=False; sending original query: {effective_query!r}")
            else:
                logger.debug(f"No reformulated query; using original: {effective_query!r}")

        try:
            # ── MOCK (inactive) ← uncomment when switching back to mock ──
            # response_text, agent_metadata = _call_tool_mock(tool_name, effective_query)
            # ──────────────────────────────────────────────────────────────

            # ── HTTP (active) ──
            response_text, agent_metadata = _call_tool_api(tool_name, endpoint, state, effective_query)
            # ──────────────────────────────────────────────────────────────

            logger.info(f"← {tool_name}_node: success")
            logger.debug(f"Response count: {len(response_text)} items")

            return {
                "tool_result": ToolResult(tool_name=tool_name, success=True, response=response_text),
                "final_answer": response_text,
                "error": None,
                "tool_metadata": agent_metadata,
            }

        except ToolTimeoutError as e:
            logger.error(f"Tool {tool_name} timeout: {e.message}")
            return {
                "error": ErrorInfo(error_type="tool_timeout", message=e.message, tool_name=tool_name),
                "tool_result": ToolResult(tool_name=tool_name, success=False, error=e.message),
            }

        except ToolUnavailableError as e:
            logger.error(f"Tool {tool_name} unavailable: {e.message}")
            return {
                "error": ErrorInfo(error_type="tool_unavailable", message=e.message, tool_name=tool_name),
                "tool_result": ToolResult(tool_name=tool_name, success=False, error=e.message),
            }

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return {
                "error": ErrorInfo(error_type="unknown", message=str(e), tool_name=tool_name),
                "tool_result": ToolResult(tool_name=tool_name, success=False, error=str(e)),
            }

    tool_node.__name__ = f"{tool_name}_node"
    tool_node.__doc__ = f"Execute {tool_name} and update state with results."

    return tool_node


# ============================================================
# MOCK implementation
# Uncomment this function when switching back to mock responses.
# ============================================================
# def _call_tool_mock(tool_name: str, effective_query: str = "") -> tuple[str, list]:
#     """Return a hardcoded mock response for the given tool."""
#     logger.info(f"[MOCK] Returning mock response for {tool_name} (query: {effective_query!r})")
#
#     mock = MOCK_RESPONSES.get(tool_name)
#     if not mock:
#         raise ToolUnavailableError(tool_name, f"No mock response defined for '{tool_name}'")
#
#     response_text = mock["responseText"]
#     raw_metadata = mock.get("metadata", [])
#     agent_metadata = [AgentMetadata.model_validate(m) for m in raw_metadata]
#
#     return response_text, agent_metadata
# ============================================================


# ============================================================
# HTTP implementation
# ============================================================
def _call_tool_api(
    tool_name: str,
    endpoint: str,
    state: OrchestratorState,
    effective_query: str = "",
) -> tuple[list[str], list]:
    """
    Call the tool's HTTP API.

    Args:
        tool_name: Name of the tool to call
        endpoint: HTTP endpoint URL
        state: Current orchestrator state
        effective_query: Reformulated query (or original query as fallback)

    Returns:
        Tuple of (response_text, agent_metadata_list)

    Raises:
        ToolTimeoutError: If the tool times out
        ToolUnavailableError: If the tool is unavailable
    """
    settings = get_settings()

    logger.info(f"Calling {tool_name} at {endpoint}")

    try:
        client = get_http_client()
        payload = {
            "userPrompt": effective_query or state.query,
            "sessionId": state.session_id,
            "context": {
                "userName": state.context.userName,
                "userType": state.context.userType,
                "source": state.context.source,
                "productId": state.context.productId,
            },
        }
        if state.context.promptId:
            payload["context"]["promptId"] = state.context.promptId

        headers = {"Content-Type": "application/json"}
        if state.authorization:
            headers["Authorization"] = state.authorization

        response = client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        logger.info("Received response from %s: %s", tool_name, data)
        response_text = data.get("responseText", "")
        raw_metadata = data.get("metadata", [])
        agent_metadata = [AgentMetadata.model_validate(m) for m in raw_metadata]

        return response_text, agent_metadata

    except httpx.TimeoutException:
        raise ToolTimeoutError(tool_name, settings.tool_timeout)
    except httpx.HTTPStatusError as e:
        raise ToolUnavailableError(tool_name, f"HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise ToolUnavailableError(tool_name, str(e))
# ============================================================
