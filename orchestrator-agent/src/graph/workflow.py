"""LangGraph workflow definition with dynamic tool routing.

This module builds the orchestrator workflow graph with explicit tool nodes.
Each tool from the registry becomes its own named node in the graph, enabling:
- Better traceability in LangGraph traces (shows "IBTAgent" not "tool_executor")
- Automatic extensibility via the tool registry
- Independent testing of each tool node
"""

from langgraph.graph import StateGraph, START, END

from src.schemas.state import OrchestratorState
from src.graph.nodes.intent_analyzer import create_intent_node
from src.graph.nodes.confidence_router import guard_rails_router, post_tool_router
from src.graph.nodes.fallback import fallback_node
from src.graph.nodes.tool_node_factory import create_tool_node
from src.graph.nodes.guardrail_node import guardrail_check_node, guardrail_router
from src.tools.registry import ToolRegistry
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_graph(registry: ToolRegistry):
    """
    Build the orchestrator workflow graph with dynamic tool nodes.

    The graph structure is:
    ```
    START
      │
      ▼
    ┌──────────────────┐
    │  guardrail_check │  ← AWS Bedrock Guardrails (skipped if not configured)
    └────────┬─────────┘
             │ blocked → END
             │ passed  ↓
    ┌─────────────┐
    │  analyzer   │  ← Intent detection + tool selection
    └──────┬──────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │         guard_rails (router)         │
    │  Returns: tool_name | "fallback"     │
    └───────┬───────────┬──────────────────┘
            │           │
       ┌────┴────┐ ┌────┴────┐ ┌──────────┐
       ▼         ▼ ▼         ▼ ▼          ▼
    ┌──────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐
    │fall- │  │IBTAgent  │  │Claims-   │  │(future      │
    │back  │  │          │  │Agent     │  │ tools...)   │
    └──┬───┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘
       │           │             │               │
       └───────────┴─────────────┴───────────────┘
                          │
                          ▼
                         END
    ```

    Args:
        registry: Tool registry containing all available tools

    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(OrchestratorState)

    # === Core Nodes ===
    registry_dict = {tool.name: tool for tool in registry}
    intent_node = create_intent_node(registry_dict)
    workflow.add_node("guardrail_check", guardrail_check_node)
    workflow.add_node("analyzer", intent_node)
    workflow.add_node("fallback", fallback_node)

    # === Dynamic Tool Nodes ===
    # Build routes map: router_return_value → node_name
    routes = {"fallback": "fallback"}

    tool_names = []
    for tool_def in registry:
        tool_name = tool_def.name

        # Create node function for this tool
        node_fn = create_tool_node(tool_def)

        # Register node with tool name (e.g., "IBTAgent")
        workflow.add_node(tool_name, node_fn)

        # Route to fallback on failure, END on success
        workflow.add_conditional_edges(
            tool_name,
            post_tool_router,
            {"fallback": "fallback", END: END},
        )

        # Add route: "IBTAgent" → "IBTAgent" node
        routes[tool_name] = tool_name
        tool_names.append(tool_name)

    logger.info(f"Registered tool nodes: {tool_names}")

    # === Entry Point ===
    workflow.set_entry_point("guardrail_check")

    # === Guardrail Routing ===
    # guardrail_router returns "end" (blocked) or "analyzer" (passed)
    workflow.add_conditional_edges(
        "guardrail_check",
        guardrail_router,
        {"end": END, "analyzer": "analyzer"}
    )

    # === Conditional Routing ===
    # guard_rails_router returns tool name or "fallback"
    workflow.add_conditional_edges(
        "analyzer",
        guard_rails_router,
        routes
    )

    # === Terminal Edges ===
    workflow.add_edge("fallback", END)

    logger.debug(f"Graph built with routes: {routes}")

    return workflow.compile()
