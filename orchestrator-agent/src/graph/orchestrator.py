"""Orchestrator agent - main entry point for workflow execution."""

import time
from typing import Any

from src.exceptions import OrchestratorError
from src.schemas.api import (
    InvocationRequest,
    InvocationResponse,
    AgentMetadata,
    MetadataItem,
)
from src.schemas.state import OrchestratorState
from src.tools.registry import ToolRegistry
from src.graph.workflow import build_graph
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OrchestratorAgent:
    """Main orchestrator that runs the LangGraph workflow."""

    def __init__(self, registry_path: str = "src/tools/definitions/tools.yaml"):
        self.registry_path = registry_path
        # Load registry as ToolRegistry class for graph building
        self.registry = ToolRegistry.from_local_yaml(registry_path)
        # Build graph with registry (creates tool-specific nodes dynamically)
        self.graph_app = build_graph(self.registry)
        logger.info(f"OrchestratorAgent initialized with {len(self.registry)} tools: {self.registry.list_tool_names()}")

    def handle_invocation(self, payload: InvocationRequest, authorization: str | None = None) -> InvocationResponse:
        """Execute the workflow and return structured response."""
        start_time = time.time()
        logger.info(f"Invocation started: session={payload.session_id}")
        logger.debug(f"Query: {payload.user_prompt[:100]}{'...' if len(payload.user_prompt) > 100 else ''}")

        # Validation is handled at FastAPI layer via Pydantic validators
        # Build initial state
        state = OrchestratorState(
            query=payload.user_prompt,
            session_id=payload.session_id,
            context=payload.context,
            authorization=authorization,
        )

        # Run the graph - catch all exceptions and return graceful response
        try:
            logger.debug("Invoking LangGraph workflow")
            out_dict: dict[str, Any] = self.graph_app.invoke(state.model_dump())
            out_state = OrchestratorState(**out_dict)
            logger.debug("Workflow completed successfully")
        except OrchestratorError as e:
            logger.error(f"Workflow error: {e.message}")
            return self._error_response(
                payload.session_id,
                e.message,
                start_time,
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return self._error_response(
                payload.session_id,
                "An unexpected error occurred. Please try again later.",
                start_time,
            )

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Build metadata array (preserved even on tool errors)
        metadata = self._build_metadata(out_state)

        # Determine success — state.error is the authoritative error source
        tool_failed = out_state.error is not None
        success = not tool_failed
        message = out_state.error.message if out_state.error else ""

        # Log completion
        tool_name = out_state.selected_tool.tool_name if out_state.selected_tool else 'none'
        confidence = out_state.selected_tool.confidence if out_state.selected_tool else 0.0

        logger.info(
            f"Invocation completed: session={payload.session_id}, "
            f"tool={tool_name}, "
            f"confidence={confidence:.1f}, "
            f"success={success}, "
            f"time={execution_time_ms:.0f}ms"
        )

        return InvocationResponse(
            sessionId=payload.session_id,
            responseText=out_state.final_answer if out_state.final_answer is not None else [],
            metadata=metadata,
            success=success,
            message=message or "",
            execution_time_ms=execution_time_ms,
        )

    def _build_metadata(self, state: OrchestratorState) -> list[AgentMetadata]:
        """
        Build metadata array from orchestrator state.

        Aggregates orchestrator metadata and tool-specific metadata.
        """
        metadata: list[AgentMetadata] = []

        # Orchestrator metadata
        orchestrator_data = []
        if state.selected_tool:
            orchestrator_data.append(
                MetadataItem(key="confidence", value=state.selected_tool.confidence)
            )
            orchestrator_data.append(
                MetadataItem(key="selectedTool", value=state.selected_tool.tool_name)
            )
            if state.selected_tool.reasoning:
                orchestrator_data.append(
                    MetadataItem(key="reasoning", value=state.selected_tool.reasoning)
                )
            orchestrator_data.append(
                MetadataItem(key="reformulatedQuery", value=state.selected_tool.reformulated_query)
            )

        if state.input_tokens is not None:
            orchestrator_data.append(
                MetadataItem(key="inputTokens", value=state.input_tokens)
            )
        if state.output_tokens is not None:
            orchestrator_data.append(
                MetadataItem(key="outputTokens", value=state.output_tokens)
            )
        if state.total_tokens is not None:
            orchestrator_data.append(
                MetadataItem(key="totalTokens", value=state.total_tokens)
            )

        orchestrator_data.append(
            MetadataItem(key="guardrailAction", value=state.guardrail_action)
        )
        orchestrator_data.append(
            MetadataItem(key="guardrailBlocked", value=state.guardrail_blocked)
        )

        metadata.append(AgentMetadata(agent="orchestrator", data=orchestrator_data))

        # Tool-specific metadata (populated by tool nodes)
        if state.tool_metadata:
            metadata.extend(state.tool_metadata)

        return metadata

    def _error_response(
        self,
        session_id: str,
        message: str,
        start_time: float,
    ) -> InvocationResponse:
        """Build an error response with success=False."""
        execution_time_ms = (time.time() - start_time) * 1000
        return InvocationResponse(
            sessionId=session_id,
            responseText=["I'm currently experiencing technical difficulties. Please try again in a few moments or contact support if the issue persists."],
            metadata=[],  # Empty array on error
            success=False,
            message=message,
            execution_time_ms=execution_time_ms,
        )
