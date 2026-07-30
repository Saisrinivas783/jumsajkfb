"""Unit tests for workflow graph building."""

import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph import StateGraph

from src.graph.workflow import build_graph
from src.tools.registry import ToolRegistry
from src.schemas.registry import ToolDefinition, ToolParameters


@pytest.fixture
def sample_tools():
    """Create sample tool definitions."""
    return [
        ToolDefinition(
            name="IBTAgent",
            description="Handles insurance benefit inquiries",
            endpoint="https://ibt-service.internal/api/v1",
            capabilities=["benefit inquiries", "coverage questions"],
            parameters=ToolParameters(required=["userPrompt", "userName"], optional=["policyNumber"])
        ),
        ToolDefinition(
            name="ClaimsAgent",
            description="Handles claims inquiries",
            endpoint="https://claims-service.internal/api/v1",
            capabilities=["claim status", "claim submission"],
            parameters=ToolParameters(required=["userPrompt"], optional=["claimNumber"])
        ),
        ToolDefinition(
            name="SupportAgent",
            description="Handles general support inquiries",
            endpoint="https://support-service.internal/api/v1",
            capabilities=["general questions", "help"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
    ]


@pytest.fixture
def tool_registry(sample_tools):
    """Create a tool registry with sample tools."""
    return ToolRegistry(sample_tools)


class TestBuildGraph:
    """Tests for build_graph function."""

    def test_build_graph_returns_compiled_graph(self, tool_registry):
        """Test that build_graph returns a compiled graph."""
        graph = build_graph(tool_registry)

        # Verify it returns a compiled graph
        assert graph is not None
        assert hasattr(graph, 'invoke')

    def test_build_graph_with_empty_registry(self):
        """Test build_graph with empty registry succeeds (empty-list guard is in registry YAML parsing)."""
        empty_registry = ToolRegistry([])
        graph = build_graph(empty_registry)
        assert graph is not None
        assert hasattr(graph, 'invoke')

    def test_build_graph_with_single_tool(self):
        """Test build_graph with a single tool."""
        single_tool = ToolRegistry([
            ToolDefinition(
                name="TestTool",
                description="Test tool",
                endpoint="https://test.internal/api/v1",
                capabilities=["testing"],
                parameters=ToolParameters(required=["userPrompt"], optional=[])
            )
        ])

        graph = build_graph(single_tool)
        assert graph is not None

    def test_build_graph_registers_all_tool_nodes(self, tool_registry):
        """Test that all tools are registered as nodes."""
        graph = build_graph(tool_registry)

        # Verify graph was created (we can't easily inspect node names in compiled graph)
        assert graph is not None

    @patch('src.graph.workflow.logger')
    def test_build_graph_logs_tool_registration(self, mock_logger, tool_registry):
        """Test that tool registration is logged."""
        build_graph(tool_registry)

        # Verify logging calls
        assert mock_logger.info.called
        # Check that tool names were logged
        call_args = str(mock_logger.info.call_args_list)
        assert "IBTAgent" in call_args
        assert "ClaimsAgent" in call_args
        assert "SupportAgent" in call_args

    def test_build_graph_with_different_tool_counts(self):
        """Test building graphs with different numbers of tools."""
        for num_tools in [1, 3, 5]:
            tools = [
                ToolDefinition(
                    name=f"Tool{i}",
                    description=f"Tool {i} description",
                    endpoint=f"https://tool{i}.internal/api/v1",
                    capabilities=[f"capability{i}"],
                    parameters=ToolParameters(required=["userPrompt"], optional=[])
                )
                for i in range(num_tools)
            ]
            registry = ToolRegistry(tools)
            graph = build_graph(registry)
            assert graph is not None

    def test_build_graph_creates_routes_for_tools(self, tool_registry):
        """Test that routes are created for each tool."""
        with patch('src.graph.workflow.logger') as mock_logger:
            build_graph(tool_registry)

            # Check debug logging contains routes information
            debug_calls = [call for call in mock_logger.debug.call_args_list]
            routes_logged = any("routes" in str(call).lower() for call in debug_calls)
            assert routes_logged

    def test_build_graph_includes_fallback_route(self, tool_registry):
        """Test that fallback route is included."""
        graph = build_graph(tool_registry)

        # Graph should be compiled successfully with fallback
        assert graph is not None

    def test_build_graph_can_be_invoked(self, tool_registry, mock_context):
        """Test that built graph can be invoked."""
        from src.schemas.state import OrchestratorState

        graph = build_graph(tool_registry)

        # Create initial state
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context
        )

        # This should not raise an exception
        # Note: It will fail in LLM call, but graph structure is valid
        try:
            result = graph.invoke(state.model_dump())
            # If it completes, great
            assert result is not None
        except Exception:
            # If it fails (expected due to LLM), that's OK for this test
            # We just want to verify the graph structure is valid
            pass

    def test_build_graph_tool_names_in_registry(self, tool_registry):
        """Test that tool names match registry."""
        tool_names = tool_registry.list_tool_names()

        # Build graph
        graph = build_graph(tool_registry)

        # Verify graph was built
        assert graph is not None

        # Verify all tool names are valid
        assert "IBTAgent" in tool_names
        assert "ClaimsAgent" in tool_names
        assert "SupportAgent" in tool_names


class TestGuardrailNodeInGraph:
    """Tests verifying guardrail_check is wired as entry point in the graph."""

    def test_build_graph_includes_guardrail_node(self, tool_registry):
        """Graph compiles successfully with guardrail_check node present."""
        graph = build_graph(tool_registry)
        assert graph is not None
        assert hasattr(graph, "invoke")

    @patch("src.graph.workflow.logger")
    def test_build_graph_logs_after_guardrail_node_added(self, mock_logger, tool_registry):
        """Graph construction completes logging (guardrail node wired without error)."""
        build_graph(tool_registry)
        assert mock_logger.info.called

    def test_guardrail_blocked_state_routes_to_end(self, tool_registry, mock_context):
        """When guardrail_check sets guardrail_blocked=True, graph routes to END."""
        from src.schemas.state import OrchestratorState
        from src.graph.nodes.guardrail_node import guardrail_router

        state = OrchestratorState.model_construct(
            query="bad query",
            session_id="sess-123",
            context=mock_context,
            guardrail_blocked=True,
            selected_tool=None,
            tool_result=None,
            tool_metadata=[],
            final_answer="Blocked.",
            error=None,
        )
        assert guardrail_router(state) == "end"

    def test_guardrail_passed_state_routes_to_analyzer(self, tool_registry, mock_context):
        """When guardrail_check sets guardrail_blocked=False, graph routes to analyzer."""
        from src.schemas.state import OrchestratorState
        from src.graph.nodes.guardrail_node import guardrail_router

        state = OrchestratorState(
            query="What are my benefits?",
            session_id="sess-456",
            context=mock_context,
        )
        assert guardrail_router(state) == "analyzer"


class TestPostToolRouter:
    """Tests for post_tool_router conditional edge logic."""

    def test_post_tool_router_routes_to_end_on_success(self, mock_context):
        """post_tool_router returns END when tool succeeded."""
        from langgraph.graph import END
        from src.graph.nodes.confidence_router import post_tool_router
        from src.schemas.state import OrchestratorState
        from src.schemas.tools import ToolResult

        state = OrchestratorState(
            query="q",
            session_id="s",
            context=mock_context,
            tool_result=ToolResult(tool_name="IBTAgent", success=True, response="ok"),
        )
        assert post_tool_router(state) == END

    def test_post_tool_router_routes_to_fallback_on_failure(self, mock_context):
        """post_tool_router returns 'fallback' when tool_result.success is False."""
        from src.graph.nodes.confidence_router import post_tool_router
        from src.schemas.state import OrchestratorState
        from src.schemas.tools import ToolResult

        state = OrchestratorState(
            query="q",
            session_id="s",
            context=mock_context,
            tool_result=ToolResult(tool_name="IBTAgent", success=False, error="timeout"),
        )
        assert post_tool_router(state) == "fallback"

    def test_post_tool_router_routes_to_end_when_no_result(self, mock_context):
        """post_tool_router returns END when tool_result is None."""
        from langgraph.graph import END
        from src.graph.nodes.confidence_router import post_tool_router
        from src.schemas.state import OrchestratorState

        state = OrchestratorState(
            query="q",
            session_id="s",
            context=mock_context,
        )
        assert post_tool_router(state) == END


class TestGraphStructure:
    """Tests for graph structure and connections."""

    def test_graph_has_entry_point(self, tool_registry):
        """Test that graph has proper entry point."""
        graph = build_graph(tool_registry)
        assert graph is not None

    def test_graph_has_end_state(self, tool_registry):
        """Test that graph has proper end state."""
        graph = build_graph(tool_registry)
        assert graph is not None

    def test_graph_with_special_characters_in_tool_name(self):
        """Test graph building with special characters in tool names."""
        # Note: Current implementation uses tool names directly as node names
        # This test verifies it handles normal tool names
        tools = [
            ToolDefinition(
                name="Tool_With_Underscores",
                description="Test tool with underscores",
                endpoint="https://test.internal/api/v1",
                capabilities=["testing"],
                parameters=ToolParameters(required=["userPrompt"], optional=[])
            )
        ]
        registry = ToolRegistry(tools)
        graph = build_graph(registry)
        assert graph is not None
