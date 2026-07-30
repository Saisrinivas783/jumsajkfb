"""Unit tests for tool_node_factory."""

import pytest
from unittest.mock import patch, MagicMock
from src.graph.nodes.tool_node_factory import create_tool_node
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool, ToolResult, ErrorInfo
from src.schemas.registry import ToolDefinition, ToolParameters
from src.schemas.api import InvocationContext, AgentMetadata, MetadataItem
from src.exceptions import ToolTimeoutError, ToolUnavailableError


@pytest.fixture
def ibt_tool_def():
    """Create IBTAgent tool definition."""
    return ToolDefinition(
        name="IBTAgent",
        description="Handles insurance benefit inquiries",
        endpoint="http://localhost:8001/invocations",
        capabilities=["benefit inquiries", "coverage questions"],
        parameters=ToolParameters(required=["userPrompt", "userName"], optional=["policyNumber"])
    )


@pytest.fixture
def claims_tool_def():
    """Create ClaimsAgent tool definition."""
    return ToolDefinition(
        name="ClaimsAgent",
        description="Handles claims inquiries",
        endpoint="https://claims-service.internal/api/v1",
        capabilities=["claim status", "claim submission"],
        parameters=ToolParameters(required=["userPrompt"], optional=["claimNumber"])
    )


@pytest.fixture
def mock_state_with_tool(ibt_tool_def):
    """Create a mock state with selected tool."""
    return OrchestratorState(
        query="What are my dental benefits?",
        session_id="test-session-123",
        context=InvocationContext(
            userName="john_doe",
            userType="member",
            source="DXAIService",
            promptId="p-123",
            productId="PROD-001"
        ),
        selected_tool=SelectedTool(
            tool_name="IBTAgent",
            confidence=9.0,
            reasoning="Benefit inquiry",
            reformulated_query="dental coverage benefits",
        )
    )


class TestCreateToolNode:
    """Tests for create_tool_node factory function."""

    def test_create_tool_node_returns_callable(self, ibt_tool_def):
        """Test that factory returns a callable function."""
        tool_node = create_tool_node(ibt_tool_def)

        assert callable(tool_node)
        assert tool_node.__name__ == "IBTAgent_node"
        assert "IBTAgent" in tool_node.__doc__

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_success(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test tool node execution with mocked response."""
        mock_call.return_value = ("Your dental benefits include preventive care.", [])
        tool_node = create_tool_node(ibt_tool_def)

        result = tool_node(mock_state_with_tool)

        assert result["tool_result"] is not None
        assert result["tool_result"].success is True
        assert result["tool_result"].tool_name == "IBTAgent"
        assert result["final_answer"] == "Your dental benefits include preventive care."
        assert result["error"] is None

    def test_tool_node_different_tools(self, ibt_tool_def, claims_tool_def):
        """Test that different tools create different nodes."""
        ibt_node = create_tool_node(ibt_tool_def)
        claims_node = create_tool_node(claims_tool_def)

        assert ibt_node.__name__ == "IBTAgent_node"
        assert claims_node.__name__ == "ClaimsAgent_node"
        assert ibt_node != claims_node

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_handles_timeout_error(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test tool node sets ErrorInfo on timeout; no final_answer on error."""
        tool_node = create_tool_node(ibt_tool_def)
        mock_call.side_effect = ToolTimeoutError("IBTAgent", 30.0)

        result = tool_node(mock_state_with_tool)

        assert result["error"] is not None
        assert isinstance(result["error"], ErrorInfo)
        assert result["error"].error_type == "tool_timeout"
        assert result["error"].tool_name == "IBTAgent"
        assert result["tool_result"] is not None
        assert result["tool_result"].success is False
        assert "final_answer" not in result

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_handles_unavailable_error(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test tool node sets ErrorInfo on unavailable; no final_answer on error."""
        tool_node = create_tool_node(ibt_tool_def)
        mock_call.side_effect = ToolUnavailableError("IBTAgent", "Service down")

        result = tool_node(mock_state_with_tool)

        assert result["error"] is not None
        assert isinstance(result["error"], ErrorInfo)
        assert result["error"].error_type == "tool_unavailable"
        assert result["error"].tool_name == "IBTAgent"
        assert result["tool_result"].success is False
        assert "final_answer" not in result

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_handles_generic_exception(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test tool node sets ErrorInfo on generic exception; no final_answer on error."""
        tool_node = create_tool_node(ibt_tool_def)
        mock_call.side_effect = Exception("Unexpected error")

        result = tool_node(mock_state_with_tool)

        assert isinstance(result["error"], ErrorInfo)
        assert result["error"].error_type == "unknown"
        assert result["tool_result"].success is False
        assert "final_answer" not in result

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_uses_agent_metadata_from_response(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test that metadata from tool response is forwarded as-is."""
        agent_meta = AgentMetadata(
            agent="ibt",
            data=[MetadataItem(key="resultCount", value=5)]
        )
        mock_call.return_value = ("Benefits text.", [agent_meta])
        tool_node = create_tool_node(ibt_tool_def)

        result = tool_node(mock_state_with_tool)

        assert result["tool_metadata"] == [agent_meta]

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_empty_metadata_when_agent_returns_none(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test that empty metadata from tool results in empty tool_metadata."""
        mock_call.return_value = ("Benefits text.", [])
        tool_node = create_tool_node(ibt_tool_def)

        result = tool_node(mock_state_with_tool)

        assert result["tool_metadata"] == []

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_uses_reformulated_query_as_effective_query(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test that query selection respects USE_REFORMULATED_QUERY setting."""
        from src.config.settings import get_settings
        mock_call.return_value = ("Benefits text.", [])
        tool_node = create_tool_node(ibt_tool_def)

        tool_node(mock_state_with_tool)

        settings = get_settings()
        expected_query = "dental coverage benefits" if settings.use_reformulated_query else "What are my dental benefits?"
        mock_call.assert_called_once_with("IBTAgent", "http://localhost:8001/invocations", mock_state_with_tool, expected_query)

    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    def test_tool_node_falls_back_to_original_query_when_no_reformulation(self, mock_call, ibt_tool_def):
        """Test that original query is used as fallback when reformulated_query is None."""
        mock_call.return_value = ("Benefits text.", [])
        state = OrchestratorState(
            query="what is my teeths coverage",
            session_id="test-session-456",
            context=InvocationContext(
                userName="jane_doe",
                userType="member",
                source="DXAIService",
                promptId="p-456",
                productId="PROD-002"
            ),
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=8.0,
                reasoning="Dental query",
                reformulated_query=None,
            )
        )
        tool_node = create_tool_node(ibt_tool_def)

        tool_node(state)

        mock_call.assert_called_once_with("IBTAgent", "http://localhost:8001/invocations", state, "what is my teeths coverage")

