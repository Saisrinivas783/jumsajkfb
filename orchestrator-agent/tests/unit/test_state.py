"""Unit tests for OrchestratorState schema."""

import pytest
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool, ToolResult, ErrorInfo
from src.schemas.api import InvocationContext


@pytest.fixture
def mock_context():
    """Fixture for test context."""
    return InvocationContext(
        userName="test_user",
        userType="member",
        source="TestPage",
        productId="PROD-001"
    )


class TestOrchestratorState:
    """Tests for OrchestratorState schema."""

    def test_state_initialization_minimal(self, mock_context):
        """Test minimal state initialization."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context
        )

        assert state.query == "Test query"
        assert state.session_id == "test-123"
        assert state.context.userName == "test_user"
        assert state.selected_tool is None
        assert state.tool_result is None
        assert state.final_answer is None
        assert state.error is None

    def test_state_initialization_with_context(self):
        """Test state initialization with context."""
        context = InvocationContext(
            userName="john_doe",
            userType="member",
            source="DXAIService",
            promptId="p-123",
            productId="PROD-001"
        )

        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=context
        )

        assert state.context.userName == "john_doe"
        assert state.context.userType == "member"

    def test_state_with_selected_tool(self, mock_context):
        """Test state with selected tool."""
        selected_tool = SelectedTool(
            tool_name="TestTool",
            confidence=9.0,
            reasoning="Test reasoning",
        )

        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context,
            selected_tool=selected_tool
        )

        assert state.selected_tool.tool_name == "TestTool"
        assert state.selected_tool.confidence == 9.0

    def test_state_with_tool_result(self, mock_context):
        """Test state with tool result."""
        tool_result = ToolResult(
            tool_name="TestTool",
            success=True,
            response="Test response"
        )

        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context,
            tool_result=tool_result
        )

        assert state.tool_result.tool_name == "TestTool"
        assert state.tool_result.success is True

    def test_state_with_final_answer(self, mock_context):
        """Test state with final answer."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context,
            final_answer=["This is the final answer"]
        )

        assert state.final_answer == ["This is the final answer"]

    def test_state_model_dump(self, mock_context):
        """Test state serialization."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context
        )

        dumped = state.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["query"] == "Test query"
        assert dumped["session_id"] == "test-123"

    def test_state_model_dump_json(self, mock_context):
        """Test state JSON serialization."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context
        )

        json_str = state.model_dump_json()

        assert isinstance(json_str, str)
        assert "Test query" in json_str
        assert "test-123" in json_str

    def test_state_update_fields(self, mock_context):
        """Test updating state fields."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context
        )

        # Update fields
        state.final_answer = "Updated answer"
        state.selected_tool = SelectedTool(
            tool_name="NewTool",
            confidence=8.5,
            reasoning="New reasoning"
        )

        assert state.final_answer == "Updated answer"
        assert state.selected_tool.tool_name == "NewTool"

    def test_state_with_error_info_is_serializable(self, mock_context):
        """ErrorInfo stored in state must serialize without TypeError."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context,
            error=ErrorInfo(error_type="tool_timeout", message="timed out", tool_name="IBTAgent"),
        )

        # Must not raise TypeError
        json_str = state.model_dump_json()
        assert "tool_timeout" in json_str
        assert "IBTAgent" in json_str

    def test_state_error_defaults_to_none(self, mock_context):
        """State error field defaults to None."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context,
        )
        assert state.error is None