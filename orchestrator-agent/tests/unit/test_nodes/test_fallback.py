"""Unit tests for fallback node."""

import pytest
from src.graph.nodes.fallback import fallback_node, FALLBACK_MESSAGES
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool


class TestFallbackNode:
    """Tests for fallback_node function."""

    def test_fallback_no_selected_tool(self, mock_context):
        """Test fallback when no tool is selected in state."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context,
            selected_tool=None
        )

        result = fallback_node(state)

        assert result["final_answer"] == FALLBACK_MESSAGES["service_unavailable"]
        assert "did not return any results" in result["final_answer"][0]

    def test_fallback_no_tool_match(self, mock_context):
        """Test fallback when NO_TOOL is selected."""
        state = OrchestratorState(
            query="What's the weather?",
            session_id="test-123",
            context=mock_context,
            selected_tool=SelectedTool(
                tool_name="NO_TOOL",
                confidence=0.0,
                reasoning="Out of scope question"
            )
        )

        result = fallback_node(state)

        assert result["final_answer"] == FALLBACK_MESSAGES["no_tool_found"]
        assert "did not return any results" in result["final_answer"][0]

    def test_fallback_low_confidence(self, mock_context):
        """Test fallback for low confidence score."""
        state = OrchestratorState(
            query="Unclear query",
            session_id="test-123",
            context=mock_context,
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=5.0,
                reasoning="Uncertain match"
            )
        )

        result = fallback_node(state)

        assert result["final_answer"] == FALLBACK_MESSAGES["low_confidence"]
        assert "did not return any results" in result["final_answer"][0]

    @pytest.mark.parametrize("confidence", [0.0, 3.5, 6.0, 6.9, 6.99])
    def test_fallback_various_low_confidences(self, mock_context, confidence):
        """Test fallback with various low confidence values."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context,
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=confidence,
                reasoning="Low confidence"
            )
        )

        result = fallback_node(state)

        assert result["final_answer"] == FALLBACK_MESSAGES["low_confidence"]

    def test_fallback_preserves_state_fields(self, mock_context):
        """Test that fallback preserves other state fields."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-session-456",
            context=mock_context,
            selected_tool=SelectedTool(
                tool_name="NO_TOOL",
                confidence=0.0,
                reasoning="Out of scope"
            )
        )

        result = fallback_node(state)

        assert result["final_answer"] == FALLBACK_MESSAGES["no_tool_found"]

    def test_fallback_sets_final_answer(self, mock_context):
        """Test that fallback sets the final_answer field."""
        state = OrchestratorState(
            query="Test",
            session_id="test-123",
            context=mock_context,
            selected_tool=SelectedTool(
                tool_name="NO_TOOL",
                confidence=0.0,
                reasoning="Test"
            )
        )

        result = fallback_node(state)

        assert result["final_answer"] is not None
        assert isinstance(result["final_answer"], list)
        assert len(result["final_answer"]) > 0


class TestFallbackMessages:
    """Tests for fallback message constants."""

    def test_no_tool_found_message_exists(self):
        """Verify no_tool_found message exists."""
        assert "no_tool_found" in FALLBACK_MESSAGES
        assert len(FALLBACK_MESSAGES["no_tool_found"]) > 0

    def test_low_confidence_message_exists(self):
        """Verify low_confidence message exists."""
        assert "low_confidence" in FALLBACK_MESSAGES
        assert len(FALLBACK_MESSAGES["low_confidence"]) > 0

    def test_service_unavailable_message_exists(self):
        """Verify service_unavailable message exists."""
        assert "service_unavailable" in FALLBACK_MESSAGES
        assert len(FALLBACK_MESSAGES["service_unavailable"]) > 0

    def test_messages_are_user_friendly(self):
        """Verify messages are polite and user-friendly."""
        for key, message in FALLBACK_MESSAGES.items():
            assert "please" in message[0].lower()
            assert len(message[0]) > 20  # Reasonable length
            assert message[0][0].isupper()  # Starts with capital letter

    def test_all_messages_are_same(self):
        """Verify all fallback messages use the same unified message."""
        messages = list(FALLBACK_MESSAGES.values())
        assert all(m == messages[0] for m in messages)

    def test_messages_contain_html_links(self):
        """Verify fallback message contains HTML anchor tags with correct URLs."""
        for message in FALLBACK_MESSAGES.values():
            assert "<a href='https://www.fepblue.org/plan-brochures'" in message[0]
            assert "<a href='https://www.fepblue.org/plan-summaries'" in message[0]
            assert "target='_blank'" in message[0]
            assert "Opens in a new window" not in message[0]