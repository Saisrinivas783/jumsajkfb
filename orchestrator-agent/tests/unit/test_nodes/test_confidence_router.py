"""Unit tests for confidence_router node."""

import pytest
from src.graph.nodes.confidence_router import guard_rails_router
from src.config.settings import orchestrator_settings
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool


class TestGuardRailsRouter:
    """Tests for guard_rails_router function."""

    def test_router_no_selected_tool(self, mock_context):
        """Test routing when no tool is selected."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            selected_tool=None,
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "fallback"

    def test_router_no_tool_match(self, mock_context):
        """Test routing when NO_TOOL is selected."""
        state = OrchestratorState(
            query="What's the weather?",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="NO_TOOL",
                confidence=0.0,
                reasoning="Out of scope"
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "fallback"

    def test_router_low_confidence(self, mock_context):
        """Test routing when confidence is below threshold."""
        state = OrchestratorState(
            query="Unclear query",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=6.5,  # Below 7.0 threshold
                reasoning="Uncertain match"
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "fallback"

    def test_router_exact_threshold(self, mock_context):
        """Test routing when confidence is exactly at threshold."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=7.0,  # Exactly at threshold
                reasoning="Reasonable match",
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "IBTAgent"

    def test_router_high_confidence(self, mock_context):
        """Test routing when confidence is above threshold."""
        state = OrchestratorState(
            query="What are my benefits?",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=9.5,
                reasoning="Strong match",
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "IBTAgent"

    def test_router_maximum_confidence(self, mock_context):
        """Test routing with maximum confidence score."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=10.0,
                reasoning="Perfect match",
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "IBTAgent"

    def test_router_minimum_confidence(self, mock_context):
        """Test routing with minimum confidence score."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=0.0,
                reasoning="No match"
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "fallback"

    def test_router_different_tools(self, mock_context):
        """Test routing returns correct tool name."""
        state = OrchestratorState(
            query="What's my claim status?",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="ClaimsAgent",
                confidence=9.0,
                reasoning="Claims query",
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == "ClaimsAgent"

    @pytest.mark.parametrize("confidence,expected_route", [
        (0.0, "fallback"),
        (3.5, "fallback"),
        (6.9, "fallback"),
        (6.99, "fallback"),
        (7.0, "IBTAgent"),
        (7.01, "IBTAgent"),
        (8.5, "IBTAgent"),
        (10.0, "IBTAgent"),
    ])
    def test_router_confidence_boundaries(self, mock_context, confidence, expected_route):
        """Test routing with various confidence levels."""
        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            selected_tool=SelectedTool(
                tool_name="IBTAgent",
                confidence=confidence,
                reasoning="Test",
            ),
            context=mock_context
        )

        route = guard_rails_router(state)
        assert route == expected_route


class TestConfidenceThreshold:
    """Tests for confidence threshold setting."""

    def test_threshold_value(self):
        """Verify the confidence threshold default is 7.0."""
        assert orchestrator_settings.confidence_threshold_high == 7.0
