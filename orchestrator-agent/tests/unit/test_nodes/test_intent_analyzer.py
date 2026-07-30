"""Unit tests for intent_analyzer node."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph.nodes.intent_analyzer import create_intent_node
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool
from src.schemas.llm import ToolSelectionOutput
from src.schemas.registry import ToolDefinition, ToolParameters
from src.schemas.api import InvocationContext
from src.exceptions import LLMFailureError


@pytest.fixture
def mock_registry():
    """Create a mock tool registry."""
    return {
        "IBTAgent": ToolDefinition(
            name="IBTAgent",
            description="Handles insurance benefit inquiries",
            endpoint="https://ibt-service.internal/api/v1",
            capabilities=["benefit inquiries", "coverage questions"],
            parameters=ToolParameters(required=["userPrompt", "userName"], optional=["policyNumber"])
        ),
        "ClaimsAgent": ToolDefinition(
            name="ClaimsAgent",
            description="Handles claims inquiries",
            endpoint="https://claims-service.internal/api/v1",
            capabilities=["claim status", "claim submission"],
            parameters=ToolParameters(required=["userPrompt"], optional=["claimNumber"])
        )
    }


@pytest.fixture
def mock_state(mock_registry):
    """Create a mock orchestrator state."""
    return OrchestratorState(
        query="What are my dental benefits?",
        session_id="test-session-123",
        context=InvocationContext(
            userName="john_doe",
            userType="member",
            source="DXAIService",
            promptId="p-123",
            productId="PROD-001"
        )
    )


def _make_llm_mock(content_dict: dict) -> MagicMock:
    """Return a mock LLM whose with_structured_output().invoke() returns a dict with parsed response."""
    mock_llm = MagicMock()
    mock_llm.model_id = "test-model"
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = {
        "parsed": ToolSelectionOutput(**content_dict),
        "raw": MagicMock(),
        "parsing_error": None
    }
    mock_llm.with_structured_output.return_value = structured_mock
    return mock_llm


class TestIntentNode:
    """Tests for create_intent_node factory and returned closure."""

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_success(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state
    ):
        """Test successful intent analysis."""
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = _make_llm_mock({
            "selected_tool": "IBTAgent",
            "confidence_score": 9.0,
            "reasoning": "User asking about dental coverage",
            "reformulated_query": "dental coverage benefits",
        })
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.return_value = mock_llm
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        result = intent_node(mock_state)

        assert result["selected_tool"] is not None
        assert result["selected_tool"].tool_name == "IBTAgent"
        assert result["selected_tool"].confidence == 9.0
        assert result["selected_tool"].reasoning == "User asking about dental coverage"
        assert result["selected_tool"].reformulated_query == "dental coverage benefits"

        mock_llm.with_structured_output.assert_called_once_with(ToolSelectionOutput, include_raw=True)
        structured_mock = mock_llm.with_structured_output.return_value
        structured_mock.invoke.assert_called_once()
        call_args = structured_mock.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    def test_create_intent_node_get_model_failure(self, mock_get_chat, mock_registry, mock_state):
        """Test that a ClientError from get_model() propagates at invocation time."""
        from botocore.exceptions import ClientError
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "InvokeModel"
        )
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        with pytest.raises(LLMFailureError):
            intent_node(mock_state)

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_invoke_failure(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state,
    ):
        """Test that a ClientError from structured_llm.invoke() raises LLMFailureError."""
        from botocore.exceptions import ClientError
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = MagicMock()
        mock_llm.model_id = "test-model"
        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "InvokeModel"
        )
        mock_llm.with_structured_output.return_value = structured_mock
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.return_value = mock_llm
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        with pytest.raises(LLMFailureError) as exc_info:
            intent_node(mock_state)

        assert "Intent analysis failed" in str(exc_info.value)

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_no_tool_match(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state
    ):
        """Test NO_TOOL selection."""
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = _make_llm_mock({
            "selected_tool": "NO_TOOL",
            "confidence_score": 0.0,
            "reasoning": "Out of scope question",
            "reformulated_query": "out of scope question",
        })
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.return_value = mock_llm
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        result = intent_node(mock_state)

        assert result["selected_tool"].tool_name == "NO_TOOL"
        assert result["selected_tool"].confidence == 0.0

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_with_null_context(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry
    ):
        """Test that intent_node succeeds when state.context is None."""
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = _make_llm_mock({
            "selected_tool": "IBTAgent",
            "confidence_score": 8.0,
            "reasoning": "Benefit query",
            "reformulated_query": "insurance benefits inquiry",
        })
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.return_value = mock_llm
        mock_get_chat.return_value = mock_chat_models

        # Use model_construct to bypass validation and set context=None
        state = OrchestratorState.model_construct(
            query="What are my benefits?",
            session_id="test-123",
            context=None,
            selected_tool=None,
            tool_result=None,
            tool_metadata=[],
            final_answer=None,
            error=None,
        )

        intent_node = create_intent_node(mock_registry)
        result = intent_node(state)
        assert result["selected_tool"] is not None

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_none_response_fallback(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state,
    ):
        """Test that None from structured LLM (safety bypass) falls back to NO_TOOL."""
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = MagicMock()
        mock_llm.model_id = "test-model"
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = {
            "parsed": None,
            "raw": MagicMock(),
            "parsing_error": None
        }
        mock_llm.with_structured_output.return_value = structured_mock
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.return_value = mock_llm
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        result = intent_node(mock_state)

        assert result["selected_tool"].tool_name == "NO_TOOL"
        assert result["selected_tool"].confidence == 0.0
        assert result["selected_tool"].reformulated_query is None

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_output_parser_exception_fallback(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state,
    ):
        """Test that OutputParserException gracefully falls back to NO_TOOL."""
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = MagicMock()
        mock_llm.model_id = "test-model"
        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = OutputParserException("Failed to parse")
        mock_llm.with_structured_output.return_value = structured_mock
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.return_value = mock_llm
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        result = intent_node(mock_state)

        assert result["selected_tool"].tool_name == "NO_TOOL"
        assert result["selected_tool"].confidence == 0.0
        assert result["selected_tool"].reformulated_query is None

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_empty_after_cleaning_returns_no_tool(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
    ):
        """Punctuation/symbol-only queries clean to '' and must short-circuit to NO_TOOL."""
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_chat_models = MagicMock()
        mock_get_chat.return_value = mock_chat_models

        state = OrchestratorState(
            query="^&*~",  # Characters that get removed completely
            session_id="test-session-empty",
            context=InvocationContext(
                userName="john_doe",
                userType="member",
                source="DXAIService",
                promptId="p-123",
                productId="PROD-001",
            ),
        )

        intent_node = create_intent_node(mock_registry)
        result = intent_node(state)

        assert result["selected_tool"].tool_name == "NO_TOOL"
        assert result["selected_tool"].confidence == 0.0
        assert result["selected_tool"].reformulated_query is None
        assert result["input_tokens"] is None
        # LLM must not have been called
        mock_chat_models.get_model.assert_not_called()

    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    def test_intent_node_low_confidence(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state
    ):
        """Test low confidence tool selection."""
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = _make_llm_mock({
            "selected_tool": "IBTAgent",
            "confidence_score": 5.0,
            "reasoning": "Unclear query",
            "reformulated_query": "insurance benefits query",
        })
        mock_chat_models = MagicMock()
        mock_chat_models.get_model.return_value = mock_llm
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        result = intent_node(mock_state)

        assert result["selected_tool"].confidence == 5.0
        assert result["selected_tool"].tool_name == "IBTAgent"