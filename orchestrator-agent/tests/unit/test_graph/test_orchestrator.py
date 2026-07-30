"""Unit tests for OrchestratorAgent."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import time
from pydantic import ValidationError as PydanticValidationError

from src.graph.orchestrator import OrchestratorAgent
from src.schemas.api import InvocationRequest, InvocationContext, InvocationResponse
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool
from src.exceptions import OrchestratorError, LLMFailureError


@pytest.fixture
def mock_registry_path(tmp_path):
    """Create a temporary YAML file for testing."""
    yaml_content = """tools:
  - name: IBTAgent
    description: Handles insurance benefit inquiries
    endpoint: https://ibt-service.internal/api/v1
    capabilities:
      - benefit inquiries
      - coverage questions
    parameters:
      required:
        - userPrompt
        - userName
      optional:
        - policyNumber
    examples:
      - prompt: "What are my dental benefits?"
        reasoning: "User asking about specific benefit coverage"
"""
    yaml_file = tmp_path / "tools.yaml"
    yaml_file.write_text(yaml_content)
    return str(yaml_file)


@pytest.fixture
def sample_request():
    """Create a sample invocation request with required context."""
    return InvocationRequest(
        user_prompt="What are my dental benefits?",
        session_id="test-session-123",
        context=InvocationContext(
            userName="john_doe",
            userType="member",
            source="IBTPage",
            promptId="p-123",
            productId="PROD-001"
        )
    )


class TestOrchestratorAgentInit:
    """Tests for OrchestratorAgent initialization."""

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_orchestrator_init_success(self, mock_build_graph, mock_from_yaml, mock_registry_path):
        """Test successful orchestrator initialization."""
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)

        assert agent.registry == mock_registry
        assert agent.graph_app == mock_graph
        mock_from_yaml.assert_called_once_with(mock_registry_path)
        mock_build_graph.assert_called_once_with(mock_registry)

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_orchestrator_init_with_default_path(self, mock_build_graph, mock_from_yaml):
        """Test initialization with default registry path."""
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=0)
        mock_registry.list_tool_names.return_value = []
        mock_from_yaml.return_value = mock_registry
        mock_build_graph.return_value = MagicMock()

        agent = OrchestratorAgent()

        mock_from_yaml.assert_called_once_with("src/tools/definitions/tools.yaml")


class TestHandleInvocation:
    """Tests for handle_invocation method."""

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_handle_invocation_success(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test successful invocation handling."""
        # Setup mocks with proper ToolDefinition
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        # Mock graph execution
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "What are my dental benefits?",
            "session_id": "test-session-123",
            "registry": {},
            "context": sample_request.context.model_dump(),
            "selected_tool": {
                "tool_name": "IBTAgent",
                "confidence": 9.0,
                "reasoning": "Benefit inquiry",
                "reformulated_query": "dental coverage benefits",
                "parameters": {}
            },
            "tool_result": None,
            "tool_metadata": [],
            "final_answer": ["Your dental benefits include..."],
            "error": None
        }
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        response = agent.handle_invocation(sample_request)

        assert isinstance(response, InvocationResponse)
        assert response.session_id == "test-session-123"
        assert response.success is True
        assert response.response_text == ["Your dental benefits include..."]
        assert response.execution_time_ms >= 0.0  # Can be 0.0 for fast execution

        # Check metadata structure
        assert len(response.metadata) == 1
        assert response.metadata[0].agent == "orchestrator"

        # Extract metadata values
        metadata_dict = {item.key: item.value for item in response.metadata[0].data}
        assert metadata_dict["confidence"] == 9.0
        assert metadata_dict["selectedTool"] == "IBTAgent"
        assert metadata_dict["reasoning"] == "Benefit inquiry"
        assert metadata_dict["reformulatedQuery"] == "dental coverage benefits"

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_handle_invocation_empty_prompt(
        self,
        mock_build_graph,
        mock_from_yaml,
        mock_registry_path
    ):
        """Test that empty user prompt is rejected by Pydantic validation."""
        # Validation now happens at Pydantic level, not in orchestrator
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="",
                session_id="test-123",
                context=InvocationContext(
                    userName="test_user",
                    userType="member",
                    source="IBTPage",
                    productId="PROD-001"
                )
            )

        # Verify the validation error is for the empty user_prompt
        errors = exc_info.value.errors()
        assert any("user_prompt" in str(e["loc"]) for e in errors)
        assert any("empty" in str(e["msg"]).lower() for e in errors)

    def test_handle_invocation_whitespace_prompt(self):
        """Test that whitespace-only prompt is rejected by Pydantic validation."""
        # Validation now happens at Pydantic level, not in orchestrator
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="   \t\n  ",
                session_id="test-123",
                context=InvocationContext(
                    userName="test_user",
                    userType="member",
                    source="IBTPage",
                    productId="PROD-001"
                )
            )

        # Verify the validation error is for the whitespace user_prompt
        errors = exc_info.value.errors()
        assert any("user_prompt" in str(e["loc"]) for e in errors)
        assert any("empty" in str(e["msg"]).lower() for e in errors)

    def test_validation_empty_session_id(self):
        """Test that empty sessionId is rejected by Pydantic validation."""
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="Test query",
                session_id="",
                context=InvocationContext(
                    userName="test_user",
                    userType="member",
                    source="IBTPage",
                    productId="PROD-001"
                )
            )

        errors = exc_info.value.errors()
        assert any("session_id" in str(e["loc"]) for e in errors)

    def test_validation_empty_context_fields(self):
        """Test that empty context fields are rejected by Pydantic validation."""
        # Test empty userName
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="Test query",
                session_id="test-123",
                context=InvocationContext(
                    userName="",
                    userType="member",
                    source="IBTPage",
                    productId="PROD-001"
                )
            )
        errors = exc_info.value.errors()
        assert any("userName" in str(e["loc"]) for e in errors)

        # Test empty userType
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="Test query",
                session_id="test-123",
                context=InvocationContext(
                    userName="test_user",
                    userType="",
                    source="IBTPage",
                    productId="PROD-001"
                )
            )
        errors = exc_info.value.errors()
        assert any("userType" in str(e["loc"]) for e in errors)

        # Test empty source
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="Test query",
                session_id="test-123",
                context=InvocationContext(
                    userName="test_user",
                    userType="member",
                    source="",
                    productId="PROD-001"
                )
            )
        errors = exc_info.value.errors()
        assert any("source" in str(e["loc"]) for e in errors)

        # Test empty productId
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="Test query",
                session_id="test-123",
                context=InvocationContext(
                    userName="test_user",
                    userType="member",
                    source="IBTPage",
                    productId=""
                )
            )
        errors = exc_info.value.errors()
        assert any("productId" in str(e["loc"]) for e in errors)

        # Test whitespace-only productId
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationRequest(
                user_prompt="Test query",
                session_id="test-123",
                context=InvocationContext(
                    userName="test_user",
                    userType="member",
                    source="IBTPage",
                    productId="   "
                )
            )
        errors = exc_info.value.errors()
        assert any("productId" in str(e["loc"]) for e in errors)

    def test_validation_missing_productId(self):
        """Test that missing productId is rejected by Pydantic validation."""
        with pytest.raises(PydanticValidationError) as exc_info:
            InvocationContext(
                userName="test_user",
                userType="member",
                source="IBTPage"
                # productId intentionally omitted
            )
        errors = exc_info.value.errors()
        assert any("productId" in str(e["loc"]) for e in errors)
        assert any(e["type"] == "missing" for e in errors)

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_handle_invocation_orchestrator_error(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test handling of OrchestratorError."""
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = LLMFailureError("Invalid input")
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        response = agent.handle_invocation(sample_request)

        assert response.success is False
        assert "Invalid input" in response.message

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_handle_invocation_unexpected_error(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test handling of unexpected errors."""
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = Exception("Unexpected error")
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        response = agent.handle_invocation(sample_request)

        assert response.success is False
        assert "unexpected" in response.message.lower()

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_handle_invocation_no_selected_tool(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test handling when no tool is selected."""
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "Test",
            "session_id": "test-123",
            "registry": {},
            "context": sample_request.context.model_dump(),
            "selected_tool": None,
            "tool_result": None,
            "tool_metadata": [],
            "final_answer": ["Fallback response"],
            "error": None
        }
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        response = agent.handle_invocation(sample_request)

        assert response.success is True
        # When no tool is selected, orchestrator data only has guardrail keys
        assert len(response.metadata) == 1
        assert response.metadata[0].agent == "orchestrator"
        metadata_dict = {item.key: item.value for item in response.metadata[0].data}
        assert metadata_dict["guardrailAction"] == "NONE"
        assert metadata_dict["guardrailBlocked"] is False

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_handle_invocation_execution_time(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test that execution time is measured."""
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "Test",
            "session_id": "test-123",
            "registry": {},
            "context": sample_request.context.model_dump(),
            "selected_tool": None,
            "tool_result": None,
            "tool_metadata": [],
            "final_answer": ["Response"],
            "error": None
        }

        # Add delay to graph execution
        def slow_invoke(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return {
                "query": "Test",
                "session_id": "test-123",
                "registry": {},
                "context": sample_request.context.model_dump(),
                "selected_tool": None,
                "tool_result": None,
                "tool_metadata": [],
                "final_answer": ["Response"],
                "error": None
            }

        mock_graph.invoke.side_effect = slow_invoke
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        response = agent.handle_invocation(sample_request)

        assert response.execution_time_ms >= 10.0


    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_handle_invocation_with_tool_metadata(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test that tool_metadata is included in response (line 128)."""
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        tool_meta = {"agent": "IBTAgent", "data": [{"key": "result", "value": "ok"}]}
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "Test",
            "session_id": "test-session-123",
            "registry": {},
            "context": sample_request.context.model_dump(),
            "selected_tool": {
                "tool_name": "IBTAgent",
                "confidence": 9.0,
                "reasoning": "Benefit inquiry",
                "parameters": {}
            },
            "tool_result": None,
            "tool_metadata": [tool_meta],
            "final_answer": ["Response with tool metadata"],
            "error": None
        }
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        response = agent.handle_invocation(sample_request)

        assert response.success is True
        # Orchestrator metadata + tool metadata
        assert len(response.metadata) == 2


class TestGuardrailMetadata:
    """Tests that guardrail outcome is reflected in orchestrator metadata block."""

    def _make_agent(self, mock_from_yaml, mock_build_graph, mock_registry_path, graph_return):
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = graph_return
        mock_build_graph.return_value = mock_graph

        return OrchestratorAgent(registry_path=mock_registry_path)

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_guardrail_none_in_orchestrator_metadata(
        self, mock_build_graph, mock_from_yaml, sample_request, mock_registry_path
    ):
        """Default state (no guardrail triggered) exposes NONE / False in orchestrator metadata."""
        graph_return = {
            "query": "Test",
            "session_id": "test-session-123",
            "context": sample_request.context.model_dump(),
            "selected_tool": None,
            "tool_result": None,
            "tool_metadata": [],
            "final_answer": ["Response"],
            "guardrail_blocked": False,
            "guardrail_action": "NONE",
            "error": None,
        }
        agent = self._make_agent(mock_from_yaml, mock_build_graph, mock_registry_path, graph_return)
        response = agent.handle_invocation(sample_request)

        orchestrator_meta = next(m for m in response.metadata if m.agent == "orchestrator")
        data = {item.key: item.value for item in orchestrator_meta.data}
        assert data["guardrailAction"] == "NONE"
        assert data["guardrailBlocked"] is False

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_guardrail_blocked_in_orchestrator_metadata(
        self, mock_build_graph, mock_from_yaml, sample_request, mock_registry_path
    ):
        """Blocked state exposes BLOCKED / True in orchestrator metadata."""
        graph_return = {
            "query": "bad query",
            "session_id": "test-session-123",
            "context": sample_request.context.model_dump(),
            "selected_tool": None,
            "tool_result": None,
            "tool_metadata": [],
            "final_answer": ["I'm sorry, but your request cannot be processed due to content policy."],
            "guardrail_blocked": True,
            "guardrail_action": "BLOCKED",
            "error": None,
        }
        agent = self._make_agent(mock_from_yaml, mock_build_graph, mock_registry_path, graph_return)
        response = agent.handle_invocation(sample_request)

        orchestrator_meta = next(m for m in response.metadata if m.agent == "orchestrator")
        data = {item.key: item.value for item in orchestrator_meta.data}
        assert data["guardrailAction"] == "BLOCKED"
        assert data["guardrailBlocked"] is True


class TestErrorResponse:
    """Tests for _error_response method."""

    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    def test_error_response_structure(
        self,
        mock_build_graph,
        mock_from_yaml,
        mock_registry_path
    ):
        """Test error response structure."""
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=0)
        mock_registry.list_tool_names.return_value = []
        mock_from_yaml.return_value = mock_registry
        mock_build_graph.return_value = MagicMock()

        agent = OrchestratorAgent(registry_path=mock_registry_path)

        start_time = time.time()
        response = agent._error_response("test-123", "Test error", start_time)

        assert isinstance(response, InvocationResponse)
        assert response.success is False
        assert response.message == "Test error"
        assert response.session_id == "test-123"
        assert "technical difficulties" in response.response_text[0]
        assert response.metadata == []  # Empty metadata on error
        assert response.execution_time_ms >= 0