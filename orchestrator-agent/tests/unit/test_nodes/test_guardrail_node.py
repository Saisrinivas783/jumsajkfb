"""Unit tests for guardrail_node — guardrail check node and router."""

import pytest
from unittest.mock import MagicMock, patch

from src.graph.nodes.guardrail_node import guardrail_check_node, guardrail_router, _is_hard_block
from src.schemas.state import OrchestratorState
from src.schemas.api import InvocationContext


# ---------------------------------------------------------------------------
# Response factories
# ---------------------------------------------------------------------------

def _make_none_response() -> dict:
    return {"action": "NONE", "assessments": [], "outputs": []}


def _make_hard_block_response() -> dict:
    return {
        "action": "INTERVENED",
        "assessments": [
            {
                "topicPolicy": {
                    "topics": [{"name": "Violence", "type": "DENY", "action": "BLOCKED"}]
                }
            }
        ],
        "outputs": [{"text": "Sorry, that topic is not allowed."}],
    }


def _make_content_filter_block_response() -> dict:
    return {
        "action": "INTERVENED",
        "assessments": [
            {
                "contentPolicy": {
                    "filters": [{"type": "HATE", "confidence": "HIGH", "action": "BLOCKED"}]
                }
            }
        ],
        "outputs": [{"text": "Content filtered."}],
    }


def _make_pii_mask_response(redacted_text: str) -> dict:
    return {
        "action": "INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "EMAIL", "match": "foo@bar.com", "action": "ANONYMIZED"}]
                }
            }
        ],
        "outputs": [{"text": redacted_text}],
    }


def _make_pii_mask_no_output_response() -> dict:
    return {
        "action": "INTERVENED",
        "assessments": [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "EMAIL", "match": "foo@bar.com", "action": "ANONYMIZED"}]
                }
            }
        ],
        "outputs": [],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_state():
    """Base orchestrator state for guardrail tests."""
    return OrchestratorState(
        query="What are my dental benefits?",
        session_id="test-session-123",
        context=InvocationContext(
            userName="test_user",
            userType="member",
            source="TestPage",
            productId="PROD-001",
        ),
    )


@pytest.fixture
def blocked_state():
    """State where guardrail has blocked the query."""
    return OrchestratorState.model_construct(
        query="bad query",
        session_id="test-session-456",
        context=InvocationContext(
            userName="test_user",
            userType="member",
            source="TestPage",
            productId="PROD-001",
        ),
        guardrail_blocked=True,
        selected_tool=None,
        tool_result=None,
        tool_metadata=[],
        final_answer=["Blocked message"],
        error=None,
    )


@pytest.fixture
def passed_state():
    """State where guardrail has passed the query."""
    return OrchestratorState.model_construct(
        query="What are my benefits?",
        session_id="test-session-789",
        context=InvocationContext(
            userName="test_user",
            userType="member",
            source="TestPage",
            productId="PROD-001",
        ),
        guardrail_blocked=False,
        selected_tool=None,
        tool_result=None,
        tool_metadata=[],
        final_answer=None,
        error=None,
    )


# ---------------------------------------------------------------------------
# TestIsHardBlock
# ---------------------------------------------------------------------------

class TestIsHardBlock:
    """Unit tests for the _is_hard_block helper."""

    def test_empty_assessments_returns_false(self):
        assert _is_hard_block([]) is False

    def test_only_anonymized_pii_returns_false(self):
        assessments = [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]
                }
            }
        ]
        assert _is_hard_block(assessments) is False

    def test_blocked_topic_returns_true(self):
        assessments = [
            {
                "topicPolicy": {
                    "topics": [{"name": "Violence", "action": "BLOCKED"}]
                }
            }
        ]
        assert _is_hard_block(assessments) is True

    def test_blocked_content_filter_returns_true(self):
        assessments = [
            {
                "contentPolicy": {
                    "filters": [{"type": "HATE", "action": "BLOCKED"}]
                }
            }
        ]
        assert _is_hard_block(assessments) is True

    def test_mixed_pii_and_blocked_topic_returns_true(self):
        assessments = [
            {
                "sensitiveInformationPolicy": {
                    "piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]
                },
                "topicPolicy": {
                    "topics": [{"name": "Violence", "action": "BLOCKED"}]
                },
            }
        ]
        assert _is_hard_block(assessments) is True

    def test_topic_with_non_blocked_action_returns_false(self):
        assessments = [
            {
                "topicPolicy": {
                    "topics": [{"name": "Investments", "action": "NONE"}]
                }
            }
        ]
        assert _is_hard_block(assessments) is False

    def test_word_policy_custom_word_blocked_returns_true(self):
        assessments = [
            {
                "wordPolicy": {
                    "customWords": [{"match": "badword", "action": "BLOCKED"}],
                    "managedWordLists": [],
                }
            }
        ]
        assert _is_hard_block(assessments) is True

    def test_word_policy_managed_word_blocked_returns_true(self):
        assessments = [
            {
                "wordPolicy": {
                    "customWords": [],
                    "managedWordLists": [{"match": "profanity", "action": "BLOCKED"}],
                }
            }
        ]
        assert _is_hard_block(assessments) is True

    def test_blocked_in_second_of_multiple_assessments_returns_true(self):
        assessments = [
            {"topicPolicy": {"topics": []}},
            {
                "contentPolicy": {
                    "filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]
                }
            },
        ]
        assert _is_hard_block(assessments) is True


# ---------------------------------------------------------------------------
# TestGuardrailCheckNode
# ---------------------------------------------------------------------------

class TestGuardrailCheckNode:
    """Tests for guardrail_check_node function using apply_guardrail mock."""

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_skips_check_when_no_guardrail_configured(self, mock_get_chat, mock_state):
        """When AWS_BEDROCK_GUARDRAIL_ID is not set, check is skipped and apply_guardrail not called."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = None
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is False
        assert result["guardrail_action"] == "NONE"
        mock_chat_models.apply_guardrail.assert_not_called()

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_action_none_returns_not_blocked(self, mock_get_chat, mock_state):
        """When action is NONE, guardrail_blocked is False with no query/final_answer."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = _make_none_response()
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is False
        assert result["guardrail_action"] == "NONE"
        assert "final_answer" not in result
        assert "query" not in result

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_apply_guardrail_called_with_query_and_input_source(self, mock_get_chat, mock_state):
        """apply_guardrail is called with the state query and source=INPUT."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = _make_none_response()
        mock_get_chat.return_value = mock_chat_models

        guardrail_check_node(mock_state)

        mock_chat_models.apply_guardrail.assert_called_once_with(
            text=mock_state.query, source="INPUT"
        )

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_intervened_blocked_topic_returns_hard_block(self, mock_get_chat, mock_state):
        """INTERVENED + BLOCKED topic → guardrail_blocked=True, final_answer set."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = _make_hard_block_response()
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is True
        assert result["guardrail_action"] == "BLOCKED"
        assert result["final_answer"] == ["Sorry, that topic is not allowed."]

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_intervened_content_filter_returns_hard_block(self, mock_get_chat, mock_state):
        """INTERVENED + BLOCKED content filter → hard block."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = _make_content_filter_block_response()
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is True
        assert result["guardrail_action"] == "BLOCKED"
        assert result["final_answer"] == ["Content filtered."]

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_intervened_blocked_empty_outputs_uses_default_message(self, mock_get_chat, mock_state):
        """INTERVENED + BLOCKED + empty outputs → fallback default message used."""
        response = _make_hard_block_response()
        response["outputs"] = []
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = response
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is True
        assert result["guardrail_action"] == "BLOCKED"
        assert "content policy" in result["final_answer"][0].lower()

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_intervened_pii_only_updates_query_with_redacted_text(self, mock_get_chat, mock_state):
        """INTERVENED + PII only → guardrail_blocked=False, query updated with redacted text."""
        redacted = "My email is [EMAIL]"
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = _make_pii_mask_response(redacted)
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is False
        assert result["guardrail_action"] == "PII_MASKED"
        assert result["query"] == redacted
        assert "final_answer" not in result

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_intervened_pii_empty_outputs_preserves_original_query(self, mock_get_chat, mock_state):
        """INTERVENED + PII + empty outputs → original query preserved."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = _make_pii_mask_no_output_response()
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is False
        assert result["guardrail_action"] == "PII_MASKED"
        assert result["query"] == mock_state.query

    @patch('src.graph.nodes.guardrail_node.get_chat_models')
    def test_boto_client_error_raises_guardrail_error(self, mock_get_chat_models, mock_state):
        """Test that botocore ClientError raises GuardrailError."""
        from botocore.exceptions import ClientError
        from src.exceptions import GuardrailError
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "gr-123"
        mock_chat_models.apply_guardrail.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "ApplyGuardrail"
        )
        mock_get_chat_models.return_value = mock_chat_models

        with pytest.raises(GuardrailError):
            guardrail_check_node(mock_state)

    @patch('src.graph.nodes.guardrail_node.get_chat_models')
    def test_boto_core_error_raises_guardrail_error(self, mock_get_chat_models, mock_state):
        """Test that botocore BotoCoreError raises GuardrailError."""
        from botocore.exceptions import BotoCoreError
        from src.exceptions import GuardrailError
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "gr-123"
        mock_chat_models.apply_guardrail.side_effect = BotoCoreError()
        mock_get_chat_models.return_value = mock_chat_models

        with pytest.raises(GuardrailError):
            guardrail_check_node(mock_state)

    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    def test_missing_action_key_defaults_to_none_pass_through(self, mock_get_chat, mock_state):
        """Missing 'action' key in response defaults to NONE (pass-through)."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail.return_value = {"assessments": [], "outputs": []}
        mock_get_chat.return_value = mock_chat_models

        result = guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is False
        assert result["guardrail_action"] == "NONE"
        assert "final_answer" not in result
        assert "query" not in result


# ---------------------------------------------------------------------------
# TestGuardrailRouter
# ---------------------------------------------------------------------------

class TestGuardrailRouter:
    """Tests for guardrail_router function."""

    def test_returns_end_when_blocked(self, blocked_state):
        """Blocked state routes to 'end'."""
        assert guardrail_router(blocked_state) == "end"

    def test_returns_analyzer_when_not_blocked(self, passed_state):
        """Passed state routes to 'analyzer'."""
        assert guardrail_router(passed_state) == "analyzer"

    def test_default_state_routes_to_analyzer(self, mock_state):
        """Default state (guardrail_blocked=False) routes to analyzer."""
        assert guardrail_router(mock_state) == "analyzer"
