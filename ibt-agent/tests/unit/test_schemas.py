"""Unit tests for API schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas.api import HealthResponse, InvocationRequest, InvocationResponse


VALID_CONTEXT = {
    "userName": "test_user",
    "userType": "member",
    "source": "IBTPage",
    "productId": "6",
}


class TestInvocationRequest:
    """Tests for InvocationRequest schema."""

    def test_invocation_request_valid(self):
        """Test valid InvocationRequest creation."""
        request = InvocationRequest(
            sessionId="session-123",
            userPrompt="What are my benefits?",
            context=VALID_CONTEXT,
        )

        assert request.session_id == "session-123"
        assert request.user_prompt == "What are my benefits?"
        assert request.context.productId == "6"

    def test_invocation_request_aliases(self):
        """Test InvocationRequest field aliases."""
        request = InvocationRequest(
            session_id="session-123",
            user_prompt="Test prompt",
            context=VALID_CONTEXT,
        )

        assert request.session_id == "session-123"
        assert request.user_prompt == "Test prompt"
        assert request.context.productId == "6"

    def test_invocation_request_missing_session_id(self):
        """Test InvocationRequest requires session_id."""
        with pytest.raises(ValidationError):
            InvocationRequest(userPrompt="Test", context=VALID_CONTEXT)

    def test_invocation_request_missing_user_prompt(self):
        """Test InvocationRequest requires user_prompt."""
        with pytest.raises(ValidationError):
            InvocationRequest(sessionId="session-123", context=VALID_CONTEXT)

    def test_invocation_request_missing_context(self):
        """Test InvocationRequest requires context."""
        with pytest.raises(ValidationError):
            InvocationRequest(sessionId="session-123", userPrompt="Test prompt")

    def test_invocation_request_missing_product_id(self):
        """Test InvocationRequest requires context.productId."""
        context = {key: value for key, value in VALID_CONTEXT.items() if key != "productId"}

        with pytest.raises(ValidationError):
            InvocationRequest(sessionId="session-123", userPrompt="Test prompt", context=context)

    def test_invocation_request_blank_product_id(self):
        """Test InvocationRequest rejects blank context.productId."""
        context = {**VALID_CONTEXT, "productId": "   "}

        with pytest.raises(ValidationError):
            InvocationRequest(sessionId="session-123", userPrompt="Test prompt", context=context)

    def test_invocation_request_empty_prompt_values_are_still_accepted(self):
        """Test current request behavior with empty non-context values."""
        request = InvocationRequest(
            sessionId="",
            userPrompt="",
            context=VALID_CONTEXT,
        )

        assert request.session_id == ""
        assert request.user_prompt == ""

    def test_invocation_request_long_values(self):
        """Test InvocationRequest with long values."""
        long_session = "session-" + "x" * 1000
        long_prompt = "What are my benefits? " * 100

        request = InvocationRequest(
            sessionId=long_session,
            userPrompt=long_prompt,
            context=VALID_CONTEXT,
        )

        assert request.session_id == long_session
        assert request.user_prompt == long_prompt
        assert request.context.productId == "6"


class TestInvocationResponse:
    """Tests for InvocationResponse schema."""

    def test_invocation_response_valid(self):
        """Test valid InvocationResponse creation."""
        response = InvocationResponse(
            sessionId="session-123",
            confidence=8.0,
            responseText="Here are your benefits...",
            success=True,
            message="Success",
            execution_time_ms=150.5
        )

        assert response.session_id == "session-123"
        assert response.confidence == 8.0
        assert response.response_text == "Here are your benefits..."
        assert response.success is True
        assert response.message == "Success"
        assert response.execution_time_ms == 150.5
        assert isinstance(response.timestamp, datetime)

    def test_invocation_response_defaults(self):
        """Test InvocationResponse default values."""
        response = InvocationResponse(sessionId="session-123")

        assert response.session_id == "session-123"
        assert response.confidence == 0.0
        assert response.response_text == ""
        assert response.success is True
        assert response.message == ""
        assert response.execution_time_ms == 0.0
        assert isinstance(response.timestamp, datetime)

    def test_invocation_response_confidence_validation(self):
        """Test InvocationResponse confidence validation."""
        response = InvocationResponse(sessionId="session-123", confidence=0.0)
        assert response.confidence == 0.0

        response = InvocationResponse(sessionId="session-123", confidence=10.0)
        assert response.confidence == 10.0

        response = InvocationResponse(sessionId="session-123", confidence=5.5)
        assert response.confidence == 5.5

        with pytest.raises(ValidationError):
            InvocationResponse(sessionId="session-123", confidence=15.0)

        with pytest.raises(ValidationError):
            InvocationResponse(sessionId="session-123", confidence=-2.0)

    def test_invocation_response_aliases(self):
        """Test InvocationResponse field aliases."""
        response = InvocationResponse(
            session_id="session-123",
            response_text="Test response"
        )

        assert response.session_id == "session-123"
        assert response.response_text == "Test response"

    def test_invocation_response_error_case(self):
        """Test InvocationResponse for error scenarios."""
        response = InvocationResponse(
            sessionId="session-error",
            confidence=0.0,
            responseText="Service temporarily unavailable",
            success=False,
            message="Kendra service error",
            execution_time_ms=5000.0
        )

        assert response.success is False
        assert response.confidence == 0.0
        assert "unavailable" in response.response_text
        assert "error" in response.message


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_health_response_defaults(self):
        """Test HealthResponse default values."""
        response = HealthResponse()

        assert response.status == "ok"
        assert isinstance(response.timestamp, datetime)

    def test_health_response_custom_status(self):
        """Test HealthResponse with custom status."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        response = HealthResponse(status="healthy", timestamp=custom_time)

        assert response.status == "healthy"
        assert response.timestamp == custom_time

    def test_health_response_various_statuses(self):
        """Test HealthResponse with various status values."""
        statuses = ["ok", "healthy", "degraded", "unhealthy", ""]

        for status in statuses:
            response = HealthResponse(status=status)
            assert response.status == status


class TestSchemaIntegration:
    """Tests for schema integration and serialization."""

    def test_invocation_request_serialization(self):
        """Test InvocationRequest JSON serialization."""
        request = InvocationRequest(
            sessionId="session-123",
            userPrompt="Test prompt",
            context=VALID_CONTEXT,
        )

        json_data = request.model_dump(by_alias=True)

        assert json_data["sessionId"] == "session-123"
        assert json_data["userPrompt"] == "Test prompt"
        assert json_data["context"]["productId"] == "6"

    def test_invocation_response_serialization(self):
        """Test InvocationResponse JSON serialization."""
        response = InvocationResponse(
            sessionId="session-123",
            confidence=8.0,
            responseText="Test response"
        )

        json_data = response.model_dump(by_alias=True)

        assert json_data["sessionId"] == "session-123"
        assert json_data["confidence"] == 8.0
        assert json_data["responseText"] == "Test response"
        assert json_data["success"] is True
        assert "timestamp" in json_data

    def test_health_response_serialization(self):
        """Test HealthResponse JSON serialization."""
        response = HealthResponse(status="healthy")

        json_data = response.model_dump()

        assert json_data["status"] == "healthy"
        assert "timestamp" in json_data

    def test_schema_deserialization(self):
        """Test schema deserialization from JSON."""
        request_data = {
            "sessionId": "session-456",
            "userPrompt": "Deserialized prompt",
            "context": VALID_CONTEXT,
        }

        request = InvocationRequest(**request_data)
        assert request.session_id == "session-456"
        assert request.user_prompt == "Deserialized prompt"
        assert request.context.productId == "6"

        response_data = {
            "sessionId": "session-456",
            "confidence": 7.5,
            "responseText": "Deserialized response",
            "success": True,
            "message": "",
            "execution_time_ms": 200.0
        }

        response = InvocationResponse(**response_data)
        assert response.session_id == "session-456"
        assert response.confidence == 7.5
        assert response.response_text == "Deserialized response"

    def test_config_populate_by_name(self):
        """Test that populate_by_name config works correctly."""
        request1 = InvocationRequest(sessionId="test", userPrompt="test", context=VALID_CONTEXT)
        request2 = InvocationRequest(session_id="test", user_prompt="test", context=VALID_CONTEXT)

        assert request1.session_id == request2.session_id
        assert request1.user_prompt == request2.user_prompt
        assert request1.context.productId == request2.context.productId
