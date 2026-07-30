"""Integration tests for FastAPI exception handlers."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api.app import create_app
from src.api.dependencies import get_orchestrator
from src.schemas.api import InvocationResponse

VALID_PAYLOAD = {
    "userPrompt": "What are my benefits?",
    "sessionId": "sess-001",
    "context": {
        "userName": "john_doe",
        "userType": "member",
        "source": "IBTPage",
        "productId": "PROD-001",
    },
}


@pytest.fixture
def mock_orchestrator():
    mock = MagicMock()
    mock.handle_invocation.return_value = InvocationResponse(
        sessionId="sess-001",
        responseText=["Test response"],
        metadata=[],
    )
    return mock


@pytest.fixture
def client(mock_orchestrator):
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestRequestValidationErrors:
    """Tests for RequestValidationError handler."""

    def test_missing_context_returns_400(self, client):
        payload = {"userPrompt": "Hello", "sessionId": "sess-001"}
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "context" in data["message"]
        assert "productId" in data["message"]

    def test_missing_context_userName_returns_specific_message(self, client):
        payload = {
            "userPrompt": "Hello",
            "sessionId": "sess-001",
            "context": {"userType": "member", "source": "IBTPage", "productId": "PROD-001"},
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "'context.userName' is required" in response.json()["message"]

    def test_missing_context_userType_returns_specific_message(self, client):
        payload = {
            "userPrompt": "Hello",
            "sessionId": "sess-001",
            "context": {"userName": "john", "source": "IBTPage", "productId": "PROD-001"},
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "'context.userType' is required" in response.json()["message"]

    def test_missing_context_source_returns_specific_message(self, client):
        payload = {
            "userPrompt": "Hello",
            "sessionId": "sess-001",
            "context": {"userName": "john", "userType": "member", "productId": "PROD-001"},
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "'context.source' is required" in response.json()["message"]

    def test_missing_context_productId_returns_specific_message(self, client):
        payload = {
            "userPrompt": "Hello",
            "sessionId": "sess-001",
            "context": {"userName": "john", "userType": "member", "source": "IBTPage"},
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "'context.productId' is required" in response.json()["message"]

    def test_empty_context_userName_returns_cannot_be_empty(self, client):
        payload = {
            "userPrompt": "Hello",
            "sessionId": "sess-001",
            "context": {
                "userName": "",
                "userType": "member",
                "source": "IBTPage",
                "productId": "PROD-001",
            },
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["message"]

    def test_empty_context_productId_returns_cannot_be_empty(self, client):
        payload = {
            "userPrompt": "Hello",
            "sessionId": "sess-001",
            "context": {
                "userName": "john",
                "userType": "member",
                "source": "IBTPage",
                "productId": "   ",
            },
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["message"]

    def test_missing_userPrompt_returns_400(self, client):
        payload = {
            "sessionId": "sess-001",
            "context": {
                "userName": "john",
                "userType": "member",
                "source": "IBTPage",
                "productId": "PROD-001",
            },
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert response.json()["success"] is False
        assert "required" in response.json()["message"]

    def test_empty_userPrompt_returns_cannot_be_empty(self, client):
        payload = {
            "userPrompt": "",
            "sessionId": "sess-001",
            "context": {
                "userName": "john",
                "userType": "member",
                "source": "IBTPage",
                "productId": "PROD-001",
            },
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["message"]

    def test_missing_sessionId_uses_generic_required_message(self, client):
        payload = {
            "userPrompt": "Hello",
            "context": {
                "userName": "john",
                "userType": "member",
                "source": "IBTPage",
                "productId": "PROD-001",
            },
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        assert "required" in response.json()["message"]

    def test_error_response_has_all_required_fields(self, client):
        response = client.post("/OrchestratorAgent/v2/invocations", json={"userPrompt": "Hello"})
        assert response.status_code == 400
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "timestamp" in data
        assert "sessionId" in data
        assert "responseText" in data
        assert "metadata" in data
        assert data["success"] is False
        assert data["sessionId"] == ""
        assert data["responseText"] == ""
        assert data["metadata"] == []


    def test_type_mismatch_uses_generic_validation_message(self, client):
        """Lines 80-82: non-missing, non-value_error validation failure."""
        # Sending a string for 'context' (expects dict) triggers a type error
        payload = {
            "userPrompt": "Hello",
            "sessionId": "sess-001",
            "context": "not-a-dict",
        }
        response = client.post("/OrchestratorAgent/v2/invocations", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "message" in data


class TestHTTPExceptionHandler:
    """Tests for StarletteHTTPException handler."""

    def test_nonexistent_route_returns_404(self, client):
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "message" in data

    def test_404_response_has_all_required_fields(self, client):
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "timestamp" in data


class TestGenericExceptionHandler:
    """Tests for the catch-all Exception handler."""

    def test_unexpected_exception_from_orchestrator_returns_500(self, mock_orchestrator):
        mock_orchestrator.handle_invocation.side_effect = RuntimeError("Unexpected crash")
        app = create_app()
        app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "unexpected" in data["message"].lower()

    def test_500_response_has_all_required_fields(self, mock_orchestrator):
        mock_orchestrator.handle_invocation.side_effect = Exception("boom")
        app = create_app()
        app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        assert response.status_code == 500
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "timestamp" in data
        assert data["sessionId"] == ""
        assert data["metadata"] == []