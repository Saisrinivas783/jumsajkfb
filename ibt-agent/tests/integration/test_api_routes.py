"""Integration tests for IBT agent API routes."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.config.constants import API_PREFIX

@pytest.fixture
def client(mock_hybrid_agent):
    """Test client with mocked agent."""
    # The autouse fixture in conftest.py already mocks kendra_service
    # We need to patch the dependencies after the app is created
    from src.api.app import create_app
    app = create_app()
    
    # Override the dependency after app creation
    def override_get_ibt():
        return mock_hybrid_agent
    
    # Import get_ibt and override it in the app
    from src.api.dependencies import get_ibt
    app.dependency_overrides[get_ibt] = override_get_ibt
    
    with TestClient(app) as c:
        yield c

class TestHealthRoute:
    """Tests for GET /IbtAgent/v2/ping."""
    
    def test_ping_returns_200(self, client):
        response = client.get(f"{API_PREFIX}/ping")
        assert response.status_code == 200
    
    def test_ping_returns_healthy_status(self, client):
        response = client.get(f"{API_PREFIX}/ping")
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

class TestInvocationsRoute:
    """Tests for POST /IbtAgent/v2/invocations."""
    
    def test_valid_invocation_returns_200(self, client, valid_query_payload):
        response = client.post(f"{API_PREFIX}/invocations", json=valid_query_payload)
        assert response.status_code == 200
    
    def test_valid_invocation_calls_agent(self, client, mock_hybrid_agent, valid_query_payload):
        client.post(f"{API_PREFIX}/invocations", json=valid_query_payload)
        mock_hybrid_agent.process_query.assert_called_once()
    
    def test_valid_invocation_response_structure(self, client, valid_query_payload):
        response = client.post(f"{API_PREFIX}/invocations", json=valid_query_payload)
        data = response.json()
        assert "sessionId" in data
        assert "confidence" in data
        assert "responseText" in data
        assert "success" in data
        assert "execution_time_ms" in data
        assert "timestamp" in data
    
    def test_agent_receives_correct_request(self, client, mock_hybrid_agent, valid_query_payload):
        client.post(f"{API_PREFIX}/invocations", json=valid_query_payload)
        call_args = mock_hybrid_agent.process_query.call_args
        # Check keyword arguments
        assert call_args.kwargs["user_prompt"] == "What are my dental benefits?"
        assert call_args.kwargs["session_id"] == "sess-001"
        # Context should be None when not provided in request
        assert call_args.kwargs["context"] is None
    
    def test_agent_receives_context_when_provided(self, client, mock_hybrid_agent):
        payload_with_context = {
            "userPrompt": "What are my dental benefits?",
            "sessionId": "sess-001",
            "context": {
                "userName": "John Doe",
                "userType": "member",
                "productId": "1"
            }
        }
        response = client.post(f"{API_PREFIX}/invocations", json=payload_with_context)
        assert response.status_code == 200
        
        # Verify the agent was called
        assert mock_hybrid_agent.process_query.called
        call_args = mock_hybrid_agent.process_query.call_args
        
        # Check that context is passed correctly
        context = call_args.kwargs["context"]
        assert context["userName"] == "John Doe"
        assert context["userType"] == "member"
        assert context["productId"] == "1"

class TestModeRoutes:
    """Tests for mode switching endpoints."""
    
    def test_get_mode_returns_200(self, client):
        response = client.get(f"{API_PREFIX}/mode")
        assert response.status_code == 200
    
    def test_get_mode_returns_mode_info(self, client):
        response = client.get(f"{API_PREFIX}/mode")
        data = response.json()
        assert "current_mode" in data
        assert "kendra_index_id" in data
        assert "aws_region" in data
    
    def test_set_mode_returns_200(self, client):
        response = client.post(f"{API_PREFIX}/mode", json={"use_llm": False})
        assert response.status_code == 200
    
    def test_set_mode_calls_agent(self, client, mock_hybrid_agent):
        client.post(f"{API_PREFIX}/mode", json={"use_llm": False})
        mock_hybrid_agent.set_mode.assert_called_once_with(False)