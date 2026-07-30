"""Integration tests for health and invocations routes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.app import _lifespan, create_app
from src.api.dependencies import get_orchestrator
from src.schemas.api import InvocationResponse, AgentMetadata, MetadataItem

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
        responseText=["Your benefits include dental and vision."],
        metadata=[
            AgentMetadata(
                agent="orchestrator",
                data=[
                    MetadataItem(key="confidence", value=9.0),
                    MetadataItem(key="selectedTool", value="IBTAgent"),
                    MetadataItem(key="reasoning", value="Benefit inquiry"),
                    MetadataItem(key="reformulatedQuery", value="dental and vision coverage benefits"),
                    MetadataItem(key="guardrailAction", value="NONE"),
                    MetadataItem(key="guardrailBlocked", value=False),
                ],
            )
        ],
        success=True,
        message="",
    )
    return mock


@pytest.fixture
def client(mock_orchestrator):
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    with TestClient(app) as c:
        yield c


class TestHealthRoute:
    """Tests for GET /ping."""

    def test_ping_returns_200(self, client):
        response = client.get("/OrchestratorAgent/v2/ping")
        assert response.status_code == 200

    def test_ping_returns_ok_status(self, client):
        response = client.get("/OrchestratorAgent/v2/ping")
        assert response.json() == {"status": "ok"}


class TestInvocationsRoute:
    """Tests for POST /invocations."""

    def test_valid_invocation_returns_200(self, client):
        response = client.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_valid_invocation_calls_orchestrator(self, client, mock_orchestrator):
        client.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        mock_orchestrator.handle_invocation.assert_called_once()

    def test_valid_invocation_returns_response_text(self, client):
        response = client.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        data = response.json()
        assert data["responseText"] == ["Your benefits include dental and vision."]

    def test_valid_invocation_response_has_sessionId(self, client):
        response = client.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        data = response.json()
        assert data["sessionId"] == "sess-001"

    def test_valid_invocation_response_structure(self, client):
        response = client.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        data = response.json()
        assert "success" in data
        assert "sessionId" in data
        assert "responseText" in data
        assert "metadata" in data
        assert "timestamp" in data

    def test_orchestrator_receives_correct_request(self, client, mock_orchestrator):
        client.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        call_args = mock_orchestrator.handle_invocation.call_args[0][0]
        assert call_args.user_prompt == "What are my benefits?"
        assert call_args.session_id == "sess-001"
        assert call_args.context.userName == "john_doe"
        assert call_args.context.productId == "PROD-001"

    def test_response_metadata_contains_reformulated_query(self, client):
        response = client.post("/OrchestratorAgent/v2/invocations", json=VALID_PAYLOAD)
        data = response.json()
        assert len(data["metadata"]) >= 1
        orchestrator_meta = next(m for m in data["metadata"] if m["agent"] == "orchestrator")
        keys = [item["key"] for item in orchestrator_meta["data"]]
        assert "reformulatedQuery" in keys


class TestCreateApp:
    """Tests for create_app factory."""

    def test_create_app_returns_fastapi_instance(self):
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_has_title(self):
        app = create_app()
        assert app.title == "Orchestrator Agent"


class TestLifespan:
    """Tests for startup credential warmup behavior."""

    @pytest.mark.asyncio
    async def test_lifespan_warms_credentials_before_starting_refresh(self):
        chat_models = MagicMock()
        chat_models.settings.bedrock_role_arn = "arn:aws:iam::123456789012:role/bedrock"

        with (
            patch("src.api.app.get_orchestrator"),
            patch("src.api.app.get_chat_models", return_value=chat_models),
            patch("src.api.app.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            async with _lifespan(FastAPI()):
                mock_to_thread.assert_awaited_once_with(chat_models.refresh_credentials)
                chat_models.start_credential_refresh.assert_called_once()

            chat_models.stop_credential_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_continues_when_initial_warmup_fails(self):
        chat_models = MagicMock()
        chat_models.settings.bedrock_role_arn = "arn:aws:iam::123456789012:role/bedrock"

        with (
            patch("src.api.app.get_orchestrator"),
            patch("src.api.app.get_chat_models", return_value=chat_models),
            patch("src.api.app.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            mock_to_thread.side_effect = RuntimeError("sts unavailable")

            async with _lifespan(FastAPI()):
                chat_models.start_credential_refresh.assert_called_once()

            chat_models.stop_credential_refresh.assert_called_once()


class TestGetOrchestrator:
    """Tests for the get_orchestrator dependency."""

    def test_get_orchestrator_returns_singleton(self):
        from unittest.mock import patch, MagicMock
        from src.api.dependencies import get_orchestrator

        get_orchestrator.cache_clear()
        with patch("src.api.dependencies.OrchestratorAgent") as mock_cls:
            mock_cls.return_value = MagicMock()
            instance1 = get_orchestrator()
            instance2 = get_orchestrator()
            assert instance1 is instance2
            assert mock_cls.call_count == 1
        get_orchestrator.cache_clear()