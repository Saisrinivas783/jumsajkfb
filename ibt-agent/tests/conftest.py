"""Shared pytest fixtures for IBT agent tests."""

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import get_settings


@pytest.fixture(autouse=True)
def disable_role_assumption(monkeypatch):
    """Keep default tests off the assume-role path unless they opt in explicitly."""
    monkeypatch.delenv("KENDRA_ROLE_ARN", raising=False)
    get_settings.cache_clear()

    with patch('boto3.client') as mock_boto_client:
        mock_kendra = MagicMock()
        mock_sts = MagicMock()

        def client_side_effect(service_name, **kwargs):
            if service_name == 'kendra':
                return mock_kendra
            if service_name == 'sts':
                return mock_sts
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        with patch('src.services.kendra_service.get_kendra_service') as mock_get_kendra:
            mock_service = MagicMock()
            mock_service.get_ncct_ids_by_product.return_value = ['NCCT123']
            mock_get_kendra.return_value = mock_service
            yield

    get_settings.cache_clear()


@pytest.fixture
def mock_context():
    """Valid context for testing."""
    return {
        "userName": "test_user",
        "userType": "member",
        "source": "IBTPage",
        "productId": "6"
    }


@pytest.fixture
def valid_query_payload():
    """Valid query request payload."""
    return {
        "userPrompt": "What are my dental benefits?",
        "sessionId": "sess-001",
        "context": {
            "userName": "test_user",
            "userType": "member",
            "source": "IBTPage",
            "productId": "6"
        }
    }


@pytest.fixture
def mock_kendra_response():
    """Mock Kendra search response."""
    return {
        'ResultItems': [
            {
                'DocumentAttributes': [
                    {'Key': 'NCCT_ID', 'Value': {'StringValue': 'NCCT123'}},
                    {'Key': 'Service_Name', 'Value': {'StringValue': 'Dental Coverage'}}
                ],
                'DocumentExcerpt': {'Text': 'Comprehensive dental coverage including preventive care'},
                'ScoreAttributes': {'ScoreConfidence': 'HIGH'}
            }
        ]
    }


@pytest.fixture
def mock_hybrid_agent():
    """Mock HybridIBTAgent for testing."""
    mock = MagicMock()
    mock.kendra_index_id = "test-index"
    mock.aws_region = "us-east-1"
    mock.process_query.return_value = {
        "sessionId": "sess-001",
        "confidence": 8.0,
        "responseText": ["NCCT123"],
        "success": True,
        "execution_time_ms": 250.5,
        "timestamp": "2024-01-15T10:30:00Z",
        "mode": "direct_kendra"
    }
    return mock
