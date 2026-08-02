"""Shared pytest fixtures for IBT agent tests."""

import pytest
from unittest.mock import MagicMock, patch
import os

# Disable role assumption for all tests
@pytest.fixture(autouse=True)
def disable_role_assumption():
    """Automatically disable role assumption for all tests."""
    # Mock boto3.client to prevent AWS calls during testing
    # Only mock if kendra_service is not being explicitly mocked
    with patch('boto3.client') as mock_boto_client:
        mock_kendra = MagicMock()
        mock_sts = MagicMock()
        
        def client_side_effect(service_name, **kwargs):
            if service_name == 'kendra':
                return mock_kendra
            elif service_name == 'sts':
                return mock_sts
            return MagicMock()
        
        mock_boto_client.side_effect = client_side_effect
        
        # Also provide a default mock for kendra_service if not explicitly mocked
        with patch('src.services.kendra_service.get_kendra_service') as mock_get_kendra:
            mock_service = MagicMock()
            mock_service.search.return_value = {
                'success': True,
                'results': [
                    {
                        'ncct_id': 'NCCT123',
                        'service_name': 'Dental Coverage',
                        'confidence_score': 'HIGH'
                    }
                ]
            }
            mock_get_kendra.return_value = mock_service
            yield

@pytest.fixture
def mock_context():
    """Valid context for testing."""
    return {
        "userName": "test_user",
        "userType": "member",
        "productId": "1"
    }

@pytest.fixture
def valid_query_payload():
    """Valid query request payload."""
    return {
        "userPrompt": "What are my dental benefits?",
        "sessionId": "sess-001"
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
    mock.process_query.return_value = {
        "sessionId": "sess-001",
        "confidence": 8.0,
        "responseText": "Here are your benefits: <a href='NCCT123'>Dental Coverage</a>",
        "success": True,
        "execution_time_ms": 250.5,
        "timestamp": "2024-01-15T10:30:00Z"
    }
    return mock