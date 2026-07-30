"""Shared pytest fixtures for all tests."""

import pytest
from src.schemas.api import InvocationContext


@pytest.fixture
def mock_context():
    """
    Fixture providing a valid InvocationContext for testing.

    Use this fixture in tests that create OrchestratorState instances,
    since context is required by the standardized API contract.
    """
    return InvocationContext(
        userName="test_user",
        userType="member",
        source="TestPage",
        promptId=None,
        productId="PROD-001"
    )
