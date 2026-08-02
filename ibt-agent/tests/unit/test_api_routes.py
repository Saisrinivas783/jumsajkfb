"""Clean unit tests for API routes."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app

class TestInvocationsRoute:
    """Tests for invocations API route."""
    
    def setup_method(self):
        """Setup test client."""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_invocations_invalid_request(self):
        """Test invocation with invalid request data."""
        # Missing required fields
        response = self.client.post("/IbtAgent/v2/invocations", json={
            "sessionId": "test-session"
            # Missing userPrompt
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_invocations_empty_request(self):
        """Test invocation with empty request."""
        response = self.client.post("/IbtAgent/v2/invocations", json={})
        
        assert response.status_code == 422  # Validation error

class TestHealthRoute:
    """Tests for health check routes."""
    
    def setup_method(self):
        """Setup test client."""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_ping_endpoint(self):
        """Test ping endpoint."""
        response = self.client.get("/IbtAgent/v2/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_endpoint(self):
        """Test detailed health endpoint."""
        response = self.client.get("/IbtAgent/v2/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ibt agent-hybrid"
        assert data["kendra_index_id"] == ""
        assert data["aws_region"] == "us-east-1"

class TestAppIntegration:
    """Tests for FastAPI app integration."""
    
    def setup_method(self):
        """Setup test client."""
        self.app = create_app()
        self.client = TestClient(self.app)
    
    def test_app_creation(self):
        """Test app creation and basic properties."""
        app = create_app()
        
        assert app.title == "IBT Agent - Hybrid"
        assert app.version == "2.0.0"
        assert "Intelligent benefits inquiry service" in app.description
    
    def test_app_routes_registered(self):
        """Test that all routes are properly registered."""
        routes = [route.path for route in self.app.routes]

        # Check required routes exist
        assert "/IbtAgent/v2/invocations" in routes
        assert "/IbtAgent/v2/ping" in routes
        assert "/IbtAgent/v2/health" in routes
    
    def test_app_openapi_schema(self):
        """Test OpenAPI schema generation."""
        response = self.client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Verify basic schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "IBT Agent - Hybrid"
        assert "paths" in schema
        
        # Verify required endpoints in schema
        paths = schema["paths"]
        assert "/IbtAgent/v2/invocations" in paths
        assert "/IbtAgent/v2/ping" in paths
        assert "/IbtAgent/v2/health" in paths
    
    def test_app_docs_endpoint(self):
        """Test that docs endpoint is accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200
    
    def test_app_redoc_endpoint(self):
        """Test that redoc endpoint is accessible."""
        response = self.client.get("/redoc")
        assert response.status_code == 200