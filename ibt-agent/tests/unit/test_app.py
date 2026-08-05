"""Clean unit tests for FastAPI app factory."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.config.constants import SERVICE_NAME, SERVICE_VERSION, API_PREFIX
from src.exceptions import UpstreamServiceError

class TestCreateApp:
    """Tests for create_app function."""
    
    def test_create_app_returns_fastapi_instance(self):
        """Test create_app returns FastAPI instance."""
        app = create_app()
        
        assert isinstance(app, FastAPI)
    
    def test_create_app_configuration(self):
        """Test create_app configures app correctly."""
        app = create_app()
        
        assert app.title == SERVICE_NAME
        assert "Intelligent benefits inquiry service" in app.description
        assert app.version == SERVICE_VERSION
    
    def test_create_app_routes_included(self):
        """Test create_app includes required routes."""
        app = create_app()
        
        # Get all routes
        routes = [route.path for route in app.routes]
        
        # Check health routes
        assert f"{API_PREFIX}/ping" in routes
        assert f"{API_PREFIX}/health" in routes
        
        # Check invocation route
        assert f"{API_PREFIX}/invocations" in routes
    
    def test_create_app_route_methods(self):
        """Test create_app routes have correct HTTP methods."""
        app = create_app()
        
        # Find specific routes and check methods
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                if route.path == f"{API_PREFIX}/ping":
                    assert "GET" in route.methods
                elif route.path == f"{API_PREFIX}/invocations":
                    assert "POST" in route.methods
    
    def test_create_app_health_endpoint_works(self):
        """Test create_app health endpoint is functional."""
        app = create_app()
        client = TestClient(app)
        
        response = client.get(f"{API_PREFIX}/ping")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_create_app_multiple_instances(self):
        """Test create_app can create multiple independent instances."""
        app1 = create_app()
        app2 = create_app()
        
        # Should be different instances
        assert app1 is not app2
        
        # But should have same configuration
        assert app1.title == app2.title
        assert app1.version == app2.version
    
    def test_create_app_openapi_schema(self):
        """Test create_app generates valid OpenAPI schema."""
        app = create_app()
        
        # Generate OpenAPI schema
        schema = app.openapi()
        
        # Basic schema validation
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check info section
        assert schema["info"]["title"] == SERVICE_NAME
        assert schema["info"]["version"] == SERVICE_VERSION
        
        # Check paths exist
        paths = schema["paths"]
        assert f"{API_PREFIX}/ping" in paths
        assert f"{API_PREFIX}/invocations" in paths
        assert f"{API_PREFIX}/health" in paths
    
    def test_create_app_cors_and_middleware(self):
        """Test create_app middleware configuration."""
        app = create_app()
        
        # Check that app has middleware stack
        assert hasattr(app, 'middleware_stack')
        
        # The app should be functional with TestClient
        client = TestClient(app)
        response = client.get(f"{API_PREFIX}/ping")
        assert response.status_code == 200


class TestExceptionHandlers:
    """Tests for app-level exception handlers."""

    def test_upstream_service_error_maps_to_500(self):
        """Test UpstreamServiceError raised from a route is converted to a structured 500."""
        app = create_app()

        @app.get("/__test_upstream_error")
        def _raise_upstream_error():
            raise UpstreamServiceError("kendra", "boom")

        client = TestClient(app)
        response = client.get("/__test_upstream_error")

        assert response.status_code == 500
        assert response.json() == {"detail": "Upstream service error (kendra): boom"}

    def test_unhandled_exception_maps_to_generic_500(self):
        """Test an arbitrary unhandled exception is converted to a generic 500, not a bare crash."""
        app = create_app()

        @app.get("/__test_unhandled_error")
        def _raise_unhandled_error():
            raise ValueError("unexpected")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/__test_unhandled_error")

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
