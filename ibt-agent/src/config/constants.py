"""Shared constants to eliminate string duplication across the IBT agent."""

# API Endpoints
API_PREFIX = "/IbtAgent/v2"
INVOCATIONS_ENDPOINT = "/invocations"
PING_ENDPOINT = "/ping"
HEALTH_ENDPOINT = "/health"

# Service Information
SERVICE_NAME = "IBT Agent - Hybrid"
SERVICE_DESCRIPTION = "Intelligent benefits inquiry service with direct Kendra search"
SERVICE_VERSION = "2.0.0"

# HTTP Status Messages
STATUS_OK = "ok"
STATUS_HEALTHY = "healthy"

# Default Values
DEFAULT_CONFIDENCE = 0.0
DEFAULT_SUCCESS = True
DEFAULT_MESSAGE = ""
DEFAULT_EXECUTION_TIME = 0.0
