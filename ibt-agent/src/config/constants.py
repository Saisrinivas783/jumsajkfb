"""Shared constants to eliminate string duplication across the IBT agent."""

# API Endpoints
API_PREFIX = "/IbtAgent/v2"
INVOCATIONS_ENDPOINT = "/invocations"
PING_ENDPOINT = "/ping"
HEALTH_ENDPOINT = "/health"
MODE_ENDPOINT = "/mode"

# Service Information
SERVICE_NAME = "IBT Agent - Hybrid"
SERVICE_DESCRIPTION = "Intelligent benefits inquiry service with configurable LLM/Direct modes"
SERVICE_VERSION = "2.0.0"

# HTTP Status Messages
STATUS_OK = "ok"
STATUS_HEALTHY = "healthy"

# Default Values
DEFAULT_CONFIDENCE = 0.0
DEFAULT_SUCCESS = True
DEFAULT_MESSAGE = ""
DEFAULT_EXECUTION_TIME = 0.0

# Kendra Search Configuration
DEFAULT_PAGE_SIZE = 10

# Document Attribute Keys
NCCT_ID_KEYS = ['NCCTID', 'NCCT_ID', 'ID', 'Document_ID', 'DocumentId']
SERVICE_NAME_KEYS = ['Service_Name', 'Title', 'Name']

# Error Messages
NO_EXCERPT_MESSAGE = "No excerpt available"
PROCESSING_ERROR_MESSAGE = "Document available but processing error occurred"