# Error Handling Guide

## Overview

The Orchestrator Agent implements a comprehensive error handling strategy with graceful degradation. All errors result in structured JSON responses with `success: false`, ensuring consistent API behavior.

## Error Response Format

All error responses follow this standard format:

```json
{
  "success": false,
  "message": "Human-readable error description",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "execution_time_ms": 0.0,
  "sessionId": "",
  "selectedTool": null,
  "confidence": 0.0,
  "responseText": ""
}
```

## Error Categories

### 1. Request Validation Errors

**Triggers:**
- Missing required fields (`userPrompt`, `sessionId`)
- Invalid JSON format
- Empty user prompt

**Handler:** `RequestValidationError` in `error_handlers.py`

**Example Response:**
```json
{
  "success": false,
  "message": "Missing required field(s): userPrompt",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "execution_time_ms": 0.0,
  "sessionId": "",
  "selectedTool": null,
  "confidence": 0.0,
  "responseText": ""
}
```

### 2. Configuration Errors

**Types:**
- `ToolRegistryError`: Tool registry loading/parsing failures
- `ConfigurationError`: Invalid configuration settings

**Triggers:**
- Missing or malformed `tools.yaml`
- Invalid tool definitions
- Missing AWS credentials
- Invalid environment variables

**Handler:** `ToolRegistryError` in `error_handlers.py`

**Example Response:**
```json
{
  "success": false,
  "message": "Service configuration error. Please try again later.",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "execution_time_ms": 0.0,
  "sessionId": "",
  "selectedTool": null,
  "confidence": 0.0,
  "responseText": ""
}
```

### 3. LLM Service Errors

**Types:**
- `LLMFailureError`: Bedrock service unavailable or timeout
- AWS authentication failures
- Model invocation errors

**Triggers:**
- AWS Bedrock service downtime
- Invalid AWS credentials
- Network connectivity issues
- Model timeout (300s default)

**Handler:** Caught in `orchestrator.py` workflow execution

**Example Response:**
```json
{
  "success": false,
  "message": "An unexpected error occurred. Please try again later.",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "execution_time_ms": 1250.5,
  "sessionId": "sess-001",
  "selectedTool": null,
  "confidence": 0.0,
  "responseText": ""
}
```

### 4. Tool Execution Errors

**Types:**
- `ToolUnavailableError`: Tool service down or unreachable
- `ToolTimeoutError`: Tool response timeout (30s default)
- `NoToolMatchError`: No applicable tool found

**Triggers:**
- Tool service downtime
- Network connectivity to tool endpoints
- Tool response timeout
- Invalid tool configuration

**Handler:** Caught in `tool_executor.py` node

**Fallback Behavior:** Routes to fallback node with appropriate message

### 5. Low Confidence Scenarios

**Types:**
- `LowConfidenceError`: Query understanding below threshold
- Ambiguous queries
- Out-of-scope requests

**Triggers:**
- Confidence score < 7.0 (configurable)
- NO_TOOL selection by LLM
- Conversational queries

**Handler:** Routes to `fallback_node` with predefined messages

**Example Response:**
```json
{
  "success": true,
  "message": "",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "execution_time_ms": 850.2,
  "sessionId": "sess-001",
  "selectedTool": {
    "toolName": "NO_TOOL",
    "confidence": 3.2,
    "reasoning": "Query too vague to match any specific tool"
  },
  "confidence": 3.2,
  "responseText": "I'm sorry, I couldn't find the right resource to help with your question. Please try rephrasing your query or contact our support team for assistance."
}
```

### 6. HTTP Errors

**Types:**
- 404 Not Found
- 405 Method Not Allowed
- Other HTTP status errors

**Handler:** `StarletteHTTPException` in `error_handlers.py`

**Example Response:**
```json
{
  "success": false,
  "message": "Not Found",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "execution_time_ms": 0.0,
  "sessionId": "",
  "selectedTool": null,
  "confidence": 0.0,
  "responseText": ""
}
```

### 7. Unexpected Errors

**Types:**
- Unhandled exceptions
- System errors
- Memory/resource issues

**Handler:** Generic `Exception` handler in `error_handlers.py`

**Behavior:**
- Logs full stack trace for debugging
- Returns generic user-friendly message
- Never exposes internal error details

## Error Flow Through LangGraph

```
START → Intent Analyzer → Guard Rails Router → [Tool Executor | Fallback] → END
  ↓           ↓                ↓                      ↓           ↓
Error     Error            Error                  Error       Error
  ↓           ↓                ↓                      ↓           ↓
Orchestrator catches all → Returns structured error response
```

**Error Propagation:**
1. **Node-level errors**: Caught by orchestrator workflow
2. **LLM errors**: Wrapped in `LLMFailureError`
3. **Tool errors**: Wrapped in tool-specific exceptions
4. **Validation errors**: Caught by FastAPI handlers
5. **All errors**: Result in structured JSON response

## Fallback Messages

Predefined messages in `fallback_node`:

```python
FALLBACK_MESSAGES = {
    "no_tool_found": "I'm sorry, I couldn't find the right resource to help with your question. Please try rephrasing your query or contact our support team for assistance.",
    "low_confidence": "I'm not entirely sure I understand your question. Could you please provide more details or rephrase your request?",
    "service_unavailable": "I'm currently experiencing technical difficulties. Please try again in a few moments or contact support if the issue persists."
}
```

## Configuration

**Timeout Settings:**
```env
BEDROCK_READ_TIMEOUT=300          # LLM response timeout
BEDROCK_CONNECT_TIMEOUT=10        # LLM connection timeout
BEDROCK_MAX_RETRIES=3             # LLM retry attempts
TOOL_TIMEOUT=30                   # Tool execution timeout
TOOL_MAX_RETRIES=3                # Tool retry attempts
```

**Confidence Thresholds:**
```env
CONFIDENCE_THRESHOLD_HIGH=7.0     # Minimum for tool execution
CONFIDENCE_THRESHOLD_LOW=5.0      # Minimum for clarification
```

## Monitoring & Logging

**Error Logging:**
- All errors logged with full context
- Structured logging with session IDs
- Stack traces for debugging (not exposed to users)
- Performance metrics (execution time)

**Log Levels:**
- `ERROR`: All exceptions and failures
- `WARNING`: Low confidence scenarios
- `INFO`: Successful operations and routing decisions
- `DEBUG`: Detailed execution flow

## Best Practices

1. **Never expose internal errors** to users
2. **Always return structured responses** with consistent format
3. **Log detailed information** for debugging
4. **Use appropriate HTTP status codes** (200 for graceful errors)
5. **Provide actionable error messages** when possible
6. **Implement circuit breakers** for external service calls
7. **Monitor error rates** and patterns for system health