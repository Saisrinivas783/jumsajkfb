# Logging Guide

## Overview

The Orchestrator Agent implements structured logging with centralized configuration, contextual information, and appropriate log levels for monitoring, debugging, and operational visibility.

## Logging Architecture

### Centralized Configuration
- **Module**: `src/utils/logging.py`
- **Initialization**: Configured once at application startup in `main.py`
- **Format**: Standardized format across all components
- **Output**: Stdout for container/cloud deployment compatibility

### Logger Factory Pattern
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
```

## Log Format

**Standard Format:**
```
2024-01-15 10:30:00 | INFO     | src.graph.orchestrator | Invocation started: session=sess-001
```

**Format Components:**
- **Timestamp**: `YYYY-MM-DD HH:MM:SS`
- **Level**: `DEBUG`, `INFO`, `WARNING`, `ERROR` (8-char padded)
- **Module**: Full module path (e.g., `src.graph.orchestrator`)
- **Message**: Contextual log message

## Log Levels

### DEBUG
**Purpose**: Detailed execution flow and internal state
**Visibility**: Only when `LOG_LEVEL=DEBUG`

**Examples:**
```python
logger.debug(f"Query: {user_query}")
logger.debug(f"Available tools: {list(state.registry.keys())}")
logger.debug("Invoking LLM for intent analysis")
logger.debug(f"Model: {llm.model_id}")
logger.debug(f"LLM params: temperature={actual_temp}, max_tokens={actual_tokens}")
```

### INFO
**Purpose**: Key operational events and workflow progress
**Visibility**: Default level, always visible in production

**Examples:**
```python
logger.info("→ intent_analyzer")
logger.info("← intent_analyzer: tool=IBTAgent, confidence=8.5")
logger.info(f"OrchestratorAgent initialized with {len(self.registry)} tools")
logger.info(f"Invocation started: session={payload.session_id}")
logger.info(f"Tool registry loaded: {len(tools)} tools")
```

### WARNING
**Purpose**: Recoverable issues and validation problems
**Visibility**: Always logged, indicates potential issues

**Examples:**
```python
logger.warning("Empty user prompt received")
```

### ERROR
**Purpose**: Exceptions, failures, and critical issues
**Visibility**: Always logged with full context and stack traces

**Examples:**
```python
logger.error(f"Workflow error: {e.message}")
logger.error(f"Unexpected error: {e}", exc_info=True)
logger.error(f"Failed to parse tool at index {idx}: {e}", exc_info=True)
logger.error(f"S3 error ({error_code}) loading s3://{bucket}/{key}: {e}", exc_info=True)
```

## Logging Patterns

### 1. Workflow Node Logging

**Entry/Exit Pattern:**
```python
def intent_node(state: OrchestratorState) -> OrchestratorState:
    logger.info("→ intent_analyzer")
    # ... processing ...
    logger.info(f"← intent_analyzer: tool={parsed.selected_tool}, confidence={parsed.confidence_score:.1f}")
```

**Utility Functions:**
```python
from src.utils.logging import log_node_entry, log_node_exit

log_node_entry(logger, "intent_analyzer", query=user_query)
log_node_exit(logger, "intent_analyzer", tool=selected_tool, confidence=confidence)
```

### 2. Request Lifecycle Logging

**Start:**
```python
logger.info(f"Invocation started: session={payload.session_id}")
logger.debug(f"Query: {payload.user_prompt[:100]}{'...' if len(payload.user_prompt) > 100 else ''}")
```

**Completion:**
```python
logger.info(
    f"Invocation completed: session={payload.session_id}, "
    f"tool={selected_tool_response.tool_name if selected_tool_response else 'none'}, "
    f"confidence={confidence:.1f}, "
    f"time={execution_time_ms:.0f}ms"
)
```

### 3. Error Logging with Context

**Exception Handling:**
```python
try:
    # ... operation ...
except SpecificError as e:
    logger.error(f"Specific error context: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise GenericError("User-friendly message")
```

**Stack Traces:**
- Use `exc_info=True` for full stack traces
- Only in ERROR level logs
- Never expose stack traces to users

### 4. Performance Logging

**Execution Time:**
```python
start_time = time.time()
# ... processing ...
execution_time_ms = (time.time() - start_time) * 1000
logger.info(f"Operation completed: time={execution_time_ms:.0f}ms")
```

### 5. Configuration and Initialization

**Service Startup:**
```python
logger.info(f"OrchestratorAgent initialized with {len(self.registry)} tools")
logger.info(f"Bedrock client initialized: region={self.settings.aws_region}")
logger.info(f"LLM ready: {actual_model_id}")
```

**Debug Configuration:**
```python
logger.debug(f"Bedrock config: read_timeout={timeout}s, connect_timeout={connect}s")
logger.debug(f"Available tools: {tool_names}")
```

## Third-Party Library Noise Reduction

**Suppressed Loggers:**
```python
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
```

## Configuration

### Environment Variables
```env
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
```

### Programmatic Configuration
```python
from src.utils.logging import configure_logging

# Custom configuration
configure_logging(
    level="DEBUG",
    format_string="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
```

## Contextual Information

### Session Tracking
- All requests include `session_id` in logs
- Enables tracing requests across workflow nodes
- Facilitates debugging and monitoring

### Tool Execution Context
```python
logger.info(f"← tool_executor: {tool_name} success")
logger.debug(f"Response length: {len(response_text)} chars")
```

### LLM Interaction Context
```python
logger.debug(f"Model: {llm.model_id}")
logger.debug(f"Reasoning: {parsed.reasoning}")
```

## Log Analysis Examples

### Successful Request Flow
```
2024-01-15 10:30:00 | INFO     | src.graph.orchestrator | Invocation started: session=sess-001
2024-01-15 10:30:00 | DEBUG    | src.graph.orchestrator | Query: What are my dental benefits?
2024-01-15 10:30:00 | INFO     | src.graph.nodes.intent_analyzer | → intent_analyzer
2024-01-15 10:30:00 | DEBUG    | src.graph.nodes.intent_analyzer | Available tools: ['IBTAgent', 'ClaimsAgent']
2024-01-15 10:30:01 | INFO     | src.graph.nodes.intent_analyzer | ← intent_analyzer: tool=IBTAgent, confidence=8.5
2024-01-15 10:30:01 | INFO     | src.graph.nodes.tool_executor | → tool_executor
2024-01-15 10:30:02 | INFO     | src.graph.nodes.tool_executor | ← tool_executor: IBTAgent success
2024-01-15 10:30:02 | INFO     | src.graph.orchestrator | Invocation completed: session=sess-001, tool=IBTAgent, confidence=8.5, time=1250ms
```

### Error Scenario
```
2024-01-15 10:30:00 | INFO     | src.graph.orchestrator | Invocation started: session=sess-002
2024-01-15 10:30:00 | INFO     | src.graph.nodes.intent_analyzer | → intent_analyzer
2024-01-15 10:30:01 | ERROR    | src.graph.nodes.intent_analyzer | LLM failure: Connection timeout
Traceback (most recent call last):
  ...
2024-01-15 10:30:01 | ERROR    | src.graph.orchestrator | Workflow error: Intent analysis failed
```

## Monitoring and Observability

### Key Metrics from Logs
- **Request Volume**: Count of "Invocation started" messages
- **Success Rate**: Ratio of "completed" vs "error" messages
- **Performance**: Execution time distributions
- **Tool Usage**: Frequency of tool selections
- **Error Patterns**: Common error types and frequencies

### Log Aggregation
- **Format**: JSON-compatible for structured logging systems
- **Fields**: Timestamp, level, module, message, session_id
- **Integration**: Compatible with CloudWatch, ELK Stack, Splunk

## Best Practices

1. **Use appropriate log levels** - Don't log sensitive data at INFO level
2. **Include context** - Session IDs, tool names, confidence scores
3. **Structured messages** - Consistent format for parsing
4. **Performance logging** - Track execution times
5. **Error context** - Full stack traces for debugging
6. **Avoid log spam** - Suppress noisy third-party libraries
7. **Security** - Never log credentials or sensitive user data
8. **Correlation** - Use session IDs to trace request flows