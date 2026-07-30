# IBT Agent

Insurance Benefits Tool Agent - A FastAPI service that processes insurance benefits queries using Langchain tools and AWS Bedrock integration with direct Kendra search.

## Overview

IBT Agent is an intelligent benefits inquiry service that handles insurance benefits questions through Langchain tool orchestration. It uses AWS Bedrock for LLM processing and direct AWS Kendra integration for semantic search, following FEPOC API guidelines.

## Features

- **Langchain Tools**: Uses `@tool` decorator for modular functionality
- **AWS Bedrock Integration**: Claude models for intelligent processing
- **Direct Kendra Search**: AWS Kendra semantic search with NCCT_ID extraction
- **FEPOC API Compliance**: Proper naming conventions and response formats
- **HTML Link Generation**: Creates `<a href='{{NCCT_ID}}'>{{Service_Name}}</a>` format
- **Simple Logging**: Self-contained logging without external dependencies

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables** (create `.env` file):
   ```bash
   # AWS Configuration
   AWS_REGION=us-east-1
   KENDRA_INDEX_ID=your-kendra-index-id
   
   # Kendra Role Configuration (Optional)
   # If specified, the agent will assume this role to access Kendra
   # KENDRA_ROLE_ARN=arn:aws:iam::123456789012:role/KendraAccessRole
   # KENDRA_SESSION_NAME=ibt-agent-kendra-session
   # KENDRA_ROLE_DURATION=3600
   
   # Bedrock Configuration
   BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
   BEDROCK_TEMPERATURE=0.0
   BEDROCK_MAX_TOKENS=1024
   
   # Mode Configuration
   USE_LLM=true  # Set to 'false' for direct Kendra, 'true' for LLM-enhanced
   
   # Logging
   LOG_LEVEL=INFO
   ```

## Usage

### Start the Server
```bash
# Standard mode
python -m src.main

# Hybrid mode (supports runtime switching)
python -m src.main_hybrid
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/dxagent/ibt/invocations`
Process insurance benefits queries using Langchain tools.

**Request:**
```json
{
  "userPrompt": "What are my dental benefits?",
  "sessionId": "session-123",
  "context": {
    "userName": "John Doe",
    "userType": "member"
  }
}
```

**Response:**
```json
{
  "sessionId": "session-123",
  "confidence": 8.0,
  "responseText": "Here are the relevant benefits: <a href='NCCT123'>Dental Coverage</a>, <a href='NCCT456'>Preventive Care</a>",
  "success": true,
  "execution_time_ms": 1250.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET `/dxagent/ibt/ping`
Health check endpoint for monitoring.

### POST `/dxagent/ibt/mode` (Hybrid only)
Switch between direct Kendra and LLM-enhanced modes.

**Request:**
```json
{"use_llm": false}
```

### GET `/dxagent/ibt/mode` (Hybrid only)
Get current mode information.

## Architecture

### IBT Agent (`src/agent/ibt.py`)
- Uses `create_tool_calling_agent` with Bedrock LLM
- Orchestrates tool execution through `AgentExecutor`
- Handles request/response processing

### Tools (`src/tools/benefits.py`)
```python
@tool
def search_benefits(query: str) -> str:
    """Search for insurance benefits and coverage information."""
```

### Kendra Service (`src/services/kendra.py`)
- Direct AWS Kendra integration
- Extracts NCCT_ID and Service_Name from document attributes
- Returns structured results for HTML link generation

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | us-east-1 | AWS region for services |
| `KENDRA_INDEX_ID` | - | AWS Kendra index identifier |
| `KENDRA_ROLE_ARN` | - | IAM role ARN for Kendra access (optional) |
| `KENDRA_SESSION_NAME` | ibt-agent-kendra | Session name for role assumption |
| `KENDRA_ROLE_DURATION` | 3600 | Role session duration in seconds |
| `BEDROCK_MODEL_ID` | us.anthropic.claude-haiku-4-5-20251001-v1:0 | Bedrock model |
| `BEDROCK_TEMPERATURE` | 0.0 | LLM temperature |
| `BEDROCK_MAX_TOKENS` | 1024 | Maximum response tokens |
| `USE_LLM` | true | Enable LLM processing (false = direct Kendra) |
| `LOG_LEVEL` | INFO | Logging level |

## AWS Role Configuration

### Direct Access (Default)
The agent uses your default AWS credentials to access Kendra directly.

### Role Assumption (Recommended)
For enhanced security and cross-account access, configure role assumption:

1. **Set environment variables:**
   ```bash
   KENDRA_ROLE_ARN=arn:aws:iam::123456789012:role/KendraAccessRole
   KENDRA_SESSION_NAME=ibt-agent-kendra-session
   KENDRA_ROLE_DURATION=3600
   ```

2. **Create IAM role with trust policy:**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "AWS": "arn:aws:iam::SOURCE-ACCOUNT:role/IBTAgentRole"
         },
         "Action": "sts:AssumeRole"
       }
     ]
   }
   ```

3. **Attach Kendra permissions to target role:**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "kendra:Query",
           "kendra:DescribeIndex",
           "kendra:GetQuerySuggestions"
         ],
         "Resource": [
           "arn:aws:kendra:us-east-1:ACCOUNT-ID:index/INDEX-ID"
         ]
       }
     ]
   }
   ```

4. **Grant source role/user permission to assume target role:**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": "sts:AssumeRole",
         "Resource": "arn:aws:iam::TARGET-ACCOUNT:role/KendraAccessRole"
       }
     ]
   }
   ```

## Error Handling

### Fallback Messages
- `no_results_found`: "I couldn't find specific information about your inquiry. Please contact customer service for detailed assistance."
- `service_unavailable`: "I'm currently experiencing technical difficulties accessing benefit information. Please try again in a few moments."

## Dependencies

- **FastAPI**: Web framework
- **Langchain**: Tool framework and agent orchestration
- **langchain-aws**: AWS Bedrock integration
- **boto3**: AWS SDK for Kendra
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## Development

### Adding New Tools
1. Create tool with `@tool` decorator in `src/tools/`
2. Import and add to `self.tools` list in IBTAgent
3. Update agent prompt if needed

### Logging
- Simple stdout logging via `src/utils/logging.py`
- No external logging dependencies
- Configurable log levels

## Deployment

### Environment Setup
- Ensure AWS credentials are configured
- Set required environment variables
- Configure Kendra index access permissions

### Health Monitoring
- Use `/dxagent/ibt/ping` for health checks
- Monitor execution times in response metadata
- Check logs for error patterns