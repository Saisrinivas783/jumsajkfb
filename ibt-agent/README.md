# IBT Agent

Insurance Benefits Tool Agent - A FastAPI service that processes insurance benefits queries using AWS Kendra semantic search.

## Overview

IBT Agent is an intelligent benefits inquiry service that handles insurance benefits questions through direct AWS Kendra semantic search, following FEPOC API guidelines.

## Features

- **Direct Kendra Search**: AWS Kendra semantic search with NCCT_ID extraction
- **FEPOC API Compliance**: Proper naming conventions and response formats
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

   # Logging
   LOG_LEVEL=INFO
   ```

## Usage

### Start the Server
```bash
python -m src.main
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/dxagent/ibt/invocations`
Process insurance benefits queries using direct Kendra search.

**Request:**
```json
{
  "userPrompt": "What are my dental benefits?",
  "sessionId": "session-123",
  "context": {
    "userName": "John Doe",
    "userType": "member",
    "productId": "1"
  }
}
```

**Response:**
```json
{
  "sessionId": "session-123",
  "confidence": 8.0,
  "responseText": ["NCCT123", "NCCT456"],
  "success": true,
  "execution_time_ms": 1250.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET `/dxagent/ibt/ping`
Health check endpoint for monitoring.

## Architecture

### IBT Agent (`src/agent/hybrid_ibt.py`)
- Extracts product ID from request context
- Queries AWS Kendra with product-based attribute filtering
- Handles request/response processing

### Kendra Service (`src/services/kendra_service.py`)
- Direct AWS Kendra integration
- Extracts NCCT_ID and Service_Name from document attributes
- Returns an array of NCCT ID strings

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | us-east-1 | AWS region for services |
| `KENDRA_INDEX_ID` | - | AWS Kendra index identifier |
| `KENDRA_ROLE_ARN` | - | IAM role ARN for Kendra access (optional) |
| `KENDRA_SESSION_NAME` | ibt-agent-kendra | Session name for role assumption |
| `KENDRA_ROLE_DURATION` | 3600 | Role session duration in seconds |
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
- **boto3**: AWS SDK for Kendra
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## Development

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