# IBT Agent - Complete Documentation

## Overview
The IBT (Insurance Benefits Tool) Agent is an AI-powered service that processes insurance benefits queries using AWS Kendra semantic search.

## Architecture
- **Direct Kendra Processing**: Product-filtered AWS Kendra semantic search
- **Department Filtering**: Maps department IDs to specific plan types and brochures
- **AWS Integration**: Uses Kendra for search
- **FastAPI Framework**: RESTful API with automatic documentation

## Key Features
- ✅ **Department-Based Filtering**: Results filtered by plan type and brochure
- ✅ **NCCT ID Response Format**: Returns arrays of NCCT IDs for direct integration
- ✅ **Role-Based AWS Access**: Supports IAM role assumption for cross-account access
- ✅ **Comprehensive Testing**: Full test coverage
- ✅ **Clean Architecture**: No fallback logic, streamlined codebase

## API Endpoints

### Health Check
```http
GET /IbtAgent/v2/ping
```

### Process Benefits Query
```http
POST /IbtAgent/v2/invocations
Content-Type: application/json

{
  "userPrompt": "What are my dental benefits?",
  "sessionId": "sess-001",
  "context": {
    "userName": "John Doe",
    "userType": "member",
    "productId": "1"
  }
}
```

## Department Mapping
- **1, 4**: FEHB Standard/Basic plans
- **6**: FEHB Blue Focus plan  
- **7, 8**: PSHB Standard/Basic plans
- **9**: PSHB Blue Focus plan

## Response Formats

Returns an array of NCCT IDs:
```json
{
  "sessionId": "sess-001",
  "responseText": ["NCCT123", "NCCT456"],
  "confidence": 8.0,
  "success": true
}
```

## Configuration

### Environment Variables
```bash
# AWS Configuration
AWS_REGION=us-east-1
KENDRA_INDEX_ID=your-kendra-index-id
KENDRA_ROLE_ARN=arn:aws:iam::ACCOUNT:role/KendraRole
```

## Development

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --reload --port 8000
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
```

### Docker Deployment
```bash
# Build image
docker build -t ibt-agent .

# Run container
docker run -p 8000:8000 --env-file .env ibt-agent
```

## Recent Changes

### Fallback Logic Removal (April 2026)
- ✅ Removed all fallback search mechanisms
- ✅ Simplified error handling to return empty results
- ✅ Cleaned up 24 fallback-related test cases
- ✅ Reduced test execution time from ~4s to ~2.8s
- ✅ Maintained 100% test pass rate (138 tests)

### NCCT ID Response Format (March 2026)
- ✅ Direct mode now returns arrays of NCCT IDs
- ✅ LLM mode returns HTML formatted strings
- ✅ Department-based filtering implemented
- ✅ Comprehensive test coverage added

### Architecture Cleanup (March 2026)
- ✅ Consolidated duplicate service files
- ✅ Removed debug and demo scripts
- ✅ Streamlined configuration management
- ✅ Updated API documentation

## File Structure
```
ibt-agent/
├── src/
│   ├── agent/           # Hybrid agent implementation
│   ├── api/             # FastAPI routes and dependencies
│   ├── config/          # Settings and constants
│   ├── schemas/         # Pydantic data models
│   ├── services/        # AWS service integrations
│   └── utils/           # Logging utilities
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── scripts/             # HTTP request examples
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # Basic project information
```

## Troubleshooting

### Common Issues
1. **AWS Credentials**: Ensure proper IAM roles and permissions
2. **Kendra Access**: Verify index ID and region configuration
3. **Empty Results**: Check department ID mapping and query format

## Performance
- **Test Execution**: Fast, comprehensive test coverage
- **API Response Time**: Typically < 2 seconds for direct mode
- **Memory Usage**: Optimized for container deployment
- **Concurrent Requests**: Supports multiple simultaneous queries

## Security
- **IAM Role Assumption**: Secure cross-account access
- **Environment Variables**: Sensitive configuration externalized
- **Input Validation**: Pydantic schema validation
- **Error Handling**: No sensitive information in error responses

---

**Last Updated**: April 2026  
**Version**: 2.0.0  
**Test Coverage**: Comprehensive test coverage passing