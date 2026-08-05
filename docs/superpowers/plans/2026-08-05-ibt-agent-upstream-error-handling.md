# IBT Agent Upstream Error Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all AWS/Kendra dependency failures in `ibt-agent` (including throttling) surface as HTTP 500 via a new `UpstreamServiceError`, instead of being swallowed into an HTTP 200 "degraded" response.

**Architecture:** Add `UpstreamServiceError(service, message)` to `src/exceptions.py`. Raise it from `KendraService` wherever `RuntimeError`/`QueryLimitExceededError` are currently raised. Stop catching it in `HybridIBTAgent.process_query` so it propagates to FastAPI. Register two exception handlers on the FastAPI app: one for `UpstreamServiceError` → 500, one catch-all `Exception` → 500.

**Tech Stack:** Python, FastAPI, boto3, pytest, unittest.mock.

## Global Constraints

- Do not modify `KendraService`'s client-refresh locking, retry/pool-size config, or timing/instrumentation logs — spec is scoped to error types only.
- `query_limit_exceeded` message key must be fully removed from `src/config/messages.py` (spec: "remove from the messages also").
- Kendra throttling (`ThrottlingException`) must raise `UpstreamServiceError`, exactly like every other Kendra `ClientError` — no special-cased 200 response for it anymore.
- `KendraSearchError` and `QueryProcessingError` in `src/agent/hybrid_ibt.py` are dead code (defined, never raised) and must be deleted, not left in place.
- All new/changed exception raises use `from e` to preserve the original traceback chain, matching the existing style in `kendra_service.py`.

---

### Task 1: Add `UpstreamServiceError` to `src/exceptions.py`

**Files:**
- Modify: `ibt-agent/src/exceptions.py`
- Test: `ibt-agent/tests/unit/test_exceptions.py` (new file)

**Interfaces:**
- Produces: `UpstreamServiceError(service: str, message: str)` — subclass of `IBTError`, with `.service` and `.message` attributes (`.message` inherited from `IBTError.__init__`).

- [ ] **Step 1: Write the failing test**

Create `ibt-agent/tests/unit/test_exceptions.py`:

```python
"""Unit tests for custom exceptions."""

from src.exceptions import IBTError, UpstreamServiceError


class TestUpstreamServiceError:
    """Tests for UpstreamServiceError."""

    def test_is_ibt_error_subclass(self):
        assert issubclass(UpstreamServiceError, IBTError)

    def test_stores_service_and_message(self):
        exc = UpstreamServiceError("kendra", "boom")
        assert exc.service == "kendra"
        assert exc.message == "boom"
        assert str(exc) == "boom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ibt-agent && python -m pytest tests/unit/test_exceptions.py -v`
Expected: FAIL with `ImportError: cannot import name 'UpstreamServiceError'`

- [ ] **Step 3: Implement `UpstreamServiceError`**

Edit `ibt-agent/src/exceptions.py` — current full content is:

```python
"""Essential exceptions for the IBT Agent."""

class IBTError(Exception):
    """Base exception for IBT Agent errors."""
    
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
```

Replace it with:

```python
"""Essential exceptions for the IBT Agent."""

class IBTError(Exception):
    """Base exception for IBT Agent errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class UpstreamServiceError(IBTError):
    """Raised when an AWS dependency (Kendra, STS) call fails.

    Mapped to HTTP 500 by the app-level exception handler so the
    orchestrator's raise_for_status() sees the failure.
    """

    def __init__(self, service: str, message: str) -> None:
        self.service = service
        super().__init__(message)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ibt-agent && python -m pytest tests/unit/test_exceptions.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ibt-agent/src/exceptions.py ibt-agent/tests/unit/test_exceptions.py
git commit -m "feat(ibt-agent): add UpstreamServiceError for AWS dependency failures"
```

---

### Task 2: Raise `UpstreamServiceError` from `KendraService`, remove `QueryLimitExceededError`

**Files:**
- Modify: `ibt-agent/src/services/kendra_service.py`
- Modify: `ibt-agent/tests/unit/test_kendra_service.py`

**Interfaces:**
- Consumes: `UpstreamServiceError` from Task 1 (`src.exceptions.UpstreamServiceError`).
- Produces: `KendraService._assume_kendra_role` and `KendraService.get_ncct_ids_by_product` now raise `UpstreamServiceError("kendra", <message>)` instead of `RuntimeError`/`QueryLimitExceededError` on any AWS failure (including throttling). `QueryLimitExceededError` class is removed from this module entirely — no longer exported.

- [ ] **Step 1: Write the failing tests**

In `ibt-agent/tests/unit/test_kendra_service.py`, replace the three tests that currently assert on `RuntimeError`/`QueryLimitExceededError` with versions asserting `UpstreamServiceError`.

Replace this test (currently lines 190-200):

```python
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_exception_handling(self, mock_boto):
        """Test get_ncct_ids_by_product raises exception on error."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception('Kendra error')
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        
        with pytest.raises(RuntimeError, match="Kendra search failed for product 1: Kendra error"):
            service.get_ncct_ids_by_product('test query', '1')
```

with:

```python
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_exception_handling(self, mock_boto):
        """Test get_ncct_ids_by_product raises UpstreamServiceError on error."""
        from src.exceptions import UpstreamServiceError

        mock_client = MagicMock()
        mock_client.query.side_effect = Exception('Kendra error')
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')

        with pytest.raises(UpstreamServiceError, match="Kendra search failed for product 1: Kendra error"):
            service.get_ncct_ids_by_product('test query', '1')
```

Replace this test (currently lines 202-218):

```python
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_throttling_raises_query_limit_exceeded(self, mock_boto):
        """Test ThrottlingException raises QueryLimitExceededError."""
        from botocore.exceptions import ClientError
        from src.services.kendra_service import QueryLimitExceededError
        
        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'Query'
        )
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        
        with pytest.raises(QueryLimitExceededError, match="Kendra query limit exceeded"):
            service.get_ncct_ids_by_product('dental benefits', '1')
```

with:

```python
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_throttling_raises_upstream_service_error(self, mock_boto):
        """Test ThrottlingException raises UpstreamServiceError, same as any other Kendra failure."""
        from botocore.exceptions import ClientError
        from src.exceptions import UpstreamServiceError

        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'Query'
        )
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')

        with pytest.raises(UpstreamServiceError, match="Kendra search failed for product 1: ThrottlingException"):
            service.get_ncct_ids_by_product('dental benefits', '1')
```

Replace this test (currently lines 220-235):

```python
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_client_error_non_throttling(self, mock_boto):
        """Test non-throttling ClientError raises RuntimeError."""
        from botocore.exceptions import ClientError
        
        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'Invalid query'}},
            'Query'
        )
        mock_boto.return_value = mock_client
        
        service = KendraService('test-index', 'us-east-1')
        
        with pytest.raises(RuntimeError, match="Kendra search failed for product 1: ValidationException"):
            service.get_ncct_ids_by_product('dental benefits', '1')
```

with:

```python
    @patch('boto3.client')
    def test_get_ncct_ids_by_product_client_error_non_throttling(self, mock_boto):
        """Test non-throttling ClientError raises UpstreamServiceError."""
        from botocore.exceptions import ClientError
        from src.exceptions import UpstreamServiceError

        mock_client = MagicMock()
        mock_client.query.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'Invalid query'}},
            'Query'
        )
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')

        with pytest.raises(UpstreamServiceError, match="Kendra search failed for product 1: ValidationException"):
            service.get_ncct_ids_by_product('dental benefits', '1')
```

Also add a new test for role-assumption failure, appended at the end of `TestKendraService` (after `test_get_ncct_ids_by_product_function`):

```python
    @patch('src.services.kendra_service.boto3.client')
    def test_assume_kendra_role_client_error_raises_upstream_service_error(self, mock_boto):
        """Test role assumption ClientError raises UpstreamServiceError."""
        from botocore.exceptions import ClientError
        from src.exceptions import UpstreamServiceError

        mock_sts_client = MagicMock()
        mock_sts_client.assume_role.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Not authorized'}},
            'AssumeRole'
        )
        mock_boto.return_value = mock_sts_client

        service = KendraService('test-index', 'us-east-1')
        service.settings.kendra_role_arn = 'arn:aws:iam::123456789012:role/test-role'

        with pytest.raises(UpstreamServiceError, match="Role assumption failed: AccessDenied"):
            service._assume_kendra_role()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ibt-agent && python -m pytest tests/unit/test_kendra_service.py -v`
Expected: FAIL — `test_get_ncct_ids_by_product_exception_handling`, `test_get_ncct_ids_by_product_throttling_raises_upstream_service_error`, `test_get_ncct_ids_by_product_client_error_non_throttling`, and `test_assume_kendra_role_client_error_raises_upstream_service_error` fail (still raising `RuntimeError`/`QueryLimitExceededError`, or `ImportError` for the removed `QueryLimitExceededError` import in the old test bodies you just replaced).

- [ ] **Step 3: Implement the change in `kendra_service.py`**

Edit `ibt-agent/src/services/kendra_service.py`.

Add the import near the top (after the `src.utils.logging` import):

```python
from src.exceptions import UpstreamServiceError
```

Remove the `QueryLimitExceededError` class entirely:

```python
class QueryLimitExceededError(Exception):
    """Raised when Kendra query limit is exceeded."""
    pass
```

In `_assume_kendra_role`, replace:

```python
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to assume Kendra role {self.settings.kendra_role_arn}: {error_code} - {error_msg}")
            raise RuntimeError(f"Role assumption failed: {error_code} - {error_msg}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RuntimeError(f"Role assumption failed: {str(e)}") from e
```

with:

```python
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to assume Kendra role {self.settings.kendra_role_arn}: {error_code} - {error_msg}")
            raise UpstreamServiceError("kendra", f"Role assumption failed: {error_code} - {error_msg}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise UpstreamServiceError("kendra", f"Role assumption failed: {str(e)}") from e
```

In `get_ncct_ids_by_product`, replace:

```python
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning(f"Query limit exceeded for product {product_id}")
                raise QueryLimitExceededError("Kendra query limit exceeded") from e
            logger.error(f"Kendra API error for product {product_id}: {error_code} - {str(e)}")
            raise RuntimeError(f"Kendra search failed for product {product_id}: {error_code}") from e
        except QueryLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Error extracting NCCT IDs for product {product_id}: {str(e)}")
            raise RuntimeError(f"Kendra search failed for product {product_id}: {str(e)}") from e
```

with:

```python
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning(f"Query limit exceeded for product {product_id}")
            else:
                logger.error(f"Kendra API error for product {product_id}: {error_code} - {str(e)}")
            raise UpstreamServiceError("kendra", f"Kendra search failed for product {product_id}: {error_code}") from e
        except Exception as e:
            logger.error(f"Error extracting NCCT IDs for product {product_id}: {str(e)}")
            raise UpstreamServiceError("kendra", f"Kendra search failed for product {product_id}: {str(e)}") from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ibt-agent && python -m pytest tests/unit/test_kendra_service.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add ibt-agent/src/services/kendra_service.py ibt-agent/tests/unit/test_kendra_service.py
git commit -m "fix(ibt-agent): raise UpstreamServiceError for all Kendra/STS failures including throttling"
```

---

### Task 3: Stop swallowing upstream errors in `HybridIBTAgent.process_query`

**Files:**
- Modify: `ibt-agent/src/agent/hybrid_ibt.py`
- Modify: `ibt-agent/tests/unit/test_hybrid_agent.py`

**Interfaces:**
- Consumes: `UpstreamServiceError` from Task 1.
- Produces: `HybridIBTAgent.process_query` no longer catches `QueryLimitExceededError`, `KendraSearchError`, or `RuntimeError`. Any `UpstreamServiceError` raised by `_process_direct_kendra` (via `get_ncct_ids_by_product`) propagates out of `process_query` uncaught. `KendraSearchError` and `QueryProcessingError` classes are deleted from this module.

- [ ] **Step 1: Write the failing tests**

In `ibt-agent/tests/unit/test_hybrid_agent.py`, replace `test_process_query_limit_exceeded` (currently lines 95-116):

```python
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_limit_exceeded(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        """Test that QueryLimitExceededError returns proper error message."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_kendra_service.return_value = MagicMock()

        from src.services.kendra_service import QueryLimitExceededError
        mock_direct_process.side_effect = QueryLimitExceededError("Kendra query limit exceeded")

        agent = HybridIBTAgent()

        result = agent.process_query("dental benefits", "sess-limit", {"productId": "1"})

        assert result['success'] is False
        assert result['sessionId'] == 'sess-limit'
        assert result['confidence'] == 0.0
        assert isinstance(result['responseText'], list)
        assert "maximum number of search requests" in result['responseText'][0]
```

with:

```python
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_propagates_upstream_service_error(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        """Test that UpstreamServiceError (e.g. throttling) propagates uncaught for FastAPI to map to a 500."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_kendra_service.return_value = MagicMock()

        from src.exceptions import UpstreamServiceError
        mock_direct_process.side_effect = UpstreamServiceError("kendra", "Kendra search failed for product 1: ThrottlingException")

        agent = HybridIBTAgent()

        with pytest.raises(UpstreamServiceError):
            agent.process_query("dental benefits", "sess-limit", {"productId": "1"})
```

Replace `test_process_query_exception_handling` (currently lines 118-138):

```python
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_exception_handling(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        # Mock settings
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        mock_get_kendra_service.return_value = MagicMock()

        from src.agent.hybrid_ibt import KendraSearchError
        mock_direct_process.side_effect = KendraSearchError("Test error")

        agent = HybridIBTAgent()

        result = agent.process_query("test", "sess-001", {"productId": "1"})

        assert result['success'] is False
        assert "technical difficulties" in result['responseText'][0]
        assert result['sessionId'] == 'sess-001'
```

with:

```python
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    def test_process_query_unexpected_exception_propagates(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        """Test that non-UpstreamServiceError exceptions also propagate (no catch-all here anymore)."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        mock_get_kendra_service.return_value = MagicMock()

        mock_direct_process.side_effect = ValueError("Test error")

        agent = HybridIBTAgent()

        with pytest.raises(ValueError, match="Test error"):
            agent.process_query("test", "sess-001", {"productId": "1"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ibt-agent && python -m pytest tests/unit/test_hybrid_agent.py -v`
Expected: FAIL — both new tests fail because `process_query` still catches `RuntimeError`/`KendraSearchError`/`QueryLimitExceededError` and returns a 200-style dict instead of raising.

- [ ] **Step 3: Implement the change in `hybrid_ibt.py`**

Edit `ibt-agent/src/agent/hybrid_ibt.py`. Current full content is:

```python
import time
from typing import Dict, Any
from src.config.messages import get_message
from src.config.settings import get_settings
from src.services.kendra_service import get_ncct_ids_by_product, QueryLimitExceededError
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Custom exceptions
class KendraSearchError(Exception):
    """Raised when Kendra search operations fail."""
    pass

class QueryProcessingError(Exception):
    """Raised when query processing fails."""
    pass

class HybridIBTAgent:
    def __init__(self):
        self.settings = get_settings()
        self.kendra_index_id = self.settings.kendra_index_id
        self.aws_region = self.settings.aws_region

    def process_query(self, user_prompt: str, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        logger.info(f"Processing query with context: userName={context.get('userName')}, userType={context.get('userType')}, productId={context.get('productId')}, source={context.get('source')}")

        try:
            result = self._process_direct_kendra(user_prompt, context)

            execution_time = (time.time() - start_time) * 1000

            response = {
                "sessionId": session_id,
                "confidence": result.get("confidence", 0.0),
                "responseText": result.get("response_text", ""),
                "success": result.get("success", False),
                "execution_time_ms": round(execution_time, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            logger.info("IBT Agent response: session_id=%s, success=%s, confidence=%s, responseText=%s",
                        response["sessionId"], response["success"], response["confidence"], response["responseText"])
            return response

        except QueryLimitExceededError:
            logger.warning(f"Query limit exceeded for session {session_id}")
            response = {
                "sessionId": session_id,
                "confidence": 0.0,
                "responseText": get_message("query_limit_exceeded"),
                "success": False,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            logger.info("IBT Agent query limit response: session_id=%s, responseText=%s",
                        response["sessionId"], response["responseText"])
            return response
        except (KendraSearchError, RuntimeError) as e:
            logger.error(f"Query processing error: {str(e)}")
            response = {
                "sessionId": session_id,
                "confidence": 0.0,
                "responseText": get_message("service_unavailable"),
                "success": False,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            logger.info("IBT Agent error response: session_id=%s, success=%s, responseText=%s",
                        response["sessionId"], response["success"], response["responseText"])
            return response

    def _process_direct_kendra(self, user_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        product_id = context['productId']

        logger.info(f"Processing direct Kendra query with product ID: {product_id}")

        # Always use product-filtered NCCT ID extraction
        ncct_ids = get_ncct_ids_by_product(user_prompt, str(product_id))
        logger.info(f"Product {product_id} filtering applied, found {len(ncct_ids)} results")

        # Remove duplicates while preserving order
        seen = set()
        unique_ncct_ids = []
        for ncct_id in ncct_ids:
            if ncct_id not in seen:
                seen.add(ncct_id)
                unique_ncct_ids.append(ncct_id)

        if not unique_ncct_ids:
            logger.info("No NCCT IDs found for the query and product combination")
            return {
                "success": True,
                "response_text": get_message("no_results_found"),
                "confidence": 0.0,
                "product_id": product_id,
                "ncct_count": 0
            }

        logger.info(f"Found {len(unique_ncct_ids)} unique NCCT IDs: {unique_ncct_ids}")

        return {
            "success": True,
            "response_text": unique_ncct_ids,
            "confidence": 8.0,  # High confidence for direct Kendra results
            "product_id": product_id,
            "ncct_count": len(unique_ncct_ids)
        }
```

Replace it with:

```python
import time
from typing import Dict, Any
from src.config.messages import get_message
from src.config.settings import get_settings
from src.services.kendra_service import get_ncct_ids_by_product
from src.utils.logging import get_logger

logger = get_logger(__name__)

class HybridIBTAgent:
    def __init__(self):
        self.settings = get_settings()
        self.kendra_index_id = self.settings.kendra_index_id
        self.aws_region = self.settings.aws_region

    def process_query(self, user_prompt: str, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        logger.info(f"Processing query with context: userName={context.get('userName')}, userType={context.get('userType')}, productId={context.get('productId')}, source={context.get('source')}")

        result = self._process_direct_kendra(user_prompt, context)

        execution_time = (time.time() - start_time) * 1000

        response = {
            "sessionId": session_id,
            "confidence": result.get("confidence", 0.0),
            "responseText": result.get("response_text", ""),
            "success": result.get("success", False),
            "execution_time_ms": round(execution_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        logger.info("IBT Agent response: session_id=%s, success=%s, confidence=%s, responseText=%s",
                    response["sessionId"], response["success"], response["confidence"], response["responseText"])
        return response

    def _process_direct_kendra(self, user_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        product_id = context['productId']

        logger.info(f"Processing direct Kendra query with product ID: {product_id}")

        # Always use product-filtered NCCT ID extraction
        ncct_ids = get_ncct_ids_by_product(user_prompt, str(product_id))
        logger.info(f"Product {product_id} filtering applied, found {len(ncct_ids)} results")

        # Remove duplicates while preserving order
        seen = set()
        unique_ncct_ids = []
        for ncct_id in ncct_ids:
            if ncct_id not in seen:
                seen.add(ncct_id)
                unique_ncct_ids.append(ncct_id)

        if not unique_ncct_ids:
            logger.info("No NCCT IDs found for the query and product combination")
            return {
                "success": True,
                "response_text": get_message("no_results_found"),
                "confidence": 0.0,
                "product_id": product_id,
                "ncct_count": 0
            }

        logger.info(f"Found {len(unique_ncct_ids)} unique NCCT IDs: {unique_ncct_ids}")

        return {
            "success": True,
            "response_text": unique_ncct_ids,
            "confidence": 8.0,  # High confidence for direct Kendra results
            "product_id": product_id,
            "ncct_count": len(unique_ncct_ids)
        }
```

Note: `get_message("service_unavailable")` is still used elsewhere (kept in `messages.py` — see Task 4), it's just no longer referenced from this file. That's expected; it remains available for any future caller and is still covered by `test_messages.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ibt-agent && python -m pytest tests/unit/test_hybrid_agent.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add ibt-agent/src/agent/hybrid_ibt.py ibt-agent/tests/unit/test_hybrid_agent.py
git commit -m "fix(ibt-agent): let UpstreamServiceError propagate out of process_query instead of returning 200"
```

---

### Task 4: Remove `query_limit_exceeded` from `src/config/messages.py`

**Files:**
- Modify: `ibt-agent/src/config/messages.py`
- Modify: `ibt-agent/tests/unit/test_messages.py`

**Interfaces:**
- Produces: `GENERIC_MESSAGES` no longer has a `"query_limit_exceeded"` key. `get_message` behavior for `"no_results_found"` and `"service_unavailable"` is unchanged.

- [ ] **Step 1: Write the failing test changes**

In `ibt-agent/tests/unit/test_messages.py`:

Replace `test_generic_messages_exist` (lines 9-18):

```python
    def test_generic_messages_exist(self):
        """Test that required generic messages are defined."""
        assert "no_results_found" in GENERIC_MESSAGES
        assert "service_unavailable" in GENERIC_MESSAGES
        assert "query_limit_exceeded" in GENERIC_MESSAGES
        
        # Verify no_results_found is the FEPOC fallback message list
        assert isinstance(GENERIC_MESSAGES["no_results_found"], list)
        assert "Your search did not return any results" in GENERIC_MESSAGES["no_results_found"][0]
        assert "technical difficulties" in GENERIC_MESSAGES["service_unavailable"][0]
```

with:

```python
    def test_generic_messages_exist(self):
        """Test that required generic messages are defined."""
        assert "no_results_found" in GENERIC_MESSAGES
        assert "service_unavailable" in GENERIC_MESSAGES
        assert "query_limit_exceeded" not in GENERIC_MESSAGES

        # Verify no_results_found is the FEPOC fallback message list
        assert isinstance(GENERIC_MESSAGES["no_results_found"], list)
        assert "Your search did not return any results" in GENERIC_MESSAGES["no_results_found"][0]
        assert "technical difficulties" in GENERIC_MESSAGES["service_unavailable"][0]
```

Replace `test_get_message_valid_key` (lines 20-28):

```python
    def test_get_message_valid_key(self):
        """Test get_message with valid keys."""
        no_results_msg = get_message("no_results_found")
        service_unavailable_msg = get_message("service_unavailable")
        query_limit_msg = get_message("query_limit_exceeded")
        
        assert no_results_msg == GENERIC_MESSAGES["no_results_found"]
        assert service_unavailable_msg == GENERIC_MESSAGES["service_unavailable"]
        assert query_limit_msg == GENERIC_MESSAGES["query_limit_exceeded"]
```

with:

```python
    def test_get_message_valid_key(self):
        """Test get_message with valid keys."""
        no_results_msg = get_message("no_results_found")
        service_unavailable_msg = get_message("service_unavailable")

        assert no_results_msg == GENERIC_MESSAGES["no_results_found"]
        assert service_unavailable_msg == GENERIC_MESSAGES["service_unavailable"]

    def test_get_message_removed_query_limit_key_returns_default(self):
        """Test get_message falls back to service_unavailable for the removed query_limit_exceeded key."""
        assert get_message("query_limit_exceeded") == GENERIC_MESSAGES["service_unavailable"]
```

Replace `test_message_content_requirements` (lines 47-68):

```python
    def test_message_content_requirements(self):
        """Test that messages meet FEPOC specification requirements."""
        no_results = GENERIC_MESSAGES["no_results_found"]
        service_unavailable = GENERIC_MESSAGES["service_unavailable"]
        query_limit = GENERIC_MESSAGES["query_limit_exceeded"]
        
        # no_results_found is a list with brochure/summary links
        assert isinstance(no_results, list)
        assert len(no_results) == 1
        assert "fepblue.org/plan-brochures" in no_results[0]
        assert "fepblue.org/plan-summaries" in no_results[0]
        # service_unavailable is a list
        assert isinstance(service_unavailable, list)
        assert len(service_unavailable) == 1
        assert "technical difficulties" in service_unavailable[0]
        assert "try again" in service_unavailable[0]
        
        # query_limit_exceeded is a list with rate limit message
        assert isinstance(query_limit, list)
        assert len(query_limit) == 1
        assert "maximum number of search requests" in query_limit[0]
        assert "wait" in query_limit[0]
```

with:

```python
    def test_message_content_requirements(self):
        """Test that messages meet FEPOC specification requirements."""
        no_results = GENERIC_MESSAGES["no_results_found"]
        service_unavailable = GENERIC_MESSAGES["service_unavailable"]

        # no_results_found is a list with brochure/summary links
        assert isinstance(no_results, list)
        assert len(no_results) == 1
        assert "fepblue.org/plan-brochures" in no_results[0]
        assert "fepblue.org/plan-summaries" in no_results[0]
        # service_unavailable is a list
        assert isinstance(service_unavailable, list)
        assert len(service_unavailable) == 1
        assert "technical difficulties" in service_unavailable[0]
        assert "try again" in service_unavailable[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ibt-agent && python -m pytest tests/unit/test_messages.py -v`
Expected: FAIL — `test_generic_messages_exist` fails (`"query_limit_exceeded" not in GENERIC_MESSAGES` is False since the key still exists); `test_get_message_removed_query_limit_key_returns_default` fails with `KeyError`-adjacent assertion mismatch since the key is still present.

- [ ] **Step 3: Implement the change in `messages.py`**

Edit `ibt-agent/src/config/messages.py`. Current full content is:

```python
"""Generic response messages per FEPOC specification."""

# Fallback message when search returns no results
_FALLBACK_MESSAGE = [
    "Your search did not return any results. Please refer to the "
    "<a href='https://www.fepblue.org/plan-brochures' target='_blank'>Blue Cross and Blue Shield Service Benefit Plan brochure</a> "
    "or to the "
    "<a href='https://www.fepblue.org/plan-summaries' target='_blank'>Health Plan Summaries</a>."
]

# Generic response messages as specified in requirements
GENERIC_MESSAGES = {
    "no_results_found": _FALLBACK_MESSAGE,
    "service_unavailable": [
        "I'm currently experiencing technical difficulties accessing benefit information. Please try again in a few moments."
    ],
    "query_limit_exceeded": [
        "You have reached the maximum number of search requests allowed. Please wait a moment before trying again."
    ]
}

def get_message(key: str):
    """Get generic message by key."""
    return GENERIC_MESSAGES.get(key, GENERIC_MESSAGES["service_unavailable"])
```

Replace it with:

```python
"""Generic response messages per FEPOC specification."""

# Fallback message when search returns no results
_FALLBACK_MESSAGE = [
    "Your search did not return any results. Please refer to the "
    "<a href='https://www.fepblue.org/plan-brochures' target='_blank'>Blue Cross and Blue Shield Service Benefit Plan brochure</a> "
    "or to the "
    "<a href='https://www.fepblue.org/plan-summaries' target='_blank'>Health Plan Summaries</a>."
]

# Generic response messages as specified in requirements
GENERIC_MESSAGES = {
    "no_results_found": _FALLBACK_MESSAGE,
    "service_unavailable": [
        "I'm currently experiencing technical difficulties accessing benefit information. Please try again in a few moments."
    ],
}

def get_message(key: str):
    """Get generic message by key."""
    return GENERIC_MESSAGES.get(key, GENERIC_MESSAGES["service_unavailable"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ibt-agent && python -m pytest tests/unit/test_messages.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add ibt-agent/src/config/messages.py ibt-agent/tests/unit/test_messages.py
git commit -m "fix(ibt-agent): remove unused query_limit_exceeded message"
```

---

### Task 5: Map `UpstreamServiceError` and unhandled exceptions to HTTP 500 in `app.py`

**Files:**
- Modify: `ibt-agent/src/api/app.py`
- Modify: `ibt-agent/tests/unit/test_app.py`

**Interfaces:**
- Consumes: `UpstreamServiceError` from Task 1, `get_logger` from `src.utils.logging` (already used elsewhere in the codebase, e.g. `kendra_service.py`).
- Produces: `create_app()` returns a `FastAPI` app with two registered exception handlers: `UpstreamServiceError` → `JSONResponse(500, {"detail": f"Upstream service error ({service}): {message}"})`; bare `Exception` → `JSONResponse(500, {"detail": "Internal server error"})`.

- [ ] **Step 1: Write the failing tests**

Append to `ibt-agent/tests/unit/test_app.py` (add these imports at the top alongside the existing ones, and add the new test class at the end of the file):

Add to the top imports:

```python
from src.exceptions import UpstreamServiceError
```

Add at the end of the file:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ibt-agent && python -m pytest tests/unit/test_app.py -v`
Expected: FAIL — both new tests fail (no exception handlers registered yet, so `UpstreamServiceError`/`ValueError` propagate as unhandled errors through `TestClient`).

- [ ] **Step 3: Implement the exception handlers in `app.py`**

Edit `ibt-agent/src/api/app.py`. Current full content is:

```python
"""FastAPI application factory."""

from fastapi import FastAPI

from src.api.routes import health, invocations
from src.config.constants import SERVICE_NAME, SERVICE_DESCRIPTION, SERVICE_VERSION, API_PREFIX


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        version=SERVICE_VERSION,
    )

    # Include routers
    app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
    app.include_router(invocations.router, prefix=API_PREFIX, tags=["Invocations"])

    return app
```

Replace it with:

```python
"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import health, invocations
from src.config.constants import SERVICE_NAME, SERVICE_DESCRIPTION, SERVICE_VERSION, API_PREFIX
from src.exceptions import UpstreamServiceError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        version=SERVICE_VERSION,
    )

    # Include routers
    app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
    app.include_router(invocations.router, prefix=API_PREFIX, tags=["Invocations"])

    @app.exception_handler(UpstreamServiceError)
    async def upstream_service_error_handler(request: Request, exc: UpstreamServiceError) -> JSONResponse:
        """Map AWS dependency failures (Kendra/STS) to HTTP 500 for the orchestrator."""
        logger.error("Upstream %s failure on %s: %s", exc.service, request.url.path, exc.message)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Upstream service error ({exc.service}): {exc.message}"},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all so unclassified bugs return a structured 500 instead of Starlette's bare default.

        The exception message isn't included in the response body since it could
        leak internals; full detail (with traceback) goes to the logs only.
        """
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ibt-agent && python -m pytest tests/unit/test_app.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Run the full test suite**

Run: `cd ibt-agent && python -m pytest -v`
Expected: PASS (all tests across the project — this is the integration checkpoint confirming Tasks 1-5 work together)

- [ ] **Step 6: Commit**

```bash
git add ibt-agent/src/api/app.py ibt-agent/tests/unit/test_app.py
git commit -m "feat(ibt-agent): map UpstreamServiceError and unhandled exceptions to HTTP 500"
```

---

## Post-plan verification

- [ ] Run `cd ibt-agent && python -m pytest -v` one final time and confirm 0 failures.
- [ ] Grep the `ibt-agent` tree for `QueryLimitExceededError`, `KendraSearchError`, and `QueryProcessingError` to confirm no remaining references outside of git history (`grep -rn "QueryLimitExceededError\|KendraSearchError\|QueryProcessingError" ibt-agent/src ibt-agent/tests`).
