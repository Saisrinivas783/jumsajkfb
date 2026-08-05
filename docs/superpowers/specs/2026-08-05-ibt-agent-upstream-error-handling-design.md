# IBT Agent: Upstream Service Error Handling

## Context

A sibling copy of `ibt-agent` (at `C:\Users\SRINIVAS\Desktop\abcde\agents\ibt-agent`) has a
distinct AWS-dependency error-handling model: a dedicated `UpstreamServiceError` exception
that is allowed to propagate out of the agent and is converted to an HTTP 500 by a
FastAPI-level exception handler. This repo's `ibt-agent`, by contrast, catches all Kendra/AWS
failures inside `HybridIBTAgent.process_query` and converts them into an HTTP 200 response
with `success: false`, and additionally special-cases Kendra throttling into its own
degraded-but-still-200 response.

This spec adopts the "let it fail loudly" model from the sibling repo, but grafts it onto this
repo's current (more advanced) `kendra_service.py`/`hybrid_ibt.py` rather than copying files
wholesale — the two implementations have diverged in unrelated ways (thread-safe client
refresh, timing instrumentation, retry/pool-size settings) that must be preserved.

## Goal

Any AWS/Kendra dependency failure (role assumption failure, Kendra API error, throttling)
should surface to the caller as an HTTP 500, not a 200 with a friendly degraded message. This
lets the orchestrator's `raise_for_status()` detect the failure instead of silently treating a
backend outage as a valid (if empty-ish) answer. Only "no results found" remains a 200.

## Non-goals

- No change to the `_process_direct_kendra` success path or product-filtering logic.
- No change to `KendraService`'s client-refresh locking, retry config, or timing logs.
- No wholesale copy of the sibling repo's files.

## Design

### 1. `src/exceptions.py`

Add:

```python
class UpstreamServiceError(IBTError):
    """Raised when an AWS dependency (Kendra, STS) call fails.

    Mapped to HTTP 500 by the app-level exception handler so the
    orchestrator's raise_for_status() sees the failure.
    """

    def __init__(self, service: str, message: str) -> None:
        self.service = service
        super().__init__(message)
```

### 2. `src/services/kendra_service.py`

- Remove the `QueryLimitExceededError` class.
- In `_assume_kendra_role`: replace both `raise RuntimeError(...)` calls (the `ClientError`
  branch and the generic `Exception` branch) with `raise UpstreamServiceError("kendra", ...)`,
  preserving the existing log messages.
- In `get_ncct_ids_by_product`:
  - The `ThrottlingException` branch (`error_code == 'ThrottlingException'`) no longer raises
    `QueryLimitExceededError`. It falls through to the same handling as any other Kendra
    `ClientError` — `raise UpstreamServiceError("kendra", f"Kendra search failed for product
    {product_id}: {error_code}")`.
  - The generic `except Exception` branch raises `UpstreamServiceError("kendra", ...)` instead
    of `RuntimeError`.
  - The now-empty `except QueryLimitExceededError: raise` branch is removed.

### 3. `src/agent/hybrid_ibt.py`

- Remove the `KendraSearchError` and `QueryProcessingError` classes (dead code — defined but
  never raised anywhere in the codebase).
- Remove the `from src.services.kendra_service import ... QueryLimitExceededError` import.
- Remove the `except QueryLimitExceededError` block and the `except (KendraSearchError,
  RuntimeError)` block from `process_query`. `UpstreamServiceError` is not caught here — it
  propagates out of `process_query` uncaught, through `invocations.py`, to FastAPI.

### 4. `src/config/messages.py`

Remove the `"query_limit_exceeded"` entry from `GENERIC_MESSAGES` (no longer referenced).

### 5. `src/api/app.py`

Add two exception handlers, using this repo's existing `get_logger` and the constants already
imported from `src.config.constants`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.exceptions import UpstreamServiceError
from src.utils.logging import get_logger

logger = get_logger(__name__)

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
```

## Error handling summary

| Failure | Before | After |
|---|---|---|
| No Kendra results | 200, `success: false`, fallback message | unchanged |
| Kendra throttling | 200, `success: false`, "query limit exceeded" message | **500**, `UpstreamServiceError` |
| Kendra role-assumption failure | 200, `success: false`, "service unavailable" message | **500**, `UpstreamServiceError` |
| Other Kendra `ClientError` | 200, `success: false`, "service unavailable" message | **500**, `UpstreamServiceError` |
| Unclassified exception | uncaught (bare 500 from Starlette) | **500**, structured `{"detail": "Internal server error"}`, full traceback logged |

## Testing

- `tests/unit/test_kendra_service.py`: `test_get_ncct_ids_by_product_throttling_raises_query_limit_exceeded`
  becomes a test that throttling raises `UpstreamServiceError`. Any other test asserting
  `RuntimeError` for role-assumption/Kendra API failures is updated to assert
  `UpstreamServiceError`.
- `tests/unit/test_hybrid_agent.py`: `test_process_query_limit_exceeded` is replaced with a
  test that `process_query` lets `UpstreamServiceError` propagate (raises) rather than
  returning a 200 response. Any test relying on the removed `(KendraSearchError, RuntimeError)`
  catch is updated similarly.
- `tests/unit/test_messages.py`: remove the assertions referencing `"query_limit_exceeded"`.
- New tests for `src/api/app.py`: one verifying `UpstreamServiceError` raised from a route
  handler produces a 500 with the expected `detail` body; one verifying an arbitrary unhandled
  exception produces the generic 500 body.

## Scope check

This is a single, focused change to one error-handling path in one service (`ibt-agent`). No
decomposition needed.
