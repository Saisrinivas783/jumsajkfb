# Async Concurrency Design — ibt-agent & orchestrator-agent

## Goal

Both services currently handle requests through an accidental concurrency ceiling: sync FastAPI route handlers run in Starlette's shared, process-wide default thread pool (capped at 40, not owned or sized by either app), and every `boto3`/`httpx` client involved (Kendra, Bedrock, STS, the HTTP call between the two services) uses default connection-pool sizes (10) that were never chosen for this workload. This design gives each service its own, explicitly sized concurrency mechanism — async request handling, a dedicated thread pool, and configurable connection-pool sizes — so concurrency limits are something the team controls and tunes, not an implementation detail of a framework default.

## Non-goals

- No change to public request/response schemas or endpoint behavior. This is purely an internal execution-model change.
- No adoption of `aioboto3` or any other AWS SDK replacement — raw boto3 calls are offloaded to a thread pool via `asyncio.to_thread`, not rewritten to a different async AWS client library.
- No backpressure/load-shedding mechanism (e.g., bounded queue, request rejection under saturation) — flagged as a known limitation and possible future enhancement, not solved here.
- No fix for the previously-identified, out-of-scope `PRODUCT_MAPPING` unfiltered-query fallback bug in ibt-agent, or the pre-existing `test_assume_kendra_role_success` failure — both remain untouched, unrelated follow-ups.

## Architecture overview

**Startup wiring (both services), added to each app's `_lifespan`:**
1. Build `ThreadPoolExecutor(max_workers=settings.thread_pool_max_workers)`.
2. `asyncio.get_running_loop().set_default_executor(executor)` — every bare `asyncio.to_thread(...)` call anywhere in that process routes through this pool automatically, without threading an executor object through every function signature.
3. orchestrator-agent only: build a shared `httpx.AsyncClient(limits=httpx.Limits(...), timeout=settings.tool_timeout)`.
4. On shutdown: `executor.shutdown(wait=True)`; orchestrator-agent also closes the `httpx.AsyncClient`.

**Call-chain shape, ibt-agent:**
`async def invocations route` → `await agent.process_query(...)` (now async) → `await get_ncct_ids_by_product(...)` (now async) → `await asyncio.to_thread(kendra_client.query, ...)` (the actual blocking AWS call, now off the event loop).

**Call-chain shape, orchestrator-agent:**
`async def invocations route` → `await agent.handle_invocation(...)` (now async) → `await graph_app.ainvoke(...)` → LangGraph awaits each node:
- `guardrail_check_node` (async) → `await asyncio.to_thread(bedrock_client.apply_guardrail, ...)`
- `intent_node` (async) → `await structured_llm.ainvoke(...)` (LangChain's built-in async wrapping — benefits from our custom default executor automatically, since that wrapping itself uses `run_in_executor(None, ...)`)
- `tool_node` (async, in `tool_node_factory.py`) → `await async_http_client.post(...)` (true async I/O, no thread needed) to call ibt-agent

**boto3 async mechanism (decision):** raw boto3 calls with no native async support (Kendra query, Kendra/Bedrock STS `assume_role`, Bedrock `apply_guardrail`) are wrapped at their call sites in `await asyncio.to_thread(...)`, riding the custom default executor set at startup. `aioboto3` was considered and rejected — it would add a new dependency, a different (async-context-manager) client lifecycle, and a much larger diff across `kendra_service.py` and `llm/client.py`, for marginal benefit on these short, low-payload calls. `ChatBedrockConverse.ainvoke()` needs no manual wrapping — LangChain's base chat model class provides async support by wrapping the sync call in an executor internally, and that wrapping also rides our custom default executor.

## Performance implications (CPU)

The mechanism itself does not meaningfully increase CPU load — these calls are I/O-bound, and Python's GIL is released during blocking I/O syscalls, so threads parked in `asyncio.to_thread(...)` waiting on AWS/HTTP responses cost memory (a thread stack) and OS scheduler bookkeeping, not CPU cycles. Two real but modest CPU costs exist: increased context-switching overhead at higher thread counts, and GIL hand-off overhead on the CPU-bound slivers of each request (Pydantic validation, JSON serialization, logging) as more requests interleave — both small relative to the tens-to-hundreds-of-ms network round-trips involved. The dominant, *expected* source of increased aggregate CPU usage is simply that if this design achieves its goal, the service handles more requests per second than before — proportionally more total work done, not architectural waste. One part of this change should *lower* per-request CPU: replacing orchestrator's current per-call `httpx.Client()` (fresh TCP+TLS handshake every request) with a shared, pooled `httpx.AsyncClient` avoids repeating that handshake via keep-alive reuse. `thread_pool_max_workers=20` is a normal, correct size for an I/O-bound pool relative to CPU core count; the constraint on that number is memory/file-descriptor budget across `pods × workers × pool size`, not CPU.

## Concurrency correctness

**The problem (present in the code today, latent):** `KendraService._get_kendra_client()` and `ChatModels._get_bedrock_client()` both do an unguarded check-then-act sequence — check if credentials are expired, and if so call `assume_role` and reassign the client. Under concurrency, multiple callers can simultaneously observe expired credentials and each independently call STS, which is wasteful and, at scale, a throttling risk. Python's GIL means this never corrupts the client object (attribute assignment is atomic), and no in-flight request uses a "torn" client (each request fetches the client once per call and uses that reference), but the duplicate-refresh inefficiency is real and gets worse as concurrency increases — which this design's whole purpose is to increase.

**The fix:** both classes gain a `threading.Lock` (not `asyncio.Lock`) guarding the check-refresh-assign sequence, using double-checked locking so the common case (valid cached client) doesn't pay lock overhead — only an actual refresh acquires the lock and re-checks the expiry condition before calling `assume_role`. `threading.Lock` is correct here, not `asyncio.Lock`, because the entire critical section (the STS call and client construction) runs inside a worker thread via `asyncio.to_thread`, never directly on the event loop and never held across an `await` — so there's no risk of blocking the loop while holding it, and no reentrancy hazard.

## Component changes — ibt-agent

- **`src/config/settings.py`** — add to `IBTSettings`: `kendra_max_pool_connections` (default 20), `sts_max_pool_connections` (default 10), `thread_pool_max_workers` (default 20).
- **`src/executor.py`** (new) — `get_executor()` / `shutdown_executor()`, singleton `ThreadPoolExecutor(max_workers=settings.thread_pool_max_workers)`.
- **`src/services/kendra_service.py`**:
  - `_get_boto_config()` gains `max_pool_connections=settings.kendra_max_pool_connections`.
  - `_assume_kendra_role()`'s STS client gains its own `Config(max_pool_connections=settings.sts_max_pool_connections)` (currently has none).
  - `__init__` gains `self._client_lock = threading.Lock()`.
  - `_get_kendra_client()` becomes `async def` with the double-checked-locking + `asyncio.to_thread(self._refresh_client_locked)` pattern described above.
  - `get_ncct_ids_by_product()` (instance and module-level) becomes `async def`; the `self.client.query(...)` call moves into `await asyncio.to_thread(...)`.
- **`src/agent/hybrid_ibt.py`** — `process_query()` and `_process_direct_kendra()` become `async def`.
- **`src/api/routes/invocations.py`** — `invocations()` becomes `async def`, `await agent.process_query(...)`.
- **`src/api/routes/health.py`** — left as sync `def`; no blocking I/O, converting adds churn for no benefit.
- **`src/api/app.py`** — add an `@asynccontextmanager _lifespan` (doesn't exist yet in this file) implementing the startup/shutdown wiring above.

## Component changes — orchestrator-agent

- **`src/config/settings.py`** — add to `OrchestratorSettings`: `bedrock_max_pool_connections` (default 20), `sts_max_pool_connections` (default 10), `tool_http_max_connections` (default 20), `tool_http_max_keepalive_connections` (default 10), `thread_pool_max_workers` (default 20).
- **`src/executor.py`** (new) — same pattern as ibt-agent's.
- **`src/http_client.py`** (new) — `get_http_client()` / `close_http_client()`, singleton `httpx.AsyncClient(limits=httpx.Limits(max_connections=settings.tool_http_max_connections, max_keepalive_connections=settings.tool_http_max_keepalive_connections), timeout=settings.tool_timeout)`.
- **`src/llm/client.py`** (`ChatModels`):
  - `_get_boto_config()` gains `max_pool_connections=settings.bedrock_max_pool_connections`.
  - `_assume_bedrock_role()`'s STS client gains `Config(max_pool_connections=settings.sts_max_pool_connections)` (currently none).
  - `__init__` gains `self._client_lock = threading.Lock()`.
  - `_get_bedrock_client()` becomes `async def`, same locking pattern as `KendraService`.
  - `bedrock_model()`, `bedrock_model_with_extended_thinking()`, `bedrock_model_with_guardrails()`, `apply_guardrail()`, `get_model()` all become `async def` (each calls `await self._get_bedrock_client()`). `apply_guardrail()`'s actual `client.apply_guardrail(...)` call is wrapped in `await asyncio.to_thread(...)`; `ChatBedrockConverse(...)` construction itself stays sync (no I/O).
- **`src/graph/nodes/guardrail_node.py`** — `guardrail_check_node` becomes `async def`.
- **`src/graph/nodes/intent_analyzer.py`** — `intent_node` and `_get_structured_llm()` become `async def`; `.invoke(messages)` → `await .ainvoke(messages)`.
- **`src/graph/nodes/tool_node_factory.py`** — `_call_tool_api` and the inner `tool_node` function become `async def`; use `get_http_client()` (shared, pooled) instead of `with httpx.Client(...) as client:` per call.
- **`src/graph/nodes/confidence_router.py`, `fallback.py`** — no I/O, left as sync `def`. LangGraph's `ainvoke` supports a mix of sync and async node/router callables in the same graph.
- **`src/graph/orchestrator.py`** — `handle_invocation()` becomes `async def`, `await self.graph_app.ainvoke(state.model_dump())`.
- **`src/api/routes/invocations.py`** — `invocations()` becomes `async def`, `await agent.handle_invocation(...)`.
- **`src/api/app.py`** — extend the existing `_lifespan` (already present here) to also build the executor, `set_default_executor(...)`, and construct/close the shared `httpx.AsyncClient`.

## Testing strategy

- Add `pytest-asyncio` as a test dependency in both services.
- New unit tests: `test_executor.py` (both services), `test_http_client.py` (orchestrator).
- Every existing test that calls a now-async function/method directly needs `await` + `@pytest.mark.asyncio`: `_get_kendra_client`, `_get_bedrock_client`, `get_ncct_ids_by_product`, `process_query`, `intent_node`, `guardrail_check_node`, `tool_node`, `handle_invocation`.
- `TestClient`-based route tests need no rewrite — `TestClient` works transparently against `async def` routes.
- New concurrency tests proving the lock works: spawn several concurrent calls into `_get_kendra_client()` / `_get_bedrock_client()` with credentials forced-expired, assert `assume_role` was called exactly once.
- New cross-request isolation test per service: fire several concurrent requests through the app (`httpx.AsyncClient(app=app)` + `asyncio.gather`, or `TestClient` calls dispatched across threads) with distinct `sessionId`/payloads, assert each response matches its own request.
- Full regression: `pytest tests/ -v` in both subprojects. ibt-agent's pre-existing, out-of-scope `test_assume_kendra_role_success` failure remains and is expected — its assertion logic is unchanged, just reached through the async-wrapped path now.

## Known limitations (accepted, not solved here)

- No explicit backpressure: under sustained overload, `asyncio.to_thread` calls queue on the executor rather than failing fast or shedding load. Consistent with today's behavior (no backpressure exists now either).
- Pool-size defaults (20/20/10/20/10) are per-process; total fleet-wide concurrency against AWS (Kendra/Bedrock/STS rate limits) scales with `pods × workers × pool size` and should be sized with that in mind — not solved by this design, a deployment/capacity-planning concern.
