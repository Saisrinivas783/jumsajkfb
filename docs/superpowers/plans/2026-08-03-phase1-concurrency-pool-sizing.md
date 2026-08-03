# Phase 1: Concurrency Pool Sizing & Credential-Lock Implementation Plan — ibt-agent & orchestrator-agent

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the boto3/httpx connection-pool-size mismatch and the latent credential-refresh race in both services, without converting any route/agent/service method to `async def`. This is a smaller, lower-risk subset of the full async-concurrency design (`docs/superpowers/specs/2026-08-03-async-concurrency-design.md`), done first to validate whether pool sizing alone measurably improves throughput before taking on the full async rewrite (phase 2, `docs/superpowers/plans/2026-08-03-async-concurrency-implementation-plan.md`).

**Architecture:** Today, sync route handlers (and everything they call, including boto3 calls) run on Starlette/anyio's implicit shared thread pool, which defaults to 40 concurrent threads process-wide — not owned or sized by either app. Underneath that, boto3's `Config` objects for Kendra/Bedrock/STS clients use an unset default of 10 pooled HTTP connections, and orchestrator's tool-call HTTP client opens a fresh `httpx.Client()` (new TCP+TLS handshake) on every single call. This phase raises the boto3/httpx pool sizes to match the existing 40-thread ceiling, replaces orchestrator's per-call `httpx.Client()` with one shared, pooled, keep-alive `httpx.Client` singleton, and fixes a latent thread-safety bug — unguarded check-then-act credential refresh in `KendraService`/`ChatModels` that can trigger duplicate STS `assume_role` calls under concurrent requests — using a `threading.Lock` with double-checked locking. No route, agent method, or service method signature changes; everything stays synchronous. Phase 2 (full async conversion + dedicated `ThreadPoolExecutor`) is deferred until this phase's impact is measured.

**Tech Stack:** Python, FastAPI (sync routes, unchanged), `boto3`/`botocore.config.Config`, `httpx.Client`, `threading.Lock`, `pytest`, `concurrent.futures.ThreadPoolExecutor` (test-only, to simulate concurrent callers).

## Global Constraints

- No `async def` conversions anywhere in this phase — routes, `KendraService`, `ChatModels`, `HybridIBTAgent`, and `tool_node_factory` all remain fully synchronous. This is the key difference from the phase-2 plan; do not add `asyncio`, `pytest-asyncio`, or `await` in any file touched here.
- No dedicated `ThreadPoolExecutor` / `set_default_executor` wiring in this phase — the thread-pool ceiling stays Starlette/anyio's implicit default (40). That mechanism is phase-2 scope only.
- Pool-size defaults for this phase (chosen to match the existing implicit 40-thread ceiling, per team decision): `kendra_max_pool_connections=40`, `bedrock_max_pool_connections=40`, `tool_http_max_connections=40`, `tool_http_max_keepalive_connections=20`. `sts_max_pool_connections=20` — STS calls only happen on credential refresh (further reduced by the lock in this same phase), so it does not need to scale to the full 40.
- All new pool-size values are env-configurable `pydantic` `Field`s with the defaults above — never hardcoded constants.
- The credential-refresh lock is `threading.Lock` in both `KendraService` and `ChatModels` — correct here because the guarded critical section (STS call + client construction) already runs on a worker thread (Starlette's implicit sync-route threadpool), never on an event loop, so there is no `asyncio` involved and no reentrancy hazard.
- Every task ends with `pytest tests/ -v` passing inside the relevant subproject (`ibt-agent/` or `orchestrator-agent/`), run from that directory.
- ibt-agent's one pre-existing, out-of-scope test failure — `tests/unit/test_kendra_assume_role.py::TestKendraAssumeRole::test_assume_kendra_role_success` (a `RoleSessionName` assertion mismatch unrelated to concurrency) — must remain untouched and is expected to still fail after every task. Do not "fix" it.
- Do not touch `src/config/product_mapping.py`, `tests/unit/test_product_mapping.py`, or orchestrator-agent's `src/tools/`, `src/schemas/`, `src/exceptions.py`, `src/utils/` — out of scope.
- No new dependencies required — `httpx`, `boto3`, and `threading` are already in both `requirements.txt` files.

---

## Part A: ibt-agent

### Task 1: Pool-size settings

**Files:**
- Modify: `ibt-agent/src/config/settings.py`
- Modify: `ibt-agent/tests/unit/test_settings.py`

**Interfaces:**
- Produces: `IBTSettings.kendra_max_pool_connections: int` (default 40), `IBTSettings.sts_max_pool_connections: int` (default 20) — consumed by Task 2.

- [ ] **Step 1: Write the failing settings tests**

Add to `ibt-agent/tests/unit/test_settings.py`, inside `TestIBTSettings`:

```python
    def test_concurrency_pool_defaults(self):
        """Test concurrency pool-size settings default values."""
        settings = IBTSettings()

        assert settings.kendra_max_pool_connections == 40
        assert settings.sts_max_pool_connections == 20

    @patch.dict('os.environ', {
        'KENDRA_MAX_POOL_CONNECTIONS': '60',
        'STS_MAX_POOL_CONNECTIONS': '30',
    })
    def test_concurrency_pool_environment_variable_override(self):
        """Test that concurrency pool-size settings can be overridden via environment variables."""
        settings = IBTSettings()

        assert settings.kendra_max_pool_connections == 60
        assert settings.sts_max_pool_connections == 30
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ibt-agent && python -m pytest tests/unit/test_settings.py -v -k concurrency_pool`
Expected: FAIL with `AttributeError: 'IBTSettings' object has no attribute 'kendra_max_pool_connections'`.

- [ ] **Step 3: Add the fields to `IBTSettings`**

Edit `ibt-agent/src/config/settings.py` — insert after the `kendra_page_size` field (currently ending at line 65) and before the `# DXAIService Configuration` comment:

```python
    # Concurrency Configuration
    kendra_max_pool_connections: int = Field(
        default=40,
        gt=0,
        description="Max HTTP connection pool size for the Kendra boto3 client"
    )
    sts_max_pool_connections: int = Field(
        default=20,
        gt=0,
        description="Max HTTP connection pool size for the STS boto3 client (Kendra role assumption)"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ibt-agent && python -m pytest tests/unit/test_settings.py -v -k concurrency_pool`
Expected: PASS, 2 passed.

- [ ] **Step 5: Run the full suite to confirm nothing broke**

Run: `cd ibt-agent && python -m pytest tests/ -v`
Expected: Same pass/fail counts as before this task, plus the 2 new tests passing.

- [ ] **Step 6: Commit**

```bash
git add ibt-agent/src/config/settings.py ibt-agent/tests/unit/test_settings.py
git commit -m "feat(ibt-agent): add Kendra/STS connection pool-size settings"
```

---

### Task 2: KendraService — pool sizing and credential-refresh lock

**Files:**
- Modify: `ibt-agent/src/services/kendra_service.py`
- Modify: `ibt-agent/tests/unit/test_kendra_service.py`
- Modify: `ibt-agent/tests/unit/test_kendra_assume_role.py`

**Interfaces:**
- Consumes: `IBTSettings.kendra_max_pool_connections`, `IBTSettings.sts_max_pool_connections` (Task 1).
- Produces: no signature changes — `KendraService._get_kendra_client()`, `KendraService.client` (property), and `KendraService.get_ncct_ids_by_product()` all keep their existing sync signatures. Only internal locking and `Config` objects change.

- [ ] **Step 1: Write the failing pool-sizing tests**

Add to `ibt-agent/tests/unit/test_kendra_service.py`, inside `TestKendraService`:

```python
    @patch('src.services.kendra_service.boto3.client')
    def test_get_boto_config_uses_kendra_max_pool_connections(self, mock_boto):
        """Test _get_boto_config sets max_pool_connections from settings."""
        service = KendraService('test-index', 'us-east-1')
        config = service._get_boto_config()
        assert config.max_pool_connections == service.settings.kendra_max_pool_connections
```

Add to `ibt-agent/tests/unit/test_kendra_assume_role.py`, inside `TestKendraAssumeRole` (after `test_boto_config_creation`):

```python
    @patch('src.services.kendra_service.boto3.client')
    def test_assume_kendra_role_sts_client_uses_sts_max_pool_connections(self, mock_boto_client):
        """Test the STS client used for role assumption is configured with sts_max_pool_connections."""
        mock_sts = Mock()
        mock_boto_client.return_value = mock_sts
        mock_sts.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': '2024-01-01T12:00:00Z'
            }
        }
        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'

        self.kendra_service._assume_kendra_role()

        sts_call = mock_boto_client.call_args
        assert sts_call[0][0] == 'sts'
        assert 'config' in sts_call[1]
        assert sts_call[1]['config'].max_pool_connections == self.kendra_service.settings.sts_max_pool_connections
```

- [ ] **Step 2: Run to verify these two fail**

Run: `cd ibt-agent && python -m pytest tests/unit/test_kendra_service.py::TestKendraService::test_get_boto_config_uses_kendra_max_pool_connections tests/unit/test_kendra_assume_role.py::TestKendraAssumeRole::test_assume_kendra_role_sts_client_uses_sts_max_pool_connections -v`
Expected: Both FAIL — `_get_boto_config` doesn't set `max_pool_connections`, and `_assume_kendra_role`'s `boto3.client('sts', region_name=self.region)` call passes no `config` kwarg at all.

- [ ] **Step 3: Write the failing concurrency (lock) test**

Add to `ibt-agent/tests/unit/test_kendra_assume_role.py`, inside `TestKendraAssumeRole`:

```python
    @patch('src.services.kendra_service.boto3.client')
    def test_concurrent_get_kendra_client_calls_assume_role_once(self, mock_boto_client):
        """Test that concurrent calls to _get_kendra_client with expired credentials
        only trigger one assume_role call, not one per caller."""
        from concurrent.futures import ThreadPoolExecutor
        import time as time_module

        mock_sts = Mock()
        mock_kendra = Mock()

        def boto_client_side_effect(service_name, **kwargs):
            if service_name == 'sts':
                time_module.sleep(0.05)  # simulate network latency, widens the race window
                return mock_sts
            elif service_name == 'kendra':
                return mock_kendra
            return Mock()

        mock_boto_client.side_effect = boto_client_side_effect
        mock_sts.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': '2024-01-01T12:00:00Z'
            }
        }

        self.kendra_service.settings.kendra_role_arn = 'arn:aws:iam::054940911799:role/ibt-ai-index-role'
        self.kendra_service._client = None

        with ThreadPoolExecutor(max_workers=10) as pool:
            clients = list(pool.map(lambda _: self.kendra_service._get_kendra_client(), range(10)))

        assert mock_sts.assume_role.call_count == 1
        assert all(c is mock_kendra for c in clients)
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd ibt-agent && python -m pytest tests/unit/test_kendra_assume_role.py::TestKendraAssumeRole::test_concurrent_get_kendra_client_calls_assume_role_once -v`
Expected: FAIL (or flaky-pass) — `_get_kendra_client` today has no lock at all, so 10 threads racing past the unguarded `if self._client is not None...` check with `self._client = None` will mostly all observe "not set" and each call `assume_role`, making `call_count == 1` fail almost every run (the injected 0.05s STS latency makes the race window wide enough to be reliably observable).

- [ ] **Step 5: Implement the Config, lock, and refactor in `kendra_service.py`**

Edit `ibt-agent/src/services/kendra_service.py`:

Add `threading` to imports (after `import boto3`):

```python
import boto3
import threading
```

Replace `_get_boto_config`:

```python
    def _get_boto_config(self) -> Config:
        """Get boto3 configuration with timeout, retry, and pool settings."""
        return Config(
            read_timeout=300,
            connect_timeout=10,
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=self.settings.kendra_max_pool_connections,
        )

    def _get_sts_boto_config(self) -> Config:
        """Get boto3 configuration for the STS client used in role assumption."""
        return Config(max_pool_connections=self.settings.sts_max_pool_connections)
```

In `_assume_kendra_role`, change the STS client construction line:

```python
            sts_client = boto3.client('sts', region_name=self.region, config=self._get_sts_boto_config())
```

Add `self._client_lock = threading.Lock()` to `__init__`, right after `self._credentials_expiration: Optional[datetime] = None`:

```python
        self._credentials_expiration: Optional[datetime] = None
        self._client_lock = threading.Lock()
```

Replace `_get_kendra_client` entirely, splitting the existing body into a fast unlocked path, a locked double-check, and an extracted `_refresh_client_locked` helper that holds the original refresh logic verbatim:

```python
    def _refresh_client_locked(self) -> boto3.client:
        """Refresh (or create) the Kendra client. Must be called while holding self._client_lock."""
        logger.info(f"Initializing Kendra client: region={self.region}, index_id={self.index_id}")

        if self.settings.kendra_role_arn:
            logger.info("Using role assumption for Kendra access")
            try:
                credentials = self._assume_kendra_role()
                self._assumed_credentials = credentials

                self._client = boto3.client(
                    'kendra',
                    region_name=self.region,
                    config=self._get_boto_config(),
                    **credentials
                )
            except Exception as e:
                logger.error(f"Role assumption failed, falling back to default credentials: {e}")

                self._client = boto3.client(
                    'kendra',
                    region_name=self.region,
                    config=self._get_boto_config()
                )
        else:
            logger.info("Using default AWS credentials for Kendra access")
            self._client = boto3.client(
                'kendra',
                region_name=self.region,
                config=self._get_boto_config()
            )

        return self._client

    def _get_kendra_client(self) -> boto3.client:
        """Get Kendra client with appropriate credentials (thread-safe)."""
        # Fast path: valid cached client, no lock needed.
        if self._client is not None and (not self.settings.kendra_role_arn or not self._credentials_expired()):
            return self._client

        with self._client_lock:
            # Re-check inside the lock: another thread may have just refreshed it.
            if self._client is not None and (not self.settings.kendra_role_arn or not self._credentials_expired()):
                return self._client

            if self._client is not None and self.settings.kendra_role_arn and self._credentials_expired():
                logger.info("Assumed role credentials expired or expiring soon, refreshing...")

            return self._refresh_client_locked()
```

The `client` property (line 136-139) is unchanged — it still calls `self._get_kendra_client()`, which keeps its existing sync signature.

- [ ] **Step 6: Run the affected test files**

Run: `cd ibt-agent && python -m pytest tests/unit/test_kendra_service.py tests/unit/test_kendra_assume_role.py tests/unit/test_kendra_edge_cases.py -v`
Expected: All pass except `test_assume_kendra_role_success` (pre-existing, expected, unrelated `RoleSessionName` assertion).

- [ ] **Step 7: Run the full suite**

Run: `cd ibt-agent && python -m pytest tests/ -v`
Expected: All green except the one pre-existing `test_assume_kendra_role_success` failure. No other files should be affected — `hybrid_ibt.py`, routes, and their tests are untouched since `get_ncct_ids_by_product`/`_get_kendra_client` keep the same sync signatures.

- [ ] **Step 8: Commit**

```bash
git add ibt-agent/src/services/kendra_service.py ibt-agent/tests/unit/test_kendra_service.py ibt-agent/tests/unit/test_kendra_assume_role.py
git commit -m "fix(ibt-agent): size Kendra/STS connection pools and lock credential refresh against races"
```

---

## Part B: orchestrator-agent

### Task 3: Pool-size settings

**Files:**
- Modify: `orchestrator-agent/src/config/settings.py`
- Create: `orchestrator-agent/tests/unit/test_settings.py`

**Interfaces:**
- Produces: `OrchestratorSettings.bedrock_max_pool_connections: int` (default 40), `OrchestratorSettings.sts_max_pool_connections: int` (default 20), `OrchestratorSettings.tool_http_max_connections: int` (default 40), `OrchestratorSettings.tool_http_max_keepalive_connections: int` (default 20) — consumed by Tasks 4 and 5.

- [ ] **Step 1: Write the failing settings tests**

Create `orchestrator-agent/tests/unit/test_settings.py` (this file does not exist yet):

```python
"""Unit tests for orchestrator configuration settings."""

from unittest.mock import patch

from src.config.settings import OrchestratorSettings, get_settings


class TestOrchestratorSettings:
    """Tests for OrchestratorSettings concurrency pool-size fields."""

    def test_concurrency_pool_defaults(self):
        """Test concurrency pool-size settings default values."""
        settings = OrchestratorSettings()

        assert settings.bedrock_max_pool_connections == 40
        assert settings.sts_max_pool_connections == 20
        assert settings.tool_http_max_connections == 40
        assert settings.tool_http_max_keepalive_connections == 20

    @patch.dict('os.environ', {
        'BEDROCK_MAX_POOL_CONNECTIONS': '60',
        'STS_MAX_POOL_CONNECTIONS': '30',
        'TOOL_HTTP_MAX_CONNECTIONS': '60',
        'TOOL_HTTP_MAX_KEEPALIVE_CONNECTIONS': '30',
    })
    def test_concurrency_pool_environment_variable_override(self):
        """Test that concurrency pool-size settings can be overridden via environment variables."""
        settings = OrchestratorSettings()

        assert settings.bedrock_max_pool_connections == 60
        assert settings.sts_max_pool_connections == 30
        assert settings.tool_http_max_connections == 60
        assert settings.tool_http_max_keepalive_connections == 30

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_settings.py -v`
Expected: FAIL with `AttributeError: 'OrchestratorSettings' object has no attribute 'bedrock_max_pool_connections'`.

- [ ] **Step 3: Add the fields to `OrchestratorSettings`**

Edit `orchestrator-agent/src/config/settings.py` — insert after the `bedrock_max_retries` field (currently ending at line 82) and before the `# Extended Thinking Configuration` comment:

```python
    # Concurrency Configuration
    bedrock_max_pool_connections: int = Field(
        default=40,
        gt=0,
        description="Max HTTP connection pool size for the Bedrock boto3 client"
    )
    sts_max_pool_connections: int = Field(
        default=20,
        gt=0,
        description="Max HTTP connection pool size for the STS boto3 client (Bedrock role assumption)"
    )
    tool_http_max_connections: int = Field(
        default=40,
        gt=0,
        description="Max total connections in the shared pooled HTTP client used to call tool APIs"
    )
    tool_http_max_keepalive_connections: int = Field(
        default=20,
        gt=0,
        description="Max keep-alive (reusable, idle) connections in the shared pooled HTTP client"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_settings.py -v`
Expected: PASS, 3 passed.

- [ ] **Step 5: Run the full suite to confirm nothing broke**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: Same pass/fail counts as before this task, plus the 3 new tests passing.

- [ ] **Step 6: Commit**

```bash
git add orchestrator-agent/src/config/settings.py orchestrator-agent/tests/unit/test_settings.py
git commit -m "feat(orchestrator-agent): add Bedrock/STS/tool-HTTP connection pool-size settings"
```

---

### Task 4: ChatModels — pool sizing and credential-refresh lock

**Files:**
- Modify: `orchestrator-agent/src/llm/client.py`
- Modify: `orchestrator-agent/tests/unit/test_llm/test_client.py`

**Interfaces:**
- Consumes: `OrchestratorSettings.bedrock_max_pool_connections`, `OrchestratorSettings.sts_max_pool_connections` (Task 3).
- Produces: no signature changes — `ChatModels._get_bedrock_client()`, `bedrock_model()`, `apply_guardrail()`, etc. keep their existing sync signatures.

- [ ] **Step 1: Write the failing pool-sizing test**

Add to `orchestrator-agent/tests/unit/test_llm/test_client.py`, inside `TestChatModelsInit`:

```python
    @patch("src.llm.client.boto3.client")
    def test_get_boto_config_uses_bedrock_max_pool_connections(self, mock_boto):
        from src.llm.client import ChatModels
        cm = ChatModels()
        config = cm._get_boto_config()
        assert config.max_pool_connections == cm.settings.bedrock_max_pool_connections
```

- [ ] **Step 2: Write the failing STS pool-sizing test**

Add a new test class to `orchestrator-agent/tests/unit/test_llm/test_client.py`:

```python
class TestAssumeBedrockRole:
    """Tests for _assume_bedrock_role STS client configuration."""

    @patch("src.llm.client.boto3.client")
    def test_sts_client_uses_sts_max_pool_connections(self, mock_boto):
        from src.llm.client import ChatModels

        mock_sts = MagicMock()
        mock_boto.return_value = mock_sts
        mock_sts.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': '2024-01-01T12:00:00Z'
            }
        }

        cm = ChatModels()
        cm.settings.bedrock_role_arn = 'arn:aws:iam::054940911799:role/orchestrator-bedrock-role'

        cm._assume_bedrock_role()

        sts_call = mock_boto.call_args
        assert sts_call[0][0] == 'sts'
        assert 'config' in sts_call[1]
        assert sts_call[1]['config'].max_pool_connections == cm.settings.sts_max_pool_connections
```

- [ ] **Step 3: Write the failing concurrency (lock) test**

Add to the same new `TestAssumeBedrockRole` class:

```python
    @patch("src.llm.client.boto3.client")
    def test_concurrent_get_bedrock_client_calls_assume_role_once(self, mock_boto):
        """Test that concurrent calls to _get_bedrock_client with expired credentials
        only trigger one assume_role call, not one per caller."""
        from concurrent.futures import ThreadPoolExecutor
        import time as time_module
        from src.llm.client import ChatModels

        mock_sts = MagicMock()
        mock_bedrock = MagicMock()

        def boto_client_side_effect(service_name=None, **kwargs):
            if service_name == 'sts':
                time_module.sleep(0.05)  # widen the race window
                return mock_sts
            return mock_bedrock

        mock_boto.side_effect = boto_client_side_effect
        mock_sts.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'ASIA123456789',
                'SecretAccessKey': 'secret123',
                'SessionToken': 'token123',
                'Expiration': '2024-01-01T12:00:00Z'
            }
        }

        cm = ChatModels()
        cm.settings.bedrock_role_arn = 'arn:aws:iam::054940911799:role/orchestrator-bedrock-role'

        with ThreadPoolExecutor(max_workers=10) as pool:
            clients = list(pool.map(lambda _: cm._get_bedrock_client(), range(10)))

        assert mock_sts.assume_role.call_count == 1
        assert all(c is mock_bedrock for c in clients)
```

Note: `boto3.client(service_name="bedrock-runtime", ...)` in `_get_bedrock_client`/`_refresh_client_locked` passes `service_name` as a keyword, while `_assume_bedrock_role`'s STS call passes it positionally (`boto3.client('sts', ...)`) — the side effect handles both by accepting `service_name=None` and falling back to positional args via `**kwargs` not being needed here since the mock records call args regardless; verify this matches by running Step 4 below.

- [ ] **Step 4: Run all three new tests to verify they fail**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_llm/test_client.py -v -k "pool_connections or concurrent_get_bedrock"`
Expected: All three FAIL — `_get_boto_config` has no `max_pool_connections`, `_assume_bedrock_role`'s STS client has no `config` kwarg at all, and `_get_bedrock_client` is unguarded so the race test's `assume_role.call_count == 1` assertion fails.

- [ ] **Step 5: Implement the Config, lock, and refactor in `client.py`**

Edit `orchestrator-agent/src/llm/client.py`:

Add `threading` to imports:

```python
import threading
from datetime import datetime, timezone, timedelta
```

Replace `_get_boto_config`:

```python
    def _get_boto_config(self) -> Config:
        return Config(
            read_timeout=self.settings.bedrock_read_timeout,
            connect_timeout=self.settings.bedrock_connect_timeout,
            retries={"max_attempts": self.settings.bedrock_max_retries, "mode": "adaptive"},
            max_pool_connections=self.settings.bedrock_max_pool_connections,
        )

    def _get_sts_boto_config(self) -> Config:
        """Get boto3 configuration for the STS client used in role assumption."""
        return Config(max_pool_connections=self.settings.sts_max_pool_connections)
```

In `_assume_bedrock_role`, change the STS client construction line:

```python
            sts_client = boto3.client('sts', region_name=self.settings.aws_region, config=self._get_sts_boto_config())
```

Add `self._client_lock = threading.Lock()` to `__init__`:

```python
    def __init__(self):
        self.settings = orchestrator_settings
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None
        self._client_lock = threading.Lock()
```

Replace `_get_bedrock_client` entirely, extracting a `_refresh_client_locked` helper (same pattern as `KendraService`):

```python
    def _refresh_client_locked(self) -> boto3.client:
        """Refresh (or create) the Bedrock client. Must be called while holding self._client_lock."""
        logger.info(f"Initializing Bedrock client: region={self.settings.aws_region}")

        if self.settings.bedrock_role_arn:
            logger.info("Using role assumption for Bedrock access")
            credentials = self._assume_bedrock_role()
            self._assumed_credentials = credentials

            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.settings.aws_region,
                config=self._get_boto_config(),
                **credentials
            )
        else:
            logger.info("Using default AWS credentials for Bedrock access")
            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.settings.aws_region,
                config=self._get_boto_config(),
            )

        return self._client

    def _get_bedrock_client(self) -> boto3.client:
        """Get Bedrock client with appropriate credentials (thread-safe)."""
        # Fast path: valid cached client, no lock needed.
        if self._client is not None and (not self.settings.bedrock_role_arn or not self._credentials_expired()):
            return self._client

        with self._client_lock:
            # Re-check inside the lock: another thread may have just refreshed it.
            if self._client is not None and (not self.settings.bedrock_role_arn or not self._credentials_expired()):
                return self._client

            if self._client is not None and self.settings.bedrock_role_arn and self._credentials_expired():
                logger.info("Assumed role credentials expired or expiring soon, refreshing...")

            return self._refresh_client_locked()
```

All other methods (`bedrock_model`, `bedrock_model_with_extended_thinking`, `bedrock_model_with_guardrails`, `apply_guardrail`, `get_model`) are unchanged — they still call `self._get_bedrock_client()` synchronously.

- [ ] **Step 6: Run the affected test file**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_llm/test_client.py -v`
Expected: All pass.

- [ ] **Step 7: Run the full suite**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: All green. No other files affected — every `ChatModels` method keeps its sync signature.

- [ ] **Step 8: Commit**

```bash
git add orchestrator-agent/src/llm/client.py orchestrator-agent/tests/unit/test_llm/test_client.py
git commit -m "fix(orchestrator-agent): size Bedrock/STS connection pools and lock credential refresh against races"
```

---

### Task 5: Shared pooled HTTP client for tool calls

**Files:**
- Create: `orchestrator-agent/src/http_client.py`
- Create: `orchestrator-agent/tests/unit/test_http_client.py`
- Modify: `orchestrator-agent/src/graph/nodes/tool_node_factory.py`
- Modify: `orchestrator-agent/src/api/app.py`

**Interfaces:**
- Consumes: `OrchestratorSettings.tool_http_max_connections`, `OrchestratorSettings.tool_http_max_keepalive_connections` (Task 3).
- Produces: `get_http_client() -> httpx.Client`, `close_http_client() -> None` — consumed by `tool_node_factory.py` and `app.py`.

- [ ] **Step 1: Write the failing tests for the new module**

Create `orchestrator-agent/tests/unit/test_http_client.py`:

```python
"""Unit tests for the shared, pooled HTTP client module."""

import httpx
import pytest

from src.http_client import get_http_client, close_http_client


class TestGetHttpClient:
    """Tests for get_http_client singleton behavior."""

    def teardown_method(self):
        close_http_client()

    def test_returns_httpx_client(self):
        client = get_http_client()
        assert isinstance(client, httpx.Client)

    def test_returns_same_instance_on_repeated_calls(self):
        first = get_http_client()
        second = get_http_client()
        assert first is second

    def test_limits_from_settings(self):
        from src.config.settings import get_settings
        settings = get_settings()

        client = get_http_client()

        assert client._limits.max_connections == settings.tool_http_max_connections
        assert client._limits.max_keepalive_connections == settings.tool_http_max_keepalive_connections


class TestCloseHttpClient:
    """Tests for shared client shutdown."""

    def test_close_allows_new_instance_after(self):
        first = get_http_client()
        close_http_client()
        second = get_http_client()
        assert first is not second
        close_http_client()

    def test_close_is_idempotent(self):
        get_http_client()
        close_http_client()
        close_http_client()  # must not raise
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_http_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.http_client'`.

- [ ] **Step 3: Implement `src/http_client.py`**

Create `orchestrator-agent/src/http_client.py`:

```python
"""Shared, pooled HTTP client for calling tool APIs (e.g. ibt-agent)."""

from typing import Optional

import httpx

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_http_client: Optional[httpx.Client] = None


def get_http_client() -> httpx.Client:
    """Get the singleton pooled httpx.Client, creating it if needed.

    Reusing one client across requests (instead of opening a fresh
    httpx.Client per call) keeps TCP+TLS connections to tool endpoints
    alive between requests via keep-alive, avoiding a repeated handshake
    on every tool call.
    """
    global _http_client
    if _http_client is None:
        settings = get_settings()
        logger.info(
            f"Creating shared pooled HTTP client: "
            f"max_connections={settings.tool_http_max_connections}, "
            f"max_keepalive_connections={settings.tool_http_max_keepalive_connections}"
        )
        _http_client = httpx.Client(
            timeout=settings.tool_timeout,
            limits=httpx.Limits(
                max_connections=settings.tool_http_max_connections,
                max_keepalive_connections=settings.tool_http_max_keepalive_connections,
            ),
        )
    return _http_client


def close_http_client() -> None:
    """Close and clear the singleton pooled httpx.Client."""
    global _http_client
    if _http_client is not None:
        logger.info("Closing shared pooled HTTP client")
        _http_client.close()
        _http_client = None
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_http_client.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Wire the shared client into `tool_node_factory.py`**

Edit `orchestrator-agent/src/graph/nodes/tool_node_factory.py` — add the import:

```python
from src.http_client import get_http_client
```

Replace the body of `_call_tool_api`'s `try` block (the `with httpx.Client(timeout=settings.tool_timeout) as client:` line and everything nested under it) so the shared client is used without a `with` block (it must not be closed after each call — it's a long-lived singleton):

```python
    try:
        client = get_http_client()
        payload = {
            "userPrompt": effective_query or state.query,
            "sessionId": state.session_id,
            "context": {
                "userName": state.context.userName,
                "userType": state.context.userType,
                "source": state.context.source,
                "productId": state.context.productId,
            },
        }
        if state.context.promptId:
            payload["context"]["promptId"] = state.context.promptId

        headers = {"Content-Type": "application/json"}
        if state.authorization:
            headers["Authorization"] = state.authorization

        response = client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        logger.info("Received response from %s: %s", tool_name, data)
        response_text = data.get("responseText", "")
        raw_metadata = data.get("metadata", [])
        agent_metadata = [AgentMetadata.model_validate(m) for m in raw_metadata]

        return response_text, agent_metadata

    except httpx.TimeoutException:
        raise ToolTimeoutError(tool_name, settings.tool_timeout)
    except httpx.HTTPStatusError as e:
        raise ToolUnavailableError(tool_name, f"HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise ToolUnavailableError(tool_name, str(e))
```

- [ ] **Step 6: Wire shutdown into `app.py`**

Edit `orchestrator-agent/src/api/app.py` — add the import and close the client on shutdown:

```python
from src.api.dependencies import get_orchestrator
from src.api.error_handlers import register_exception_handlers
from src.api.routes import health, invocations
from src.http_client import close_http_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: validate tools.yaml. If invalid, exception propagates → server aborts."""
    logger.info("Initializing Orchestrator Agent to validate configuration...")
    get_orchestrator()           # raises ToolRegistryError if tools.yaml is broken
    logger.info("Orchestrator Agent initialized successfully.")
    yield
    close_http_client()
```

- [ ] **Step 7: Run the affected test files**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_nodes/test_tool_node_factory.py tests/unit/test_http_client.py -v`
Expected: All pass — every existing `test_tool_node_factory.py` test patches `_call_tool_api` directly (not the `httpx.Client` inside it), so none of them observe or need to change for this internal swap.

- [ ] **Step 8: Run the full suite**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: All green.

- [ ] **Step 9: Commit**

```bash
git add orchestrator-agent/src/http_client.py orchestrator-agent/tests/unit/test_http_client.py orchestrator-agent/src/graph/nodes/tool_node_factory.py orchestrator-agent/src/api/app.py
git commit -m "perf(orchestrator-agent): reuse a shared, pooled HTTP client for tool calls instead of opening one per request"
```

---

## After Phase 1: measuring before Phase 2

Once all 5 tasks are committed, load-test both services (e.g. k6 or locust against `/invocations` on ibt-agent and orchestrator-agent) at realistic peak concurrency, comparing p50/p95/p99 latency and max sustained throughput against a pre-phase-1 baseline.

- If pool sizing alone shows a meaningful improvement and the remaining ceiling is comfortably above observed peak concurrency, phase 2 (full async conversion + dedicated `ThreadPoolExecutor`, per `docs/superpowers/plans/2026-08-03-async-concurrency-implementation-plan.md`) can be deferred indefinitely — phase 1 already removed the connection-pool mismatch and the credential-refresh race, which were the two correctness/efficiency issues; the remaining implicit 40-thread ceiling from Starlette/anyio may simply be sufficient.
- If load testing still shows saturation (requests queuing, p99 degrading under concurrent load well before the boto3/httpx pools are exhausted), that points at the thread-pool ceiling itself as the bottleneck, not connection pooling — proceed to phase 2, which replaces the implicit shared ceiling with an explicit, owned, tunable `ThreadPoolExecutor` per service.
- Either way, phase 2's diff shrinks from here: the pool-size settings, the `_get_sts_boto_config`/`_refresh_client_locked` extraction, and the credential-refresh lock added in this phase are exactly the pieces phase 2's design spec calls for too (see spec's "Component changes" sections) — phase 2 only needs to add the `async def` conversions, the dedicated executor, and wrap the now-already-locked, already-pool-sized client/query calls in `asyncio.to_thread`. None of this phase's work is thrown away.
