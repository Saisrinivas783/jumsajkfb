# Async Concurrency Implementation Plan — ibt-agent & orchestrator-agent

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give ibt-agent and orchestrator-agent explicit, owned concurrency: async request handling end-to-end, a dedicated `ThreadPoolExecutor` set as each process's event-loop default executor, and configurable `boto3`/`httpx` connection-pool sizes — replacing today's accidental ceiling (Starlette's shared 40-thread default pool, boto3's unset-default 10-connection pool). Also fixes a latent thread-safety bug: unguarded credential-refresh races in `KendraService`/`ChatModels` that duplicate STS calls under concurrency.

**Architecture:** See `docs/superpowers/specs/2026-08-03-async-concurrency-design.md` for the full design and rationale (approved). Summary: `async def` routes → async agent/orchestrator methods → (orchestrator only) async LangGraph nodes → leaf `boto3` calls wrapped in `await asyncio.to_thread(...)`, riding a custom default executor set once at startup; `ChatBedrockConverse.ainvoke()` needs no wrapping (LangChain's built-in async-via-executor already rides that same default executor); the orchestrator's HTTP call to ibt-agent moves to a shared, pooled `httpx.AsyncClient`.

**Tech Stack:** Python, FastAPI (async), `concurrent.futures.ThreadPoolExecutor`, `boto3`/`botocore.config.Config`, `httpx.AsyncClient`, LangGraph `ainvoke`, `pytest-asyncio`.

## Global Constraints

- No change to public request/response schemas or endpoint behavior — purely an internal execution-model change.
- No `aioboto3` adoption — raw boto3 calls are wrapped in `asyncio.to_thread`, per the approved design's rejected-alternatives section.
- No backpressure/load-shedding mechanism — out of scope, called out as a known limitation in the spec.
- Pool-size defaults (moderate tier, approved): `thread_pool_max_workers=20`, `kendra_max_pool_connections=20`, `bedrock_max_pool_connections=20`, `sts_max_pool_connections=10`, `tool_http_max_connections=20`, `tool_http_max_keepalive_connections=10`. All are env-configurable `pydantic` `Field`s with these defaults.
- The credential-refresh lock is `threading.Lock`, never `asyncio.Lock` — the guarded critical section (STS call + client construction) always executes inside a worker thread via `asyncio.to_thread`, never directly on the event loop, never held across an `await`.
- Every task ends with `pytest tests/ -v` passing inside the relevant subproject (`ibt-agent/` or `orchestrator-agent/`), run from that directory.
- Do not touch `src/config/product_mapping.py`, `tests/unit/test_product_mapping.py`, or `orchestrator-agent`'s unrelated modules (`src/tools/`, `src/schemas/`, `src/exceptions.py`, `src/utils/`) — out of scope for this plan.
- ibt-agent's one pre-existing, out-of-scope test failure — `tests/unit/test_kendra_assume_role.py::TestKendraAssumeRole::test_assume_kendra_role_success` (a `RoleSessionName` assertion mismatch unrelated to concurrency) — must remain untouched and is expected to still fail after every task. Do not "fix" it.

---

## Part A: ibt-agent

### Task 1: Pool-size settings and pytest-asyncio setup

**Files:**
- Modify: `ibt-agent/src/config/settings.py`
- Modify: `ibt-agent/tests/unit/test_settings.py`
- Modify: `ibt-agent/requirements.txt`
- Modify: `ibt-agent/pytest.ini`

**Interfaces:**
- Produces: `IBTSettings.kendra_max_pool_connections: int` (default 20), `IBTSettings.sts_max_pool_connections: int` (default 10), `IBTSettings.thread_pool_max_workers: int` (default 20) — consumed by Tasks 2 and 3.

- [ ] **Step 1: Add the three new fields to `IBTSettings`**

Edit `ibt-agent/src/config/settings.py` — insert after the `kendra_page_size` field (currently ending at line 65) and before the `# DXAIService Configuration` comment:

```python
    # Concurrency Configuration
    kendra_max_pool_connections: int = Field(
        default=20,
        gt=0,
        description="Max HTTP connection pool size for the Kendra boto3 client"
    )
    sts_max_pool_connections: int = Field(
        default=10,
        gt=0,
        description="Max HTTP connection pool size for the STS boto3 client (Kendra role assumption)"
    )
    thread_pool_max_workers: int = Field(
        default=20,
        gt=0,
        description="Max worker threads in this process's dedicated ThreadPoolExecutor"
    )
```

- [ ] **Step 2: Write the failing settings test**

Add to `ibt-agent/tests/unit/test_settings.py`, inside `TestIBTSettings`:

```python
    def test_concurrency_defaults(self):
        """Test concurrency-related settings default values."""
        settings = IBTSettings()

        assert settings.kendra_max_pool_connections == 20
        assert settings.sts_max_pool_connections == 10
        assert settings.thread_pool_max_workers == 20

    @patch.dict('os.environ', {
        'KENDRA_MAX_POOL_CONNECTIONS': '50',
        'STS_MAX_POOL_CONNECTIONS': '15',
        'THREAD_POOL_MAX_WORKERS': '30',
    })
    def test_concurrency_environment_variable_override(self):
        """Test that concurrency settings can be overridden via environment variables."""
        settings = IBTSettings()

        assert settings.kendra_max_pool_connections == 50
        assert settings.sts_max_pool_connections == 15
        assert settings.thread_pool_max_workers == 30
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ibt-agent && python -m pytest tests/unit/test_settings.py -v -k concurrency`
Expected: FAIL with `AttributeError: 'IBTSettings' object has no attribute 'kendra_max_pool_connections'` (Step 1 not yet applied) — if you did Step 1 first, instead do Step 2 first and confirm this failure, then apply Step 1.

- [ ] **Step 4: Verify it passes**

Apply Step 1 if not already done. Run: `cd ibt-agent && python -m pytest tests/unit/test_settings.py -v -k concurrency`
Expected: PASS, 2 passed.

- [ ] **Step 5: Add `pytest-asyncio` to `requirements.txt`**

Edit `ibt-agent/requirements.txt` — insert alphabetically after `python-dotenv==1.2.1`:

```
pytest-asyncio==1.2.0
```

Run: `cd ibt-agent && pip install -r requirements.txt`

- [ ] **Step 6: Configure `asyncio_mode` in `pytest.ini`**

Edit `ibt-agent/pytest.ini` — add `asyncio_mode = auto` so `async def test_*` functions run without needing `@pytest.mark.asyncio` on each one (still fine to use the marker explicitly where it aids readability, but not required):

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

- [ ] **Step 7: Run the full suite to confirm nothing broke**

Run: `cd ibt-agent && python -m pytest tests/ -v`
Expected: Same pass/fail counts as before this task (all green except the one pre-existing `test_assume_kendra_role_success` failure), plus the 2 new concurrency-settings tests passing.

- [ ] **Step 8: Commit**

```bash
git add ibt-agent/src/config/settings.py ibt-agent/tests/unit/test_settings.py ibt-agent/requirements.txt ibt-agent/pytest.ini
git commit -m "feat(ibt-agent): add concurrency pool-size settings and pytest-asyncio setup"
```

---

### Task 2: Dedicated ThreadPoolExecutor module

**Files:**
- Create: `ibt-agent/src/executor.py`
- Create: `ibt-agent/tests/unit/test_executor.py`

**Interfaces:**
- Consumes: `IBTSettings.thread_pool_max_workers` (from Task 1).
- Produces: `get_executor() -> ThreadPoolExecutor`, `set_as_default_executor() -> None`, `shutdown_executor() -> None` — consumed by Task 5 (`app.py` lifespan).

- [ ] **Step 1: Write the failing tests**

Create `ibt-agent/tests/unit/test_executor.py`:

```python
"""Unit tests for the dedicated ThreadPoolExecutor module."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.executor import get_executor, set_as_default_executor, shutdown_executor


class TestGetExecutor:
    """Tests for get_executor singleton behavior."""

    def teardown_method(self):
        shutdown_executor()

    def test_returns_thread_pool_executor(self):
        executor = get_executor()
        assert isinstance(executor, ThreadPoolExecutor)

    def test_returns_same_instance_on_repeated_calls(self):
        first = get_executor()
        second = get_executor()
        assert first is second

    def test_max_workers_from_settings(self):
        executor = get_executor()
        assert executor._max_workers == 20


class TestSetAsDefaultExecutor:
    """Tests for wiring the executor into the running event loop."""

    def teardown_method(self):
        shutdown_executor()

    @pytest.mark.asyncio
    async def test_asyncio_to_thread_uses_dedicated_executor(self):
        set_as_default_executor()
        executor = get_executor()

        result = await asyncio.to_thread(lambda: "ran-in-dedicated-pool")

        assert result == "ran-in-dedicated-pool"
        # The executor's thread names carry our pool's default naming;
        # verifying via a live-thread-count sanity check is sufficient here.
        assert executor._threads


class TestShutdownExecutor:
    """Tests for executor shutdown."""

    def test_shutdown_allows_new_instance_after(self):
        first = get_executor()
        shutdown_executor()
        second = get_executor()
        assert first is not second

    def test_shutdown_is_idempotent(self):
        get_executor()
        shutdown_executor()
        shutdown_executor()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ibt-agent && python -m pytest tests/unit/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.executor'`.

- [ ] **Step 3: Implement `src/executor.py`**

Create `ibt-agent/src/executor.py`:

```python
"""Dedicated ThreadPoolExecutor for offloading blocking calls (boto3, etc.)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    """Get the singleton dedicated ThreadPoolExecutor, creating it if needed."""
    global _executor
    if _executor is None:
        settings = get_settings()
        logger.info(f"Creating dedicated ThreadPoolExecutor: max_workers={settings.thread_pool_max_workers}")
        _executor = ThreadPoolExecutor(max_workers=settings.thread_pool_max_workers)
    return _executor


def set_as_default_executor() -> None:
    """Set the dedicated executor as the running event loop's default executor.

    After this call, bare `asyncio.to_thread(...)` calls anywhere in this
    process route through the dedicated pool instead of asyncio's own
    internal default.
    """
    loop = asyncio.get_running_loop()
    loop.set_default_executor(get_executor())


def shutdown_executor() -> None:
    """Shut down the dedicated executor, waiting for in-flight work to finish."""
    global _executor
    if _executor is not None:
        logger.info("Shutting down dedicated ThreadPoolExecutor")
        _executor.shutdown(wait=True)
        _executor = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ibt-agent && python -m pytest tests/unit/test_executor.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Run the full suite**

Run: `cd ibt-agent && python -m pytest tests/ -v`
Expected: All green except the one pre-existing unrelated failure.

- [ ] **Step 6: Commit**

```bash
git add ibt-agent/src/executor.py ibt-agent/tests/unit/test_executor.py
git commit -m "feat(ibt-agent): add dedicated ThreadPoolExecutor module"
```

---

### Task 3: KendraService — pool sizing, credential-refresh lock, async conversion

**Files:**
- Modify: `ibt-agent/src/services/kendra_service.py`
- Modify: `ibt-agent/tests/unit/test_kendra_service.py`
- Modify: `ibt-agent/tests/unit/test_kendra_assume_role.py`

**Interfaces:**
- Consumes: `IBTSettings.kendra_max_pool_connections`, `IBTSettings.sts_max_pool_connections` (Task 1).
- Produces: `KendraService.get_ncct_ids_by_product(query: str, product_id: str = None) -> List[str]` becomes `async def` (same name, same signature, now a coroutine). Module-level `get_ncct_ids_by_product(query: str, product_id: str = None) -> List[str]` becomes `async def` too. `KendraService._get_kendra_client()` becomes `async def KendraService._get_kendra_client() -> boto3.client`. `KendraService.client` property is **removed** (a property can't be async — Task 4 calls `await service._get_kendra_client()` directly instead of the `.client` property). Consumed by Task 4.

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
        import asyncio
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

        async def run_concurrent_calls():
            return await asyncio.gather(*[
                self.kendra_service._get_kendra_client() for _ in range(10)
            ])

        clients = asyncio.run(run_concurrent_calls())

        assert mock_sts.assume_role.call_count == 1
        assert all(c is mock_kendra for c in clients)
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd ibt-agent && python -m pytest tests/unit/test_kendra_assume_role.py::TestKendraAssumeRole::test_concurrent_get_kendra_client_calls_assume_role_once -v`
Expected: FAIL — `_get_kendra_client` is currently sync (not awaitable) and unguarded, so this test fails at the `await` (TypeError, since a sync method isn't awaitable) before it can even test the race.

- [ ] **Step 5: Implement the Config, lock, and async changes in `kendra_service.py`**

Edit `ibt-agent/src/services/kendra_service.py` — replace `import threading` addition, `_get_boto_config`, `_assume_kendra_role`, `_get_kendra_client`, the `client` property, and `get_ncct_ids_by_product` (both instance and module-level):

```python
"""Direct AWS Kendra integration without fallback logic."""

import asyncio
import boto3
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError
from botocore.config import Config
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QueryLimitExceededError(Exception):
    """Raised when Kendra query limit is exceeded."""
    pass


# Product to Plan/Brochure mapping
PRODUCT_MAPPING = {
    "1": {"plan": "standard/basic", "brochure": "fehb"},
    "4": {"plan": "standard/basic", "brochure": "fehb"},
    "6": {"plan": "blue focus", "brochure": "fehb"},
    "7": {"plan": "standard/basic", "brochure": "pshb"},
    "8": {"plan": "standard/basic", "brochure": "pshb"},
    "9": {"plan": "blue focus", "brochure": "pshb"},
}

class KendraService:
    """Direct AWS Kendra semantic search client without fallback logic."""

    CREDENTIALS_REFRESH_BUFFER = timedelta(minutes=5)

    def __init__(self, index_id: Optional[str] = None, region: Optional[str] = None):
        self.settings = get_settings()
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None
        self._client_lock = threading.Lock()

        if index_id and region:
            self.index_id = index_id
            self.region = region
        else:
            self.index_id = self.settings.kendra_index_id
            self.region = self.settings.aws_region

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

    def _assume_kendra_role(self) -> Dict[str, str]:
        """Assume the Kendra role and return temporary credentials."""
        if not self.settings.kendra_role_arn:
            raise ValueError("KENDRA_ROLE_ARN is not configured")

        try:
            logger.info(f"Assuming Kendra role: {self.settings.kendra_role_arn}")

            sts_client = boto3.client('sts', region_name=self.region, config=self._get_sts_boto_config())

            response = sts_client.assume_role(
                RoleArn=self.settings.kendra_role_arn,
                RoleSessionName=self.settings.kendra_session_name,
                DurationSeconds=self.settings.kendra_role_duration
            )

            credentials = response['Credentials']
            self._credentials_expiration = credentials['Expiration']
            logger.info(f"Successfully assumed Kendra role. Session expires at: {credentials['Expiration']}")

            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to assume Kendra role {self.settings.kendra_role_arn}: {error_code} - {error_msg}")
            raise RuntimeError(f"Role assumption failed: {error_code} - {error_msg}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RuntimeError(f"Role assumption failed: {str(e)}") from e

    def _credentials_expired(self) -> bool:
        """Check if assumed credentials are expired or about to expire."""
        if self._credentials_expiration is None:
            return True
        return datetime.now(timezone.utc) >= self._credentials_expiration - self.CREDENTIALS_REFRESH_BUFFER

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

    def _get_kendra_client_sync(self) -> boto3.client:
        """Get Kendra client with appropriate credentials (thread-safe, blocking)."""
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

    async def _get_kendra_client(self) -> boto3.client:
        """Get Kendra client with appropriate credentials (async, offloaded to the executor)."""
        return await asyncio.to_thread(self._get_kendra_client_sync)

    def _build_attribute_filter(self, product_config: Dict[str, str]) -> Dict[str, Any]:
        """Build Kendra AttributeFilter for product filtering."""
        if not product_config:
            return None

        return {
            "AndAllFilters": [
                {
                    "EqualsTo": {
                        "Key": "plan",
                        "Value": {
                            "StringValue": product_config['plan']
                        }
                    }
                },
                {
                    "EqualsTo": {
                        "Key": "brochure",
                        "Value": {
                            "StringValue": product_config['brochure']
                        }
                    }
                }
            ]
        }

    async def get_ncct_ids_by_product(self, query: str, product_id: str = None) -> List[str]:
        """Search Kendra with product filter and return only NCCT IDs."""
        try:
            logger.info(f"Getting NCCT IDs for product {product_id} with query: {query[:50]}...")

            product_config = None
            attribute_filter = None

            if product_id and product_id in PRODUCT_MAPPING:
                product_config = PRODUCT_MAPPING[product_id]
                attribute_filter = self._build_attribute_filter(product_config)
                logger.info(f"Product {product_id} requires plan: {product_config['plan']}, brochure: {product_config['brochure']}")

            query_params = {
                'IndexId': self.index_id,
                'QueryText': query,
                'PageSize': self.settings.kendra_page_size,
                'RequestedDocumentAttributes': ['NCCTID']
            }

            if attribute_filter:
                query_params['AttributeFilter'] = attribute_filter

            client = await self._get_kendra_client()
            response = await asyncio.to_thread(client.query, **query_params)
            items = response.get('ResultItems', [])

            if not items:
                logger.info("No results found")
                return []

            ncct_ids = [
                attr['Value']['StringValue']
                for item in items
                for attr in item.get('DocumentAttributes', [])
                if (attr['Key'] == 'NCCTID' and attr.get('Value', {}).get('StringValue') and
                    item.get('ScoreAttributes', {}).get('ScoreConfidence') in ['VERY_HIGH', 'HIGH', 'MEDIUM'])
            ]

            logger.info(f"Extracted {len(ncct_ids)} NCCT IDs for product {product_id}")
            return ncct_ids

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

_kendra_service = None

def get_kendra_service() -> KendraService:
    """Get singleton KendraService instance."""
    global _kendra_service
    if _kendra_service is None:
        _kendra_service = KendraService()
    return _kendra_service

async def get_ncct_ids_by_product(query: str, product_id: str = None) -> List[str]:
    """Get NCCT IDs from Kendra search filtered by product."""
    service = get_kendra_service()
    return await service.get_ncct_ids_by_product(query, product_id)
```

Note: the `client` property (`@property def client`) is removed — it can't be `async`, and its only consumers (`get_ncct_ids_by_product`, and tests) are updated in this same step to call `await self._get_kendra_client()` directly, or in tests, `await service._get_kendra_client()`.

- [ ] **Step 6: Update `test_kendra_service.py` for the async API**

Edit `ibt-agent/tests/unit/test_kendra_service.py` — every test that calls `service.get_ncct_ids_by_product(...)` or `service.client` needs updating. Full rewrite of the affected tests:

```python
class TestKendraService:
    """Tests for KendraService class."""

    @pytest.mark.asyncio
    @patch('src.services.kendra_service.boto3.client')
    async def test_init(self, mock_boto):
        mock_client = MagicMock()
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')
        assert service.index_id == 'test-index'

        # Force client initialization
        _ = await service._get_kendra_client()

        assert mock_boto.called

    @pytest.mark.asyncio
    @patch('boto3.client')
    async def test_get_ncct_ids_by_product_success(self, mock_boto):
        """Test get_ncct_ids_by_product returns only NCCT IDs with AttributeFilter."""
        mock_client = MagicMock()
        mock_client.query.return_value = {
            'ResultItems': [
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT123'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Dental'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'VERY_HIGH'}
                },
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT456'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Vision'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'}
                },
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT789'}},
                        {'Key': 'Service_Name', 'Value': {'StringValue': 'Low Confidence'}},
                        {'Key': 'plan', 'Value': {'StringValue': 'standard/basic'}},
                        {'Key': 'brochure', 'Value': {'StringValue': 'fehb'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'MEDIUM'}
                }
            ]
        }
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')
        ncct_ids = await service.get_ncct_ids_by_product('dental benefits', '1')

        assert ncct_ids == ['NCCT123', 'NCCT456', 'NCCT789']

        expected_filter = {
            "AndAllFilters": [
                {
                    "EqualsTo": {
                        "Key": "plan",
                        "Value": {
                            "StringValue": "standard/basic"
                        }
                    }
                },
                {
                    "EqualsTo": {
                        "Key": "brochure",
                        "Value": {
                            "StringValue": "fehb"
                        }
                    }
                }
            ]
        }

        mock_client.query.assert_called_once_with(
            IndexId='test-index',
            QueryText='dental benefits',
            PageSize=DEFAULT_PAGE_SIZE,
            RequestedDocumentAttributes=['NCCTID'],
            AttributeFilter=expected_filter
        )

    @pytest.mark.asyncio
    @patch('boto3.client')
    async def test_get_ncct_ids_by_product_no_product_filter(self, mock_boto):
        """Test get_ncct_ids_by_product without product filter."""
        mock_client = MagicMock()
        mock_client.query.return_value = {
            'ResultItems': [
                {
                    'DocumentAttributes': [
                        {'Key': 'NCCTID', 'Value': {'StringValue': 'NCCT789'}}
                    ],
                    'ScoreAttributes': {'ScoreConfidence': 'HIGH'}
                }
            ]
        }
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')
        ncct_ids = await service.get_ncct_ids_by_product('general query', None)

        assert ncct_ids == ['NCCT789']

        mock_client.query.assert_called_once_with(
            IndexId='test-index',
            QueryText='general query',
            PageSize=DEFAULT_PAGE_SIZE,
            RequestedDocumentAttributes=['NCCTID']
        )

    @patch('src.services.kendra_service.get_settings')
    def test_build_attribute_filter(self, mock_settings):
        """Test _build_attribute_filter creates correct filter structure. (unchanged: sync, no I/O)"""
        mock_settings.return_value = MagicMock(kendra_index_id='test-index', aws_region='us-east-1')
        service = KendraService()

        product_config = {"plan": "standard/basic", "brochure": "pshb"}
        filter_result = service._build_attribute_filter(product_config)

        expected = {
            "AndAllFilters": [
                {
                    "EqualsTo": {
                        "Key": "plan",
                        "Value": {
                            "StringValue": "standard/basic"
                        }
                    }
                },
                {
                    "EqualsTo": {
                        "Key": "brochure",
                        "Value": {
                            "StringValue": "pshb"
                        }
                    }
                }
            ]
        }

        assert filter_result == expected

    @patch('src.services.kendra_service.get_settings')
    def test_build_attribute_filter_none_config(self, mock_settings):
        """Test _build_attribute_filter returns None for empty config. (unchanged: sync, no I/O)"""
        mock_settings.return_value = MagicMock(kendra_index_id='test-index', aws_region='us-east-1')
        service = KendraService()

        filter_result = service._build_attribute_filter(None)
        assert filter_result is None

        filter_result = service._build_attribute_filter({})
        assert filter_result is None

    @pytest.mark.asyncio
    @patch('boto3.client')
    async def test_get_ncct_ids_by_product_empty_results(self, mock_boto):
        """Test get_ncct_ids_by_product returns empty list when no results."""
        mock_client = MagicMock()
        mock_client.query.return_value = {'ResultItems': []}
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')
        ncct_ids = await service.get_ncct_ids_by_product('unknown query', '1')

        assert ncct_ids == []

    @pytest.mark.asyncio
    @patch('boto3.client')
    async def test_get_ncct_ids_by_product_exception_handling(self, mock_boto):
        """Test get_ncct_ids_by_product raises exception on error."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception('Kendra error')
        mock_boto.return_value = mock_client

        service = KendraService('test-index', 'us-east-1')

        with pytest.raises(RuntimeError, match="Kendra search failed for product 1: Kendra error"):
            await service.get_ncct_ids_by_product('test query', '1')

    @pytest.mark.asyncio
    @patch('boto3.client')
    async def test_get_ncct_ids_by_product_throttling_raises_query_limit_exceeded(self, mock_boto):
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
            await service.get_ncct_ids_by_product('dental benefits', '1')

    @pytest.mark.asyncio
    @patch('boto3.client')
    async def test_get_ncct_ids_by_product_client_error_non_throttling(self, mock_boto):
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
            await service.get_ncct_ids_by_product('dental benefits', '1')

    @pytest.mark.asyncio
    @patch('src.services.kendra_service.get_kendra_service')
    async def test_get_ncct_ids_by_product_function(self, mock_get_service):
        """Test the convenience function get_ncct_ids_by_product."""
        mock_service = MagicMock()

        async def fake_get_ncct_ids(query, product_id):
            return ['NCCT123', 'NCCT456']
        mock_service.get_ncct_ids_by_product = fake_get_ncct_ids
        mock_get_service.return_value = mock_service

        result = await get_ncct_ids_by_product('dental benefits', '1')

        assert result == ['NCCT123', 'NCCT456']

    @patch('src.services.kendra_service.boto3.client')
    def test_get_boto_config_uses_kendra_max_pool_connections(self, mock_boto):
        """Test _get_boto_config sets max_pool_connections from settings. (unchanged: sync, no I/O)"""
        service = KendraService('test-index', 'us-east-1')
        config = service._get_boto_config()
        assert config.max_pool_connections == service.settings.kendra_max_pool_connections
```

The transformation applied to every test above: `def test_x(self, ...)` → `async def test_x(self, ...)` with `@pytest.mark.asyncio` added, and every call to `service.get_ncct_ids_by_product(...)` or the module-level `get_ncct_ids_by_product(...)` gets `await` prepended. `test_build_attribute_filter`, `test_build_attribute_filter_none_config`, and `test_get_boto_config_uses_kendra_max_pool_connections` stay synchronous — they test methods with no I/O and no `async` in their signature.

- [ ] **Step 7: Update `test_kendra_assume_role.py` for the async API**

Apply the identical transformation pattern (add `@pytest.mark.asyncio`, `async def`, `await` before any `_get_kendra_client()` call) to these existing tests in `ibt-agent/tests/unit/test_kendra_assume_role.py`:
- `test_get_kendra_client_with_role_assumption` — change `client = self.kendra_service._get_kendra_client()` to `client = await self.kendra_service._get_kendra_client()`.
- `test_get_kendra_client_without_role_assumption` — same change.
- `test_client_caching` — change both `client1 = self.kendra_service._get_kendra_client()` and `client2 = self.kendra_service._get_kendra_client()` to `await` calls.

`test_assume_kendra_role_success`, `test_assume_kendra_role_no_arn`, `test_assume_kendra_role_access_denied`, and `test_boto_config_creation` stay synchronous and **unmodified** — `_assume_kendra_role` and `_get_boto_config` remain sync methods (only `_get_kendra_client` and `get_ncct_ids_by_product` became async; `_assume_kendra_role` is called *from* the async path via `asyncio.to_thread`, but is itself still a plain sync function, per Step 5's `_refresh_client_locked`).

`test_assume_kendra_role_success` specifically: leave its `RoleSessionName='ibt-agent-kendra-session'` assertion exactly as-is — this is the pre-existing, out-of-scope failing assertion (Global Constraints); do not touch this test.

- [ ] **Step 8: Run the affected test files**

Run: `cd ibt-agent && python -m pytest tests/unit/test_kendra_service.py tests/unit/test_kendra_assume_role.py -v`
Expected: All pass except `test_assume_kendra_role_success` (pre-existing, expected).

- [ ] **Step 9: Run the full suite**

Run: `cd ibt-agent && python -m pytest tests/ -v`
Expected: Failures elsewhere are expected at this point — Task 4 (`hybrid_ibt.py`) and Task 5 (routes/integration tests) still call the now-async `KendraService`/`get_ncct_ids_by_product` synchronously and will break until those tasks land. Confirm the failures are confined to `tests/unit/test_hybrid_agent.py`, `tests/unit/test_api_routes.py`, `tests/integration/test_api_routes.py`, `tests/integration/test_enhanced_kendra_integration.py`, `tests/unit/test_enhanced_kendra_service.py` (all fixed in Tasks 4–5), plus the one pre-existing `test_assume_kendra_role_success` failure.

- [ ] **Step 10: Commit**

```bash
git add ibt-agent/src/services/kendra_service.py ibt-agent/tests/unit/test_kendra_service.py ibt-agent/tests/unit/test_kendra_assume_role.py
git commit -m "feat(ibt-agent): async KendraService with pool sizing and credential-refresh lock"
```

---

### Task 4: HybridIBTAgent — async conversion

**Files:**
- Modify: `ibt-agent/src/agent/hybrid_ibt.py`
- Modify: `ibt-agent/tests/unit/test_hybrid_agent.py`
- Modify: `ibt-agent/tests/integration/test_enhanced_kendra_integration.py`
- Modify: `ibt-agent/tests/unit/test_enhanced_kendra_service.py`

**Interfaces:**
- Consumes: `async def get_ncct_ids_by_product(...)` (Task 3).
- Produces: `HybridIBTAgent.process_query(user_prompt: str, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]` becomes `async def`. `HybridIBTAgent._process_direct_kendra(user_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]` becomes `async def`. Consumed by Task 5.

- [ ] **Step 1: Write the failing test for the async signature**

Add to `ibt-agent/tests/unit/test_hybrid_agent.py`, inside `TestProcessQuery`:

```python
    @pytest.mark.asyncio
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    async def test_process_query_is_awaitable(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        """Test process_query is a coroutine function that must be awaited."""
        import inspect
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_kendra_service.return_value = MagicMock()

        async def fake_direct_process(*args, **kwargs):
            return {"success": True, "response_text": "ok", "confidence": 6.0}
        mock_direct_process.side_effect = fake_direct_process

        agent = HybridIBTAgent()
        assert inspect.iscoroutinefunction(agent.process_query)

        result = await agent.process_query("test", "sess-001", {"productId": "1"})
        assert result['sessionId'] == 'sess-001'
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd ibt-agent && python -m pytest tests/unit/test_hybrid_agent.py::TestProcessQuery::test_process_query_is_awaitable -v`
Expected: FAIL — `process_query` is currently a plain sync function; `await agent.process_query(...)` raises `TypeError: object dict can't be used in 'await' expression`.

- [ ] **Step 3: Convert `hybrid_ibt.py` to async**

Edit `ibt-agent/src/agent/hybrid_ibt.py` — change `process_query` and `_process_direct_kendra` to `async def`, and `await` the calls to `_process_direct_kendra` and `get_ncct_ids_by_product`:

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

    async def process_query(self, user_prompt: str, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        logger.info(f"Processing query with context: userName={context.get('userName')}, userType={context.get('userType')}, productId={context.get('productId')}, source={context.get('source')}")

        try:
            result = await self._process_direct_kendra(user_prompt, context)

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

    async def _process_direct_kendra(self, user_prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        product_id = context['productId']

        logger.info(f"Processing direct Kendra query with product ID: {product_id}")

        # Always use product-filtered NCCT ID extraction
        ncct_ids = await get_ncct_ids_by_product(user_prompt, str(product_id))
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

- [ ] **Step 4: Update the rest of `test_hybrid_agent.py`**

In `TestDirectKendraMode`, both `test_process_direct_kendra_success` and `test_process_direct_kendra_requires_product_id` need `@pytest.mark.asyncio`, `async def`, and `await` before the `agent._process_direct_kendra(...)` call. Full rewrite of `TestDirectKendraMode`:

```python
class TestDirectKendraMode:
    """Tests for direct Kendra processing."""

    @pytest.mark.asyncio
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    @patch('src.agent.hybrid_ibt.get_ncct_ids_by_product')
    async def test_process_direct_kendra_success(self, mock_get_ncct_ids, mock_settings, mock_get_kendra_service):
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        async def fake_get_ncct_ids(query, product_id):
            return ['DENTAL_001', 'DENTAL_002']
        mock_get_ncct_ids.side_effect = fake_get_ncct_ids
        mock_get_kendra_service.return_value = MagicMock()

        agent = HybridIBTAgent()

        context = {'productId': '1'}
        result = await agent._process_direct_kendra("dental benefits", context)

        assert result['success'] is True
        response_text = result['response_text']
        assert isinstance(response_text, list)
        assert 'DENTAL_001' in response_text
        assert 'DENTAL_002' in response_text
        assert result['confidence'] == 8.0

    @pytest.mark.asyncio
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    async def test_process_direct_kendra_requires_product_id(self, mock_settings, mock_get_kendra_service):
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_kendra_service.return_value = MagicMock()

        agent = HybridIBTAgent()

        with pytest.raises(KeyError):
            await agent._process_direct_kendra("dental benefits", {})
```

In `TestProcessQuery`, apply the same pattern (`@pytest.mark.asyncio`, `async def`, `await agent.process_query(...)`, and mock `_process_direct_kendra` with an `async def` side effect instead of a plain `return_value`) to `test_process_query_limit_exceeded` and `test_process_query_exception_handling` (the third, `test_process_query_success`, is superseded by `test_process_query_is_awaitable` from Step 1 — remove the older `test_process_query_success` to avoid duplicate coverage of the same behavior):

```python
    @pytest.mark.asyncio
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    async def test_process_query_limit_exceeded(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        """Test that QueryLimitExceededError returns proper error message."""
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'
        mock_get_kendra_service.return_value = MagicMock()

        from src.services.kendra_service import QueryLimitExceededError

        async def raise_limit(*args, **kwargs):
            raise QueryLimitExceededError("Kendra query limit exceeded")
        mock_direct_process.side_effect = raise_limit

        agent = HybridIBTAgent()

        result = await agent.process_query("dental benefits", "sess-limit", {"productId": "1"})

        assert result['success'] is False
        assert result['sessionId'] == 'sess-limit'
        assert result['confidence'] == 0.0
        assert isinstance(result['responseText'], list)
        assert "maximum number of search requests" in result['responseText'][0]

    @pytest.mark.asyncio
    @patch('src.agent.hybrid_ibt.HybridIBTAgent._process_direct_kendra')
    @patch('src.services.kendra_service.get_kendra_service')
    @patch('src.agent.hybrid_ibt.get_settings')
    async def test_process_query_exception_handling(self, mock_settings, mock_get_kendra_service, mock_direct_process):
        mock_settings.return_value.kendra_role_arn = None
        mock_settings.return_value.aws_region = 'us-east-1'
        mock_settings.return_value.kendra_index_id = 'test-index'

        mock_get_kendra_service.return_value = MagicMock()

        from src.agent.hybrid_ibt import KendraSearchError

        async def raise_kendra_error(*args, **kwargs):
            raise KendraSearchError("Test error")
        mock_direct_process.side_effect = raise_kendra_error

        agent = HybridIBTAgent()

        result = await agent.process_query("test", "sess-001", {"productId": "1"})

        assert result['success'] is False
        assert "technical difficulties" in result['responseText'][0]
        assert result['sessionId'] == 'sess-001'
```

- [ ] **Step 5: Update `test_enhanced_kendra_integration.py` and `test_enhanced_kendra_service.py`**

In `ibt-agent/tests/integration/test_enhanced_kendra_integration.py`, `test_direct_mode_with_real_kendra_data`: add `@pytest.mark.asyncio`, `async def`, change `mock_get_ncct_ids.return_value = [...]` to an `async def fake_get_ncct_ids` side effect (matching Step 4's pattern), and `await agent._process_direct_kendra(...)`.

In `ibt-agent/tests/unit/test_enhanced_kendra_service.py`, `test_direct_mode_with_enhanced_kendra`: identical transformation.

- [ ] **Step 6: Run the affected test files**

Run: `cd ibt-agent && python -m pytest tests/unit/test_hybrid_agent.py tests/integration/test_enhanced_kendra_integration.py tests/unit/test_enhanced_kendra_service.py -v`
Expected: All pass.

- [ ] **Step 7: Run the full suite**

Run: `cd ibt-agent && python -m pytest tests/ -v`
Expected: Only Task 5's files (routes) still fail at this point, plus the one pre-existing failure.

- [ ] **Step 8: Commit**

```bash
git add ibt-agent/src/agent/hybrid_ibt.py ibt-agent/tests/unit/test_hybrid_agent.py ibt-agent/tests/integration/test_enhanced_kendra_integration.py ibt-agent/tests/unit/test_enhanced_kendra_service.py
git commit -m "feat(ibt-agent): async HybridIBTAgent.process_query"
```

---

### Task 5: Async route, app lifespan wiring, concurrency verification

**Files:**
- Modify: `ibt-agent/src/api/routes/invocations.py`
- Modify: `ibt-agent/src/api/app.py`
- Modify: `ibt-agent/tests/integration/test_api_routes.py`

**Interfaces:**
- Consumes: `async def process_query(...)` (Task 4), `get_executor`/`set_as_default_executor`/`shutdown_executor` (Task 2).
- Produces: `POST /IbtAgent/v2/invocations` is now handled by an `async def` route.

- [ ] **Step 1: Convert the route to async**

Edit `ibt-agent/src/api/routes/invocations.py`:

```python
"""Invocation routes for the IBT agent."""

from fastapi import APIRouter, Depends
from src.schemas.api import InvocationRequest, InvocationResponse
from src.agent.hybrid_ibt import HybridIBTAgent
from src.api.dependencies import get_ibt

router = APIRouter()

@router.post("/invocations", response_model=InvocationResponse)
async def invocations(
    payload: InvocationRequest,
    agent: HybridIBTAgent = Depends(get_ibt),
):
    """Process benefit and coverage inquiries per FEPOC specification."""
    context_dict = payload.context.model_dump()

    result = await agent.process_query(
        user_prompt=payload.user_prompt,
        session_id=payload.session_id,
        context=context_dict
    )

    return InvocationResponse(
        sessionId=result["sessionId"],
        responseText=result["responseText"],
        confidence=result["confidence"],
        success=result["success"],
        execution_time_ms=result["execution_time_ms"]
    )
```

- [ ] **Step 2: Add lifespan wiring to `app.py`**

Edit `ibt-agent/src/api/app.py`:

```python
"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health, invocations
from src.config.constants import SERVICE_NAME, SERVICE_DESCRIPTION, SERVICE_VERSION, API_PREFIX
from src.executor import set_as_default_executor, shutdown_executor
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: wire the dedicated ThreadPoolExecutor as the event loop's default executor."""
    logger.info("Starting IBT Agent: initializing dedicated thread pool executor")
    set_as_default_executor()
    yield
    logger.info("Shutting down IBT Agent: closing dedicated thread pool executor")
    shutdown_executor()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        version=SERVICE_VERSION,
        lifespan=_lifespan,
    )

    # Include routers
    app.include_router(health.router, prefix=API_PREFIX, tags=["Health"])
    app.include_router(invocations.router, prefix=API_PREFIX, tags=["Invocations"])

    return app
```

- [ ] **Step 3: Verify the existing integration tests still pass unchanged**

`TestClient` runs the app's lifespan automatically when used as a context manager (the `client` fixture in `ibt-agent/tests/integration/test_api_routes.py` already does `with TestClient(app) as c: yield c`), so no changes are needed to existing tests for the lifespan itself. `mock_hybrid_agent.process_query` (the mock used by these tests) is a plain `MagicMock` returning a dict synchronously — since the route now does `await agent.process_query(...)`, the mock must return an **awaitable**. Update the `mock_hybrid_agent` fixture in `ibt-agent/tests/conftest.py`:

```python
@pytest.fixture
def mock_hybrid_agent():
    """Mock HybridIBTAgent for testing."""
    mock = MagicMock()

    async def fake_process_query(*args, **kwargs):
        return {
            "sessionId": "sess-001",
            "confidence": 8.0,
            "responseText": "Here are your benefits: <a href='NCCT123'>Dental Coverage</a>",
            "success": True,
            "execution_time_ms": 250.5,
            "timestamp": "2024-01-15T10:30:00Z"
        }
    mock.process_query = MagicMock(side_effect=fake_process_query)
    return mock
```

- [ ] **Step 4: Run the integration tests to verify they pass**

Run: `cd ibt-agent && python -m pytest tests/integration/test_api_routes.py -v`
Expected: All pass — `test_valid_invocation_returns_200`, `test_valid_invocation_calls_agent`, `test_valid_invocation_response_structure`, `test_invocation_without_context_returns_422`, `test_agent_receives_context_when_provided` (the last one's `call_args.kwargs["context"]` assertion is unaffected by the mock change, since `MagicMock(side_effect=...)` still records `call_args` normally).

- [ ] **Step 5: Add a cross-request concurrency isolation test**

Add to `ibt-agent/tests/integration/test_api_routes.py`, inside `TestInvocationsRoute`:

```python
    def test_concurrent_invocations_do_not_cross_contaminate(self, client, mock_hybrid_agent):
        """Fire several concurrent requests with distinct sessionIds; each response
        must match its own request (no shared-state leakage under concurrency)."""
        import threading

        async def fake_process_query(*args, **kwargs):
            session_id = kwargs["session_id"]
            return {
                "sessionId": session_id,
                "confidence": 8.0,
                "responseText": [f"response-for-{session_id}"],
                "success": True,
                "execution_time_ms": 1.0,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        mock_hybrid_agent.process_query = MagicMock(side_effect=fake_process_query)

        results = {}
        errors = []

        def make_request(session_id):
            try:
                response = client.post(f"{API_PREFIX}/invocations", json={
                    "userPrompt": "test query",
                    "sessionId": session_id,
                    "context": {"userName": "u", "userType": "member", "productId": "1"},
                })
                results[session_id] = response.json()
            except Exception as e:
                errors.append(e)

        session_ids = [f"sess-{i}" for i in range(10)]
        threads = [threading.Thread(target=make_request, args=(sid,)) for sid in session_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for sid in session_ids:
            assert results[sid]["sessionId"] == sid
            assert results[sid]["responseText"] == [f"response-for-{sid}"]
```

- [ ] **Step 6: Run to verify it passes**

Run: `cd ibt-agent && python -m pytest tests/integration/test_api_routes.py::TestInvocationsRoute::test_concurrent_invocations_do_not_cross_contaminate -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `cd ibt-agent && python -m pytest tests/ -v`
Expected: All green except the one pre-existing, out-of-scope `test_assume_kendra_role_success` failure. This is the completion criterion for all of Part A.

- [ ] **Step 8: Commit**

```bash
git add ibt-agent/src/api/routes/invocations.py ibt-agent/src/api/app.py ibt-agent/tests/conftest.py ibt-agent/tests/integration/test_api_routes.py
git commit -m "feat(ibt-agent): async invocations route, executor lifespan wiring, concurrency test"
```

---

## Part B: orchestrator-agent

### Task 6: Pool-size settings and pytest-asyncio setup

**Files:**
- Modify: `orchestrator-agent/src/config/settings.py`
- Create: `orchestrator-agent/tests/unit/test_settings.py` (does not exist yet — confirm with `ls orchestrator-agent/tests/unit/*.py` before creating; if it already exists by the time this task runs, add to it instead of creating fresh)
- Modify: `orchestrator-agent/requirements.txt`
- Modify: `orchestrator-agent/pytest.ini`

**Interfaces:**
- Produces: `OrchestratorSettings.bedrock_max_pool_connections: int` (default 20), `OrchestratorSettings.sts_max_pool_connections: int` (default 10), `OrchestratorSettings.tool_http_max_connections: int` (default 20), `OrchestratorSettings.tool_http_max_keepalive_connections: int` (default 10), `OrchestratorSettings.thread_pool_max_workers: int` (default 20) — consumed by Tasks 7 and 8.

- [ ] **Step 1: Add the five new fields to `OrchestratorSettings`**

Edit `orchestrator-agent/src/config/settings.py` — insert after the `bedrock_max_retries` field (currently ending at line 82) and before the `# Extended Thinking Configuration` comment:

```python
    # Concurrency Configuration
    bedrock_max_pool_connections: int = Field(
        default=20,
        gt=0,
        description="Max HTTP connection pool size for the Bedrock boto3 client"
    )
    sts_max_pool_connections: int = Field(
        default=10,
        gt=0,
        description="Max HTTP connection pool size for the STS boto3 client (Bedrock role assumption)"
    )
    tool_http_max_connections: int = Field(
        default=20,
        gt=0,
        description="Max total connections in the shared httpx.AsyncClient pool used to call tools"
    )
    tool_http_max_keepalive_connections: int = Field(
        default=10,
        gt=0,
        description="Max keep-alive connections in the shared httpx.AsyncClient pool used to call tools"
    )
    thread_pool_max_workers: int = Field(
        default=20,
        gt=0,
        description="Max worker threads in this process's dedicated ThreadPoolExecutor"
    )
```

- [ ] **Step 2: Check whether `orchestrator-agent/tests/unit/test_settings.py` exists**

Run: `ls orchestrator-agent/tests/unit/test_settings.py 2>&1 || echo "does not exist"`

If it does not exist, create it fresh (Step 3a). If it exists, read it and add the test class/methods from Step 3a to it instead, following its existing style.

- [ ] **Step 3a: Write the failing settings tests**

Create (or add to) `orchestrator-agent/tests/unit/test_settings.py`:

```python
"""Unit tests for OrchestratorSettings."""

import pytest
from unittest.mock import patch
from src.config.settings import OrchestratorSettings, get_settings


class TestConcurrencySettings:
    """Tests for concurrency-related settings."""

    def test_concurrency_defaults(self):
        settings = OrchestratorSettings()

        assert settings.bedrock_max_pool_connections == 20
        assert settings.sts_max_pool_connections == 10
        assert settings.tool_http_max_connections == 20
        assert settings.tool_http_max_keepalive_connections == 10
        assert settings.thread_pool_max_workers == 20

    @patch.dict('os.environ', {
        'BEDROCK_MAX_POOL_CONNECTIONS': '50',
        'STS_MAX_POOL_CONNECTIONS': '15',
        'TOOL_HTTP_MAX_CONNECTIONS': '40',
        'TOOL_HTTP_MAX_KEEPALIVE_CONNECTIONS': '20',
        'THREAD_POOL_MAX_WORKERS': '30',
    })
    def test_concurrency_environment_variable_override(self):
        settings = OrchestratorSettings()

        assert settings.bedrock_max_pool_connections == 50
        assert settings.sts_max_pool_connections == 15
        assert settings.tool_http_max_connections == 40
        assert settings.tool_http_max_keepalive_connections == 20
        assert settings.thread_pool_max_workers == 30
```

- [ ] **Step 4: Run to verify it fails, then apply Step 1 and verify it passes**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_settings.py -v`
Expected: FAIL before Step 1, PASS after.

- [ ] **Step 5: Add `pytest-asyncio` to `requirements.txt`**

Edit `orchestrator-agent/requirements.txt` — insert alphabetically after `python-dotenv==1.2.1`:

```
pytest-asyncio==1.2.0
```

Run: `cd orchestrator-agent && pip install -r requirements.txt`

- [ ] **Step 6: Configure `asyncio_mode` in `pytest.ini`**

Edit `orchestrator-agent/pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::UserWarning
```

- [ ] **Step 7: Run the full suite to confirm nothing broke**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: Same pass/fail counts as before this task, plus the new settings tests passing.

- [ ] **Step 8: Commit**

```bash
git add orchestrator-agent/src/config/settings.py orchestrator-agent/tests/unit/test_settings.py orchestrator-agent/requirements.txt orchestrator-agent/pytest.ini
git commit -m "feat(orchestrator-agent): add concurrency pool-size settings and pytest-asyncio setup"
```

---

### Task 7: Dedicated ThreadPoolExecutor and shared httpx.AsyncClient modules

**Files:**
- Create: `orchestrator-agent/src/executor.py`
- Create: `orchestrator-agent/src/http_client.py`
- Create: `orchestrator-agent/tests/unit/test_executor.py`
- Create: `orchestrator-agent/tests/unit/test_http_client.py`

**Interfaces:**
- Consumes: `OrchestratorSettings.thread_pool_max_workers`, `.tool_http_max_connections`, `.tool_http_max_keepalive_connections`, `.tool_timeout` (Task 6).
- Produces: `get_executor()`, `set_as_default_executor()`, `shutdown_executor()` (identical interface to ibt-agent's, Task 2). `get_http_client() -> httpx.AsyncClient`, `close_http_client() -> Awaitable[None]` — consumed by Task 10 (`tool_node_factory.py`) and Task 11 (`app.py` lifespan).

- [ ] **Step 1: `src/executor.py` — write the failing tests**

Create `orchestrator-agent/tests/unit/test_executor.py` with the identical content to `ibt-agent/tests/unit/test_executor.py` (Task 2, Step 1), except `assert executor._max_workers == 20` still holds since both services default to 20.

- [ ] **Step 2: Run to verify failure**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_executor.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/executor.py`**

Create `orchestrator-agent/src/executor.py` with content identical to `ibt-agent/src/executor.py` (Task 2, Step 3), with the import `from src.config.settings import get_settings` unchanged (module path is the same in both services) and using `get_settings()` which returns `OrchestratorSettings` in this codebase.

- [ ] **Step 4: Run to verify it passes**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_executor.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: `src/http_client.py` — write the failing tests**

Create `orchestrator-agent/tests/unit/test_http_client.py`:

```python
"""Unit tests for the shared pooled httpx.AsyncClient module."""

import httpx
import pytest

from src.http_client import get_http_client, close_http_client


class TestGetHttpClient:
    """Tests for get_http_client singleton behavior."""

    @pytest.mark.asyncio
    async def teardown_method_async(self):
        await close_http_client()

    def teardown_method(self):
        import asyncio
        asyncio.run(close_http_client())

    def test_returns_async_client(self):
        client = get_http_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_returns_same_instance_on_repeated_calls(self):
        first = get_http_client()
        second = get_http_client()
        assert first is second

    def test_limits_configured_from_settings(self):
        client = get_http_client()
        assert client._limits.max_connections == 20
        assert client._limits.max_keepalive_connections == 10


class TestCloseHttpClient:
    """Tests for http client shutdown."""

    @pytest.mark.asyncio
    async def test_close_allows_new_instance_after(self):
        first = get_http_client()
        await close_http_client()
        second = get_http_client()
        assert first is not second

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        get_http_client()
        await close_http_client()
        await close_http_client()  # must not raise
```

- [ ] **Step 6: Run to verify failure**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_http_client.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 7: Implement `src/http_client.py`**

Create `orchestrator-agent/src/http_client.py`:

```python
"""Shared, pooled httpx.AsyncClient for calling tool APIs (e.g. ibt-agent)."""

from typing import Optional

import httpx

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Get the singleton pooled httpx.AsyncClient, creating it if needed."""
    global _http_client
    if _http_client is None:
        settings = get_settings()
        logger.info(
            f"Creating shared httpx.AsyncClient: max_connections={settings.tool_http_max_connections}, "
            f"max_keepalive_connections={settings.tool_http_max_keepalive_connections}"
        )
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=settings.tool_http_max_connections,
                max_keepalive_connections=settings.tool_http_max_keepalive_connections,
            ),
            timeout=settings.tool_timeout,
        )
    return _http_client


async def close_http_client() -> None:
    """Close the shared httpx.AsyncClient."""
    global _http_client
    if _http_client is not None:
        logger.info("Closing shared httpx.AsyncClient")
        await _http_client.aclose()
        _http_client = None
```

- [ ] **Step 8: Run to verify it passes**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_http_client.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 9: Run the full suite**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: All green (no pre-existing failures in this subproject, unlike ibt-agent).

- [ ] **Step 10: Commit**

```bash
git add orchestrator-agent/src/executor.py orchestrator-agent/src/http_client.py orchestrator-agent/tests/unit/test_executor.py orchestrator-agent/tests/unit/test_http_client.py
git commit -m "feat(orchestrator-agent): add dedicated ThreadPoolExecutor and shared pooled httpx.AsyncClient modules"
```

---

### Task 8: ChatModels — pool sizing, credential-refresh lock, async conversion

**Files:**
- Modify: `orchestrator-agent/src/llm/client.py`
- Modify: `orchestrator-agent/tests/unit/test_llm/test_client.py`
- Modify: `orchestrator-agent/tests/unit/test_llm/test_bedrock_assume_role.py`

**Interfaces:**
- Consumes: `OrchestratorSettings.bedrock_max_pool_connections`, `.sts_max_pool_connections` (Task 6).
- Produces: `ChatModels._get_bedrock_client()` becomes `async def`. `ChatModels.bedrock_model()`, `.bedrock_model_with_extended_thinking()`, `.bedrock_model_with_guardrails()`, `.apply_guardrail()`, `.get_model()` all become `async def` (same names/params, now coroutines). Consumed by Task 9.

- [ ] **Step 1: Write the failing pool-sizing tests**

Add to `orchestrator-agent/tests/unit/test_llm/test_client.py`, inside `TestChatModelsInit`:

```python
    def test_get_boto_config_uses_bedrock_max_pool_connections(self):
        from src.llm.client import ChatModels
        cm = ChatModels()
        config = cm._get_boto_config()
        assert config.max_pool_connections == cm.settings.bedrock_max_pool_connections
```

Add to `orchestrator-agent/tests/unit/test_llm/test_bedrock_assume_role.py`, inside `TestBedrockAssumeRole` (after `test_boto_config_creation`):

```python
    @patch('src.llm.client.boto3.client')
    def test_assume_bedrock_role_sts_client_uses_sts_max_pool_connections(self, mock_boto_client):
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
        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'

        self.chat_models._assume_bedrock_role()

        sts_call = mock_boto_client.call_args
        assert sts_call[0][0] == 'sts'
        assert 'config' in sts_call[1]
        assert sts_call[1]['config'].max_pool_connections == self.chat_models.settings.sts_max_pool_connections
```

- [ ] **Step 2: Run to verify these fail**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_llm/test_client.py::TestChatModelsInit::test_get_boto_config_uses_bedrock_max_pool_connections tests/unit/test_llm/test_bedrock_assume_role.py::TestBedrockAssumeRole::test_assume_bedrock_role_sts_client_uses_sts_max_pool_connections -v`
Expected: Both FAIL.

- [ ] **Step 3: Write the failing concurrency (lock) test**

Add to `orchestrator-agent/tests/unit/test_llm/test_bedrock_assume_role.py`, inside `TestBedrockAssumeRole`:

```python
    @patch('src.llm.client.boto3.client')
    def test_concurrent_get_bedrock_client_calls_assume_role_once(self, mock_boto_client):
        """Test that concurrent calls to _get_bedrock_client with expired credentials
        only trigger one assume_role call, not one per caller."""
        import asyncio
        import time as time_module

        mock_sts = Mock()
        mock_bedrock = Mock()

        def boto_client_side_effect(service_name, **kwargs):
            if service_name == 'sts':
                time_module.sleep(0.05)
                return mock_sts
            elif service_name == 'bedrock-runtime':
                return mock_bedrock
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

        self.chat_models.settings.bedrock_role_arn = 'arn:aws:iam::157539276568:role/ibt-ai-eks-bedrock-role'
        self.chat_models._client = None

        async def run_concurrent_calls():
            return await asyncio.gather(*[
                self.chat_models._get_bedrock_client() for _ in range(10)
            ])

        clients = asyncio.run(run_concurrent_calls())

        assert mock_sts.assume_role.call_count == 1
        assert all(c is mock_bedrock for c in clients)
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_llm/test_bedrock_assume_role.py::TestBedrockAssumeRole::test_concurrent_get_bedrock_client_calls_assume_role_once -v`
Expected: FAIL, `_get_bedrock_client` is currently sync and unguarded.

- [ ] **Step 5: Implement the Config, lock, and async changes in `llm/client.py`**

Edit `orchestrator-agent/src/llm/client.py` — full replacement:

```python
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import orchestrator_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChatModels:
    """Factory for AWS Bedrock chat models with timeout/retry configuration and role assumption."""

    CREDENTIALS_REFRESH_BUFFER = timedelta(minutes=5)

    def __init__(self):
        self.settings = orchestrator_settings
        self._client: Optional[boto3.client] = None
        self._assumed_credentials: Optional[dict] = None
        self._credentials_expiration: Optional[datetime] = None
        self._client_lock = threading.Lock()

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

    def _credentials_expired(self) -> bool:
        """Check if assumed credentials are expired or about to expire."""
        if self._credentials_expiration is None:
            return True
        return datetime.now(timezone.utc) >= self._credentials_expiration - self.CREDENTIALS_REFRESH_BUFFER

    def _assume_bedrock_role(self) -> dict:
        """Assume the Bedrock role and return temporary credentials."""
        if not self.settings.bedrock_role_arn:
            raise ValueError("BEDROCK_ROLE_ARN is not configured")

        try:
            logger.info(f"Assuming Bedrock role: {self.settings.bedrock_role_arn}")

            sts_client = boto3.client('sts', region_name=self.settings.aws_region, config=self._get_sts_boto_config())

            response = sts_client.assume_role(
                RoleArn=self.settings.bedrock_role_arn,
                RoleSessionName=self.settings.bedrock_session_name,
                DurationSeconds=self.settings.bedrock_role_duration
            )

            credentials = response['Credentials']
            self._credentials_expiration = credentials['Expiration']
            logger.info(f"Successfully assumed Bedrock role. Session expires at: {credentials['Expiration']}")

            return {
                'aws_access_key_id': credentials['AccessKeyId'],
                'aws_secret_access_key': credentials['SecretAccessKey'],
                'aws_session_token': credentials['SessionToken']
            }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to assume Bedrock role {self.settings.bedrock_role_arn}: {error_code}")
            raise RuntimeError(f"Role assumption failed: {error_code}") from e
        except Exception as e:
            logger.error(f"Unexpected error during role assumption: {str(e)}")
            raise RuntimeError(f"Role assumption failed: {str(e)}") from e

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

    def _get_bedrock_client_sync(self) -> boto3.client:
        """Get or create a Bedrock client with optional role assumption (thread-safe, blocking)."""
        if self._client is not None and (not self.settings.bedrock_role_arn or not self._credentials_expired()):
            return self._client

        with self._client_lock:
            if self._client is not None and (not self.settings.bedrock_role_arn or not self._credentials_expired()):
                return self._client

            if self._client is not None and self.settings.bedrock_role_arn and self._credentials_expired():
                logger.info("Assumed role credentials expired or expiring soon, refreshing...")

            return self._refresh_client_locked()

    async def _get_bedrock_client(self) -> boto3.client:
        """Get or create a Bedrock client (async, offloaded to the executor)."""
        return await asyncio.to_thread(self._get_bedrock_client_sync)

    async def bedrock_model(
        self,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatBedrockConverse:
        """Return a standard ChatBedrockConverse model."""
        resolved_model = model_id or self.settings.bedrock_model_id
        logger.info(f"Creating LLM: {resolved_model}")
        return ChatBedrockConverse(
            client=await self._get_bedrock_client(),
            model=resolved_model,
            region_name=self.settings.aws_region,
            temperature=temperature if temperature is not None else self.settings.bedrock_temperature,
            max_tokens=max_tokens or self.settings.bedrock_max_tokens,
            **kwargs,
        )

    async def bedrock_model_with_extended_thinking(
        self,
        model_id: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatBedrockConverse:
        """Return a ChatBedrockConverse model with extended thinking enabled (Claude 3.5+)."""
        resolved_model = model_id or self.settings.bedrock_model_id
        thinking_budget = budget_tokens or self.settings.extended_thinking_budget_tokens
        response_max_tokens = max_tokens or self.settings.extended_thinking_max_tokens
        logger.info(f"Creating LLM (extended thinking): {resolved_model}, budget_tokens={thinking_budget}")
        return ChatBedrockConverse(
            client=await self._get_bedrock_client(),
            model=resolved_model,
            region_name=self.settings.aws_region,
            temperature=1,  # Required for extended thinking
            max_tokens=response_max_tokens,
            additional_model_request_fields={
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            },
            **kwargs,
        )

    async def bedrock_model_with_guardrails(
        self,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatBedrockConverse:
        """Return a ChatBedrockConverse model with AWS Bedrock Guardrails enabled."""
        guardrail_id = self.settings.aws_bedrock_guardrail_id
        if not guardrail_id:
            raise ValueError("AWS_BEDROCK_GUARDRAIL_ID environment variable is not set")
        resolved_model = model_id or self.settings.bedrock_model_id
        logger.info(f"Creating LLM with guardrails: {resolved_model}, guardrail={guardrail_id}")
        return ChatBedrockConverse(
            client=await self._get_bedrock_client(),
            model=resolved_model,
            region_name=self.settings.aws_region,
            temperature=temperature if temperature is not None else self.settings.bedrock_temperature,
            max_tokens=max_tokens or self.settings.bedrock_max_tokens,
            guardrail_config={
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": self.settings.bedrock_guardrail_version,
                "trace": "enabled",
            },
            **kwargs,
        )

    async def apply_guardrail(self, text: str, source: str = "INPUT") -> dict:
        """Evaluate content against AWS Bedrock Guardrails without an LLM call."""
        guardrail_id = self.settings.aws_bedrock_guardrail_id
        if not guardrail_id:
            raise ValueError("AWS_BEDROCK_GUARDRAIL_ID environment variable is not set")
        client = await self._get_bedrock_client()
        return await asyncio.to_thread(
            client.apply_guardrail,
            guardrailIdentifier=guardrail_id,
            guardrailVersion=self.settings.bedrock_guardrail_version,
            source=source,
            content=[{"text": {"text": text}}],
        )

    async def get_model(self, model_id: Optional[str] = None, **kwargs) -> BaseChatModel:
        """Return the appropriate model based on settings (standard or extended thinking)."""
        if self.settings.extended_thinking_enabled:
            return await self.bedrock_model_with_extended_thinking(model_id=model_id, **kwargs)
        return await self.bedrock_model(model_id=model_id, **kwargs)


_chat_models: Optional[ChatModels] = None


def get_chat_models() -> ChatModels:
    """Return the singleton ChatModels instance."""
    global _chat_models
    if _chat_models is None:
        _chat_models = ChatModels()
    return _chat_models
```

- [ ] **Step 6: Update `test_client.py` for the async API**

Every test in `orchestrator-agent/tests/unit/test_llm/test_client.py` that calls `cm._get_bedrock_client()`, `cm.bedrock_model(...)`, `cm.bedrock_model_with_extended_thinking(...)`, `cm.bedrock_model_with_guardrails(...)`, or `cm.get_model(...)` needs the same transformation: add `@pytest.mark.asyncio`, change `def` to `async def`, prepend `await`. Apply this to every test in the file: `test_creates_client_on_first_call`, `test_returns_cached_client_on_second_call`, `test_client_configured_with_correct_service`, `test_bedrock_model_default_params`, `test_bedrock_model_custom_model_id`, `test_bedrock_model_custom_temperature`, `test_bedrock_model_custom_max_tokens`, `test_extended_thinking_uses_temperature_1`, `test_extended_thinking_includes_thinking_config`, `test_extended_thinking_custom_budget`, `test_extended_thinking_custom_model_id`, `test_get_model_standard_when_extended_thinking_disabled`, `test_get_model_extended_thinking_when_enabled`, `test_raises_when_guardrail_id_not_set`, `test_guardrail_config_passed_to_model`, `test_guardrail_config_contains_identifier`, `test_guardrail_config_contains_version`, `test_guardrail_config_trace_enabled`, `test_uses_same_region_as_bedrock_model`, `test_custom_model_id_is_used`, `test_default_model_id_used_when_not_specified`, `test_custom_temperature_is_used`.

Two representative full examples of the transformation (apply identically to all the rest listed above):

```python
class TestGetBedrockClient:
    """Tests for _get_bedrock_client method."""

    @pytest.mark.asyncio
    @patch("src.llm.client.boto3.client")
    async def test_creates_client_on_first_call(self, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()

        cm = ChatModels()
        cm.settings.bedrock_role_arn = None
        client = await cm._get_bedrock_client()

        mock_boto.assert_called_once()
        assert client is not None
```

```python
class TestBedrockModel:
    """Tests for bedrock_model method."""

    @pytest.mark.asyncio
    @patch("src.llm.client.boto3.client")
    @patch("src.llm.client.ChatBedrockConverse")
    async def test_bedrock_model_default_params(self, mock_cbc, mock_boto):
        from src.llm.client import ChatModels
        mock_boto.return_value = MagicMock()
        mock_cbc.return_value = MagicMock()

        cm = ChatModels()
        result = await cm.bedrock_model()

        mock_cbc.assert_called_once()
        assert result is not None
```

`TestChatModelsInit` (`test_init_sets_settings`, `test_get_boto_config_returns_config`) and `TestGetChatModels` (`test_get_chat_models_returns_same_instance`, `test_get_chat_models_creates_on_first_call`) stay synchronous — they don't call any of the now-async methods.

- [ ] **Step 7: Update `test_bedrock_assume_role.py` for the async API**

Apply the same transformation to `test_get_bedrock_client_with_role_assumption` (`client = self.chat_models._get_bedrock_client()` → `client = await self.chat_models._get_bedrock_client()`), `test_get_bedrock_client_without_role_assumption` (same), `test_client_caching` (both calls), `test_bedrock_model_with_assume_role` (`model = self.chat_models.bedrock_model()` → `model = await self.chat_models.bedrock_model()`), and `test_credentials_storage` (`self.chat_models._get_bedrock_client()` → `await self.chat_models._get_bedrock_client()`).

`test_assume_bedrock_role_success`, `test_assume_bedrock_role_no_arn`, `test_assume_bedrock_role_access_denied`, `test_assume_bedrock_role_unexpected_error`, `test_settings_configuration`, and `test_boto_config_creation` stay synchronous and unmodified — `_assume_bedrock_role` and `_get_boto_config` remain plain sync methods.

- [ ] **Step 8: Run the affected test files**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_llm/test_client.py tests/unit/test_llm/test_bedrock_assume_role.py -v`
Expected: All pass.

- [ ] **Step 9: Run the full suite**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: Failures now confined to `tests/unit/test_nodes/test_guardrail_node.py`, `tests/unit/test_nodes/test_intent_analyzer.py` (both call `get_chat_models()`/`ChatModels` methods that are now async, fixed in Task 9), plus anything downstream in `test_graph/test_orchestrator.py` and `test_graph/test_workflow.py` that indirectly exercises the graph (fixed in Task 11).

- [ ] **Step 10: Commit**

```bash
git add orchestrator-agent/src/llm/client.py orchestrator-agent/tests/unit/test_llm/test_client.py orchestrator-agent/tests/unit/test_llm/test_bedrock_assume_role.py
git commit -m "feat(orchestrator-agent): async ChatModels with pool sizing and credential-refresh lock"
```

---

### Task 9: guardrail_check_node and intent_node — async conversion

**Files:**
- Modify: `orchestrator-agent/src/graph/nodes/guardrail_node.py`
- Modify: `orchestrator-agent/src/graph/nodes/intent_analyzer.py`
- Modify: `orchestrator-agent/tests/unit/test_nodes/test_guardrail_node.py`
- Modify: `orchestrator-agent/tests/unit/test_nodes/test_intent_analyzer.py`

**Interfaces:**
- Consumes: `async def apply_guardrail(...)`, `async def get_model(...)` (Task 8).
- Produces: `guardrail_check_node(state: OrchestratorState) -> dict[str, Any]` becomes `async def`. `create_intent_node(registry) -> Callable` still returns a callable, but the returned `intent_node(state)` closure becomes `async def`. Consumed by Task 11 (`workflow.py`'s `add_node` calls accept async node callables unchanged — no signature change needed in `workflow.py` itself, since LangGraph inspects each node function at graph-build time, not at `add_node`-call time).

- [ ] **Step 1: Write the failing test for `guardrail_check_node`'s async signature**

Add to `orchestrator-agent/tests/unit/test_nodes/test_guardrail_node.py`, inside `TestGuardrailCheckNode`:

```python
    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    async def test_guardrail_check_node_is_awaitable(self, mock_get_chat, mock_state):
        """Test guardrail_check_node is a coroutine function that must be awaited."""
        import inspect
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = None
        mock_get_chat.return_value = mock_chat_models

        assert inspect.iscoroutinefunction(guardrail_check_node)

        result = await guardrail_check_node(mock_state)
        assert result["guardrail_blocked"] is False
```

(Note: `pytest.ini`'s `asyncio_mode = auto` from Task 6 means no `@pytest.mark.asyncio` decorator is strictly required on `async def test_*` functions, but this plan adds it explicitly throughout for readability and to match the existing codebase style seen in other async-capable test files — for this one test, `asyncio_mode = auto` makes it run either way; add the marker for consistency: `@pytest.mark.asyncio` above the `@patch` line.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_nodes/test_guardrail_node.py::TestGuardrailCheckNode::test_guardrail_check_node_is_awaitable -v`
Expected: FAIL — `guardrail_check_node` is currently sync.

- [ ] **Step 3: Convert `guardrail_node.py` to async**

Edit `orchestrator-agent/src/graph/nodes/guardrail_node.py` — change `guardrail_check_node` to `async def` and `await chat_models.apply_guardrail(...)`:

```python
"""Guardrail check node — runs before intent analysis to filter blocked queries."""

from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from src.schemas.state import OrchestratorState
from src.llm.client import get_chat_models
from src.exceptions import GuardrailError
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_BLOCKED_MESSAGE = "I'm sorry, but your request cannot be processed due to content policy."


def _is_hard_block(assessments: list) -> bool:
    """Return True if any assessment contains a BLOCKED action (not PII ANONYMIZED)."""
    for assessment in assessments:
        topic_policy = assessment.get("topicPolicy", {})
        for topic in topic_policy.get("topics", []):
            if topic.get("action") == "BLOCKED":
                return True

        content_policy = assessment.get("contentPolicy", {})
        for f in content_policy.get("filters", []):
            if f.get("action") == "BLOCKED":
                return True

        word_policy = assessment.get("wordPolicy", {})
        for word in word_policy.get("customWords", []):
            if word.get("action") == "BLOCKED":
                return True
        for word in word_policy.get("managedWordLists", []):
            if word.get("action") == "BLOCKED":
                return True

    return False


async def guardrail_check_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Pre-check node that runs AWS Bedrock Guardrails before intent analysis.

    - If AWS_BEDROCK_GUARDRAIL_ID is not configured: skips check (pass-through).
    - action NONE: pass through unchanged.
    - action INTERVENED + hard block: sets final_answer + guardrail_blocked=True, routes to END.
    - action INTERVENED + PII only: updates state.query with redacted text, continues to analyzer.
    """
    logger.info("→ guardrail_check")

    chat_models = get_chat_models()

    if not chat_models.settings.aws_bedrock_guardrail_id:
        logger.info("No guardrail configured — skipping guardrail check")
        return {"guardrail_blocked": False, "guardrail_action": "NONE"}

    try:
        response = await chat_models.apply_guardrail(text=state.query, source="INPUT")
    except (ClientError, BotoCoreError) as e:
        logger.error("Guardrail check failed: %s", e)
        raise GuardrailError("Guardrail check failed")
    action = response.get("action", "NONE")
    logger.info("guardrail action: %s", action)

    if action == "NONE":
        logger.info("← guardrail_check: passed")
        return {"guardrail_blocked": False, "guardrail_action": "NONE"}

    # INTERVENED — distinguish hard block vs PII masking
    assessments = response.get("assessments", [])
    outputs = response.get("outputs", [])
    output_text = outputs[0].get("text", "") if outputs else ""

    if _is_hard_block(assessments):
        logger.info("← guardrail_check: BLOCKED")
        return {
            "guardrail_blocked": True,
            "guardrail_action": "BLOCKED",
            "final_answer": [output_text or _DEFAULT_BLOCKED_MESSAGE],
        }

    # PII masking — update query with redacted version
    logger.info("← guardrail_check: PII masked, continuing to analyzer")
    return {"guardrail_blocked": False, "guardrail_action": "PII_MASKED", "query": output_text or state.query}


def guardrail_router(state: OrchestratorState) -> str:
    """Routes to END if query was blocked, otherwise to the intent analyzer. (unchanged: sync, no I/O)"""
    return "end" if state.guardrail_blocked else "analyzer"
```

- [ ] **Step 4: Update the rest of `test_guardrail_node.py`**

Apply the transformation (`@pytest.mark.asyncio`, `async def`, `await guardrail_check_node(...)`, and `mock_chat_models.apply_guardrail` needs to return an awaitable — use `AsyncMock` instead of the response being set via `mock_chat_models.apply_guardrail.return_value = ...`) to every test in `TestGuardrailCheckNode`: `test_skips_check_when_no_guardrail_configured`, `test_action_none_returns_not_blocked`, `test_apply_guardrail_called_with_query_and_input_source`, `test_intervened_blocked_topic_returns_hard_block`, `test_intervened_content_filter_returns_hard_block`, `test_intervened_blocked_empty_outputs_uses_default_message`, `test_intervened_pii_only_updates_query_with_redacted_text`, `test_intervened_pii_empty_outputs_preserves_original_query`, `test_boto_client_error_raises_guardrail_error`, `test_boto_core_error_raises_guardrail_error`, `test_missing_action_key_defaults_to_none_pass_through`.

Change the import line at the top of the file to add `AsyncMock`:

```python
from unittest.mock import MagicMock, AsyncMock, patch
```

Two representative full examples (apply identically to the rest, swapping in each test's original response fixture and assertions):

```python
    @pytest.mark.asyncio
    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    async def test_skips_check_when_no_guardrail_configured(self, mock_get_chat, mock_state):
        """When AWS_BEDROCK_GUARDRAIL_ID is not set, check is skipped and apply_guardrail not called."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = None
        mock_chat_models.apply_guardrail = AsyncMock()
        mock_get_chat.return_value = mock_chat_models

        result = await guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is False
        assert result["guardrail_action"] == "NONE"
        mock_chat_models.apply_guardrail.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.graph.nodes.guardrail_node.get_chat_models")
    async def test_action_none_returns_not_blocked(self, mock_get_chat, mock_state):
        """When action is NONE, guardrail_blocked is False with no query/final_answer."""
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "guardrail-abc123"
        mock_chat_models.apply_guardrail = AsyncMock(return_value=_make_none_response())
        mock_get_chat.return_value = mock_chat_models

        result = await guardrail_check_node(mock_state)

        assert result["guardrail_blocked"] is False
        assert result["guardrail_action"] == "NONE"
        assert "final_answer" not in result
        assert "query" not in result
```

For the two error-path tests (`test_boto_client_error_raises_guardrail_error`, `test_boto_core_error_raises_guardrail_error`), the `side_effect` pattern also works directly on `AsyncMock`:

```python
    @pytest.mark.asyncio
    @patch('src.graph.nodes.guardrail_node.get_chat_models')
    async def test_boto_client_error_raises_guardrail_error(self, mock_get_chat_models, mock_state):
        """Test that botocore ClientError raises GuardrailError."""
        from botocore.exceptions import ClientError
        from src.exceptions import GuardrailError
        mock_chat_models = MagicMock()
        mock_chat_models.settings.aws_bedrock_guardrail_id = "gr-123"
        mock_chat_models.apply_guardrail = AsyncMock(side_effect=ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "ApplyGuardrail"
        ))
        mock_get_chat_models.return_value = mock_chat_models

        with pytest.raises(GuardrailError):
            await guardrail_check_node(mock_state)
```

`TestIsHardBlock` and `TestGuardrailRouter` stay entirely synchronous and unmodified — they test `_is_hard_block` and `guardrail_router`, neither of which changed.

- [ ] **Step 5: Run `test_guardrail_node.py`**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_nodes/test_guardrail_node.py -v`
Expected: All pass.

- [ ] **Step 6: Write the failing test for `intent_node`'s async signature**

Add to `orchestrator-agent/tests/unit/test_nodes/test_intent_analyzer.py`, inside `TestIntentNode`:

```python
    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    async def test_intent_node_is_awaitable(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state
    ):
        """Test the closure returned by create_intent_node is a coroutine function."""
        import inspect
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = _make_llm_mock({
            "selected_tool": "IBTAgent",
            "confidence_score": 9.0,
            "reasoning": "test",
            "reformulated_query": "test",
        })
        mock_chat_models = MagicMock()
        mock_chat_models.get_model = AsyncMock(return_value=mock_llm)
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        assert inspect.iscoroutinefunction(intent_node)

        result = await intent_node(mock_state)
        assert result["selected_tool"].tool_name == "IBTAgent"
```

Add `AsyncMock` to the imports at the top of the file:

```python
from unittest.mock import MagicMock, AsyncMock, patch
```

Also update `_make_llm_mock` — the LLM's `with_structured_output(...).invoke(...)` becomes `.ainvoke(...)` per the design (Task 11's call-chain), so the helper's mock needs an `AsyncMock` for `.ainvoke`:

```python
def _make_llm_mock(content_dict: dict) -> MagicMock:
    """Return a mock LLM whose with_structured_output().ainvoke() returns a dict with parsed response."""
    mock_llm = MagicMock()
    mock_llm.model_id = "test-model"
    structured_mock = MagicMock()
    structured_mock.ainvoke = AsyncMock(return_value={
        "parsed": ToolSelectionOutput(**content_dict),
        "raw": MagicMock(),
        "parsing_error": None
    })
    mock_llm.with_structured_output.return_value = structured_mock
    return mock_llm
```

- [ ] **Step 7: Run to verify it fails**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_nodes/test_intent_analyzer.py::TestIntentNode::test_intent_node_is_awaitable -v`
Expected: FAIL.

- [ ] **Step 8: Convert `intent_analyzer.py` to async**

Edit `orchestrator-agent/src/graph/nodes/intent_analyzer.py`:

```python
"""Intent analyzer node - LLM-based tool selection."""

from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from src.exceptions import LLMFailureError
from src.llm.client import get_chat_models
from src.llm.prompts.intent_analyzer import build_tool_selection_prompt, build_tools_context
from src.schemas.llm import ToolSelectionOutput
from src.schemas.registry import ToolDefinition
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool
from src.utils.logging import get_logger
from src.utils.text_cleaner import clean_text

logger = get_logger(__name__)


def _extract_token_usage(raw_response: Any) -> tuple[int | None, int | None, int | None]:
    """Extract token usage from LangChain raw message metadata when available."""
    usage_metadata = getattr(raw_response, "usage_metadata", None)
    if usage_metadata is None and isinstance(raw_response, dict):
        usage_metadata = raw_response.get("usage_metadata")

    if not isinstance(usage_metadata, dict):
        return None, None, None

    return (
        usage_metadata.get("input_tokens"),
        usage_metadata.get("output_tokens"),
        usage_metadata.get("total_tokens"),
    )


def create_intent_node(registry: dict[str, ToolDefinition]):
    tools_context = build_tools_context(registry)
    system_prompt = build_tool_selection_prompt(tools_context)

    chat_models = get_chat_models()
    logger.debug(f"Available tools: {list(registry.keys())}")

    async def _get_structured_llm():
        """Get a fresh structured LLM (picks up refreshed credentials)."""
        llm = await chat_models.get_model()
        return llm.with_structured_output(ToolSelectionOutput, include_raw=True)

    async def intent_node(state: OrchestratorState) -> dict[str, Any]:
        user_query = state.query
        logger.info("-> intent_analyzer")
        logger.info(f"Input query: {user_query}")
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None

        cleaned_query = clean_text(user_query)
        if not cleaned_query:
            logger.info("<- intent_analyzer: query empty after cleaning, returning NO_TOOL")
            return {
                "selected_tool": SelectedTool(
                    tool_name="NO_TOOL",
                    confidence=0.0,
                    reasoning="Query contained no meaningful content after text cleaning",
                    reformulated_query=None,
                ),
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }

        try:
            logger.debug("Invoking LLM for intent analysis")
            structured_llm = await _get_structured_llm()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=cleaned_query),
            ]
            llm_response = await structured_llm.ainvoke(messages)
            raw_response = llm_response.get("raw")
            input_tokens, output_tokens, total_tokens = _extract_token_usage(raw_response)

            parsing_error = llm_response.get("parsing_error")
            if parsing_error is not None:
                raise OutputParserException(str(parsing_error))

            parsed: ToolSelectionOutput | None = llm_response.get("parsed")

            if parsed is None:
                logger.warning("LLM bypassed tool call (safety/content filter), returning NO_TOOL")
                return {
                    "selected_tool": SelectedTool(
                        tool_name="NO_TOOL",
                        confidence=0.0,
                        reasoning="LLM returned text instead of structured tool call",
                        reformulated_query=None,
                    ),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }

            logger.info(
                "<- intent_analyzer: tool=%s, confidence=%.1f",
                parsed.selected_tool,
                parsed.confidence_score,
            )
            logger.debug(f"Reformulated query: {parsed.reformulated_query!r}")

            return {
                "selected_tool": SelectedTool(
                    tool_name=parsed.selected_tool,
                    confidence=parsed.confidence_score,
                    reasoning=parsed.reasoning,
                    reformulated_query=parsed.reformulated_query,
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        except (ClientError, BotoCoreError) as e:
            logger.error("LLM call failed: %s", e)
            raise LLMFailureError("Intent analysis failed")
        except OutputParserException as e:
            logger.warning("LLM bypassed tool call (safety/content filter): %s", e)
            return {
                "selected_tool": SelectedTool(
                    tool_name="NO_TOOL",
                    confidence=0.0,
                    reasoning=f"LLM did not produce structured output: {e}",
                    reformulated_query=None,
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

    return intent_node
```

- [ ] **Step 9: Update the rest of `test_intent_analyzer.py`**

Apply the transformation (`async def`, `await intent_node(...)`, `mock_chat_models.get_model = AsyncMock(return_value=mock_llm)` instead of `mock_chat_models.get_model.return_value = mock_llm`, and `.invoke` → `.ainvoke` wherever `_make_llm_mock` or a manually-built `structured_mock` is used) to every remaining test in `TestIntentNode`: `test_intent_node_success`, `test_create_intent_node_get_model_failure`, `test_intent_node_invoke_failure`, `test_intent_node_no_tool_match`, `test_intent_node_with_null_context`, `test_intent_node_none_response_fallback`, `test_intent_node_output_parser_exception_fallback`, `test_intent_node_empty_after_cleaning_returns_no_tool`, `test_intent_node_low_confidence`.

For `test_create_intent_node_get_model_failure` specifically (mocks `get_model` raising directly, not via `.ainvoke`):

```python
    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    async def test_create_intent_node_get_model_failure(self, mock_get_chat, mock_registry, mock_state):
        """Test that a ClientError from get_model() propagates at invocation time."""
        from botocore.exceptions import ClientError
        mock_chat_models = MagicMock()
        mock_chat_models.get_model = AsyncMock(side_effect=ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "InvokeModel"
        ))
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        with pytest.raises(LLMFailureError):
            await intent_node(mock_state)
```

For `test_intent_node_invoke_failure` (mocks `.invoke` raising — becomes `.ainvoke` raising):

```python
    @patch('src.graph.nodes.intent_analyzer.get_chat_models')
    @patch('src.graph.nodes.intent_analyzer.build_tools_context')
    @patch('src.graph.nodes.intent_analyzer.build_tool_selection_prompt')
    async def test_intent_node_invoke_failure(
        self,
        mock_build_prompt,
        mock_build_context,
        mock_get_chat,
        mock_registry,
        mock_state,
    ):
        """Test that a ClientError from structured_llm.ainvoke() raises LLMFailureError."""
        from botocore.exceptions import ClientError
        mock_build_context.return_value = "tools context"
        mock_build_prompt.return_value = "system prompt"

        mock_llm = MagicMock()
        mock_llm.model_id = "test-model"
        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(side_effect=ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "InvokeModel"
        ))
        mock_llm.with_structured_output.return_value = structured_mock
        mock_chat_models = MagicMock()
        mock_chat_models.get_model = AsyncMock(return_value=mock_llm)
        mock_get_chat.return_value = mock_chat_models

        intent_node = create_intent_node(mock_registry)
        with pytest.raises(LLMFailureError) as exc_info:
            await intent_node(mock_state)

        assert "Intent analysis failed" in str(exc_info.value)
```

For `test_intent_node_none_response_fallback` and `test_intent_node_output_parser_exception_fallback` (both manually build `structured_mock` rather than using `_make_llm_mock`), apply the same `structured_mock.ainvoke = AsyncMock(...)` and `mock_chat_models.get_model = AsyncMock(return_value=mock_llm)` substitution shown above, keeping each test's original return value / side effect content.

For `test_intent_node_empty_after_cleaning_returns_no_tool` — no LLM call happens (short-circuits before reaching the LLM), so only `async def` + `await intent_node(state)` + `mock_chat_models.get_model.assert_not_called()` stays as-is (works identically whether `get_model` is a plain `MagicMock` attribute or unset, since it's never called).

For every other test in the list that uses `_make_llm_mock(...)` (`test_intent_node_success`, `test_intent_node_no_tool_match`, `test_intent_node_with_null_context`, `test_intent_node_low_confidence`), the fix is solely: add `async def` to the test, `mock_chat_models.get_model = AsyncMock(return_value=mock_llm)` instead of `.return_value = mock_llm`, `await intent_node(...)` instead of `intent_node(...)`, and (for `test_intent_node_success`) `structured_mock.ainvoke.assert_called_once()` / `call_args = structured_mock.ainvoke.call_args[0][0]` instead of `.invoke.assert_called_once()` / `.invoke.call_args`.

- [ ] **Step 10: Run `test_intent_analyzer.py`**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_nodes/test_intent_analyzer.py -v`
Expected: All pass.

- [ ] **Step 11: Run the full suite**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: Failures now confined to `tests/unit/test_nodes/test_tool_node_factory.py` (Task 10) and graph/orchestrator-level tests that invoke the full graph (Task 11).

- [ ] **Step 12: Commit**

```bash
git add orchestrator-agent/src/graph/nodes/guardrail_node.py orchestrator-agent/src/graph/nodes/intent_analyzer.py orchestrator-agent/tests/unit/test_nodes/test_guardrail_node.py orchestrator-agent/tests/unit/test_nodes/test_intent_analyzer.py
git commit -m "feat(orchestrator-agent): async guardrail_check_node and intent_node"
```

---

### Task 10: tool_node_factory — async conversion, shared pooled httpx.AsyncClient

**Files:**
- Modify: `orchestrator-agent/src/graph/nodes/tool_node_factory.py`
- Modify: `orchestrator-agent/tests/unit/test_nodes/test_tool_node_factory.py`

**Interfaces:**
- Consumes: `get_http_client()` (Task 7).
- Produces: `create_tool_node(tool_def) -> Callable` still returns a callable; the returned `tool_node(state)` closure becomes `async def`. `_call_tool_api(...)` becomes `async def`.

- [ ] **Step 1: Write the failing test for `tool_node`'s async signature**

Add to `orchestrator-agent/tests/unit/test_nodes/test_tool_node_factory.py`, inside `TestCreateToolNode`:

```python
    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    async def test_tool_node_is_awaitable(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test the closure returned by create_tool_node is a coroutine function."""
        import inspect

        async def fake_call_tool_api(*args, **kwargs):
            return ("Your dental benefits include preventive care.", [])
        mock_call.side_effect = fake_call_tool_api

        tool_node = create_tool_node(ibt_tool_def)
        assert inspect.iscoroutinefunction(tool_node)

        result = await tool_node(mock_state_with_tool)
        assert result["final_answer"] == "Your dental benefits include preventive care."
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_nodes/test_tool_node_factory.py::TestCreateToolNode::test_tool_node_is_awaitable -v`
Expected: FAIL — `tool_node` is currently sync.

- [ ] **Step 3: Convert `tool_node_factory.py` to async**

Edit `orchestrator-agent/src/graph/nodes/tool_node_factory.py` — full replacement:

```python
"""Factory for creating tool-specific graph nodes.

This module provides a factory function that creates dedicated LangGraph nodes
for each tool in the registry. This enables:
- Explicit tool nodes in the graph (e.g., "IBTAgent" instead of "tool_executor")
- Better traceability in LangGraph traces
- Independent testing of each tool node
- Automatic registration of new tools from YAML config
"""

from typing import Any, Callable

import httpx

from src.schemas.state import OrchestratorState
from src.schemas.tools import ToolResult, ErrorInfo
from src.schemas.registry import ToolDefinition
from src.schemas.api import AgentMetadata, MetadataItem
from src.config.settings import get_settings
from src.exceptions import ToolTimeoutError, ToolUnavailableError
from src.http_client import get_http_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_tool_node(tool_def: ToolDefinition) -> Callable[[OrchestratorState], dict[str, Any]]:
    """
    Factory function that creates a node for a specific tool.

    Args:
        tool_def: Tool definition from registry

    Returns:
        A node function that executes this specific tool
    """
    tool_name = tool_def.name
    endpoint = str(tool_def.endpoint)

    async def tool_node(state: OrchestratorState) -> dict[str, Any]:
        """Execute the tool and update state with results."""
        logger.info(f"→ {tool_name}_node")

        settings = get_settings()
        reformulated = state.selected_tool.reformulated_query if state.selected_tool else None
        if settings.use_reformulated_query and reformulated:
            effective_query = reformulated
            logger.info(f"Using reformulated query: {effective_query!r} (original: {state.query!r})")
        else:
            effective_query = state.query
            if not settings.use_reformulated_query:
                logger.info(f"USE_REFORMULATED_QUERY=False; sending original query: {effective_query!r}")
            else:
                logger.debug(f"No reformulated query; using original: {effective_query!r}")

        try:
            response_text, agent_metadata = await _call_tool_api(tool_name, endpoint, state, effective_query)

            logger.info(f"← {tool_name}_node: success")
            logger.debug(f"Response count: {len(response_text)} items")

            return {
                "tool_result": ToolResult(tool_name=tool_name, success=True, response=response_text),
                "final_answer": response_text,
                "error": None,
                "tool_metadata": agent_metadata,
            }

        except ToolTimeoutError as e:
            logger.error(f"Tool {tool_name} timeout: {e.message}")
            return {
                "error": ErrorInfo(error_type="tool_timeout", message=e.message, tool_name=tool_name),
                "tool_result": ToolResult(tool_name=tool_name, success=False, error=e.message),
            }

        except ToolUnavailableError as e:
            logger.error(f"Tool {tool_name} unavailable: {e.message}")
            return {
                "error": ErrorInfo(error_type="tool_unavailable", message=e.message, tool_name=tool_name),
                "tool_result": ToolResult(tool_name=tool_name, success=False, error=e.message),
            }

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return {
                "error": ErrorInfo(error_type="unknown", message=str(e), tool_name=tool_name),
                "tool_result": ToolResult(tool_name=tool_name, success=False, error=str(e)),
            }

    tool_node.__name__ = f"{tool_name}_node"
    tool_node.__doc__ = f"Execute {tool_name} and update state with results."

    return tool_node


async def _call_tool_api(
    tool_name: str,
    endpoint: str,
    state: OrchestratorState,
    effective_query: str = "",
) -> tuple[list[str], list]:
    """
    Call the tool's HTTP API using the shared, pooled async client.

    Args:
        tool_name: Name of the tool to call
        endpoint: HTTP endpoint URL
        state: Current orchestrator state
        effective_query: Reformulated query (or original query as fallback)

    Returns:
        Tuple of (response_text, agent_metadata_list)

    Raises:
        ToolTimeoutError: If the tool times out
        ToolUnavailableError: If the tool is unavailable
    """
    settings = get_settings()

    logger.info(f"Calling {tool_name} at {endpoint}")

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

        response = await client.post(endpoint, json=payload, headers=headers, timeout=settings.tool_timeout)
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

Note: `settings.tool_timeout` is passed per-request to `client.post(..., timeout=...)` in addition to the client's own default `timeout` (set at construction in `http_client.py`, Task 7) — this preserves the exact original per-call timeout behavior (previously `httpx.Client(timeout=settings.tool_timeout)` set it per-client-instance, since a fresh client was built per call; now the client is shared, so the timeout is passed per-request instead to keep identical behavior if `tool_timeout` is ever changed at runtime via settings reload).

- [ ] **Step 4: Update the rest of `test_tool_node_factory.py`**

`test_create_tool_node_returns_callable` and `test_tool_node_different_tools` stay synchronous and unmodified — they only check `callable(...)`, `.__name__`, `.__doc__`, object identity, none of which requires invoking the (now-async) node.

Apply the transformation (`async def`, `await tool_node(...)`, and `mock_call.return_value = (...)` → `mock_call.side_effect = async def returning the same tuple` or `mock_call = AsyncMock(return_value=...)`) to every remaining test: `test_tool_node_success`, `test_tool_node_handles_timeout_error`, `test_tool_node_handles_unavailable_error`, `test_tool_node_handles_generic_exception`, `test_tool_node_uses_agent_metadata_from_response`, `test_tool_node_empty_metadata_when_agent_returns_none`, `test_tool_node_uses_reformulated_query_as_effective_query`, `test_tool_node_falls_back_to_original_query_when_no_reformulation`.

Change the import at the top of the file:

```python
from unittest.mock import patch, MagicMock, AsyncMock
```

Representative full example (apply identically to the rest, swapping in each test's original return value / side effect / assertions):

```python
    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    async def test_tool_node_success(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test tool node execution with mocked response."""
        mock_call.return_value = ("Your dental benefits include preventive care.", [])
        mock_call.side_effect = None
        async def fake_call(*a, **kw):
            return ("Your dental benefits include preventive care.", [])
        mock_call.side_effect = fake_call
        tool_node = create_tool_node(ibt_tool_def)

        result = await tool_node(mock_state_with_tool)

        assert result["tool_result"] is not None
        assert result["tool_result"].success is True
        assert result["tool_result"].tool_name == "IBTAgent"
        assert result["final_answer"] == "Your dental benefits include preventive care."
        assert result["error"] is None
```

(Simplify the above to just `mock_call.side_effect = fake_call` — the intermediate `mock_call.return_value = ...` / `mock_call.side_effect = None` lines shown are dead-end scratch and should not appear in the final test; the clean version sets `mock_call.side_effect` to the `async def fake_call` directly, once.)

For the error-path tests (`test_tool_node_handles_timeout_error`, `test_tool_node_handles_unavailable_error`, `test_tool_node_handles_generic_exception`), `mock_call.side_effect = SomeException(...)` already works unchanged on a plain `MagicMock` when awaited via `await tool_node(...)` internally calling `await _call_tool_api(...)` — since `_call_tool_api` is patched at the module level as an attribute, not called as a bound async method, setting `.side_effect` to an exception instance still raises correctly when the mock is awaited... **except** a plain `MagicMock()` patched over an `async def` function does not itself become awaitable automatically. Use `mock_call = AsyncMock(side_effect=...)` implicitly created by `@patch(...)` targeting an async function — `unittest.mock.patch` auto-detects that `_call_tool_api` is a coroutine function and replaces it with an `AsyncMock` automatically (this is standard `unittest.mock` behavior since Python 3.8 when patching an async target with `autospec` inferred from the real object). No explicit `AsyncMock(...)` construction is needed at the `@patch(...)` decorator level; `mock_call.side_effect = ToolTimeoutError("IBTAgent", 30.0)` continues to work as before, and for the success-path tests, use `mock_call.return_value = (...)` directly (no need for an inner `async def fake_call` wrapper) since `@patch` already gives you an `AsyncMock` whose `.return_value` is unwrapped correctly on await:

```python
    @patch('src.graph.nodes.tool_node_factory._call_tool_api')
    async def test_tool_node_success(self, mock_call, ibt_tool_def, mock_state_with_tool):
        """Test tool node execution with mocked response."""
        mock_call.return_value = ("Your dental benefits include preventive care.", [])
        tool_node = create_tool_node(ibt_tool_def)

        result = await tool_node(mock_state_with_tool)

        assert result["tool_result"] is not None
        assert result["tool_result"].success is True
        assert result["tool_result"].tool_name == "IBTAgent"
        assert result["final_answer"] == "Your dental benefits include preventive care."
        assert result["error"] is None
```

This simpler form (`mock_call.return_value = (...)`, `async def`, `await tool_node(...)`) is the correct, final transformation to apply to all 8 tests listed at the start of this step — `@patch` auto-detecting the patched target is a coroutine function and supplying an `AsyncMock` means `.return_value` and `.side_effect` both work exactly as they did for the sync version, with no other change needed beyond `async def` on the test and `await` on the `tool_node(...)` call.

- [ ] **Step 5: Run `test_tool_node_factory.py`**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_nodes/test_tool_node_factory.py -v`
Expected: All pass.

- [ ] **Step 6: Run the full suite**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: Failures now confined to `tests/unit/test_graph/test_orchestrator.py` and `tests/unit/test_graph/test_workflow.py` (Task 11).

- [ ] **Step 7: Commit**

```bash
git add orchestrator-agent/src/graph/nodes/tool_node_factory.py orchestrator-agent/tests/unit/test_nodes/test_tool_node_factory.py
git commit -m "feat(orchestrator-agent): async tool_node_factory using shared pooled httpx.AsyncClient"
```

---

### Task 11: OrchestratorAgent, async route, app lifespan wiring, concurrency verification

**Files:**
- Modify: `orchestrator-agent/src/graph/orchestrator.py`
- Modify: `orchestrator-agent/src/api/routes/invocations.py`
- Modify: `orchestrator-agent/src/api/app.py`
- Modify: `orchestrator-agent/tests/unit/test_graph/test_orchestrator.py`
- Modify: `orchestrator-agent/tests/unit/test_graph/test_workflow.py`
- Modify: `orchestrator-agent/tests/integration/test_api/test_routes.py`

**Interfaces:**
- Consumes: async LangGraph nodes (Tasks 9–10), `get_executor`/`set_as_default_executor`/`shutdown_executor`, `get_http_client`/`close_http_client` (Task 7).
- Produces: `OrchestratorAgent.handle_invocation(payload, authorization=None) -> InvocationResponse` becomes `async def`. `POST /OrchestratorAgent/v2/invocations` is handled by an `async def` route.

- [ ] **Step 1: Write the failing test for `handle_invocation`'s async signature**

Add to `orchestrator-agent/tests/unit/test_graph/test_orchestrator.py`, inside `TestHandleInvocation`:

```python
    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    async def test_handle_invocation_is_awaitable(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test handle_invocation is a coroutine function that must be awaited."""
        import inspect
        from unittest.mock import AsyncMock

        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "query": "What are my dental benefits?",
            "session_id": "test-session-123",
            "registry": {},
            "context": sample_request.context.model_dump(),
            "selected_tool": None,
            "tool_result": None,
            "tool_metadata": [],
            "final_answer": ["Response"],
            "error": None
        })
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        assert inspect.iscoroutinefunction(agent.handle_invocation)

        response = await agent.handle_invocation(sample_request)
        assert response.session_id == "test-session-123"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_graph/test_orchestrator.py::TestHandleInvocation::test_handle_invocation_is_awaitable -v`
Expected: FAIL — `handle_invocation` is currently sync and calls `self.graph_app.invoke(...)`, not `.ainvoke(...)`.

- [ ] **Step 3: Convert `orchestrator.py` to async**

Edit `orchestrator-agent/src/graph/orchestrator.py` — change `handle_invocation` to `async def` and `graph_app.invoke(...)` to `await self.graph_app.ainvoke(...)`; everything else in the file (`__init__`, `_build_metadata`, `_error_response`) stays unchanged since none of it does I/O:

```python
    async def handle_invocation(self, payload: InvocationRequest, authorization: str | None = None) -> InvocationResponse:
        """Execute the workflow and return structured response."""
        start_time = time.time()
        logger.info(f"Invocation started: session={payload.session_id}")
        logger.debug(f"Query: {payload.user_prompt[:100]}{'...' if len(payload.user_prompt) > 100 else ''}")

        # Validation is handled at FastAPI layer via Pydantic validators
        # Build initial state
        state = OrchestratorState(
            query=payload.user_prompt,
            session_id=payload.session_id,
            context=payload.context,
            authorization=authorization,
        )

        # Run the graph - catch all exceptions and return graceful response
        try:
            logger.debug("Invoking LangGraph workflow")
            out_dict: dict[str, Any] = await self.graph_app.ainvoke(state.model_dump())
            out_state = OrchestratorState(**out_dict)
            logger.debug("Workflow completed successfully")
        except OrchestratorError as e:
            logger.error(f"Workflow error: {e.message}")
            return self._error_response(
                payload.session_id,
                e.message,
                start_time,
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return self._error_response(
                payload.session_id,
                "An unexpected error occurred. Please try again later.",
                start_time,
            )

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Build metadata array (preserved even on tool errors)
        metadata = self._build_metadata(out_state)

        # Determine success — state.error is the authoritative error source
        tool_failed = out_state.error is not None
        success = not tool_failed
        message = out_state.error.message if out_state.error else ""

        # Log completion
        tool_name = out_state.selected_tool.tool_name if out_state.selected_tool else 'none'
        confidence = out_state.selected_tool.confidence if out_state.selected_tool else 0.0

        logger.info(
            f"Invocation completed: session={payload.session_id}, "
            f"tool={tool_name}, "
            f"confidence={confidence:.1f}, "
            f"success={success}, "
            f"time={execution_time_ms:.0f}ms"
        )

        return InvocationResponse(
            sessionId=payload.session_id,
            responseText=out_state.final_answer if out_state.final_answer is not None else [],
            metadata=metadata,
            success=success,
            message=message or "",
            execution_time_ms=execution_time_ms,
        )
```

(`__init__`, `_build_metadata`, `_error_response` are unmodified — only the `handle_invocation` method signature and body shown above change, `invoke` → `ainvoke`.)

- [ ] **Step 4: Update the rest of `test_orchestrator.py`**

Every test in `TestHandleInvocation`, `TestGuardrailMetadata`, and any other class calling `agent.handle_invocation(...)` needs `async def` + `await`, and every `mock_graph.invoke.return_value = {...}` / `mock_graph.invoke.side_effect = ...` becomes `mock_graph.ainvoke = AsyncMock(return_value={...})` / `AsyncMock(side_effect=...)`. Add `from unittest.mock import AsyncMock` to the existing import line at the top of the file (already has `from unittest.mock import patch, MagicMock, Mock` — add `AsyncMock` to it).

Apply this to: `test_handle_invocation_success`, `test_handle_invocation_orchestrator_error`, `test_handle_invocation_unexpected_error`, `test_handle_invocation_no_selected_tool`, `test_handle_invocation_execution_time` (note: its `slow_invoke` helper becomes `async def slow_invoke(*args, **kwargs): await asyncio.sleep(0.01); return {...}`, with `import asyncio` added at the top of the file, and `mock_graph.ainvoke = AsyncMock(side_effect=slow_invoke)`), `test_handle_invocation_with_tool_metadata`, `test_guardrail_none_in_orchestrator_metadata`, `test_guardrail_blocked_in_orchestrator_metadata`. The `_make_agent` helper in `TestGuardrailMetadata` also needs updating: `mock_graph.invoke.return_value = graph_return` → `mock_graph.ainvoke = AsyncMock(return_value=graph_return)`.

`TestOrchestratorAgentInit` (`test_orchestrator_init_success`, `test_orchestrator_init_with_default_path`) stays synchronous and unmodified — `__init__` didn't change. `test_handle_invocation_empty_prompt`, `test_handle_invocation_whitespace_prompt`, `test_validation_empty_session_id`, `test_validation_empty_context_fields`, `test_validation_missing_productId` stay synchronous and unmodified — they test Pydantic validation on `InvocationRequest`/`InvocationContext` construction, never reaching `handle_invocation`. `TestErrorResponse::test_error_response_structure` stays synchronous and unmodified — `_error_response` didn't change (it's not called via the graph).

Representative full example of the transformation:

```python
    @patch('src.graph.orchestrator.ToolRegistry.from_local_yaml')
    @patch('src.graph.orchestrator.build_graph')
    async def test_handle_invocation_success(
        self,
        mock_build_graph,
        mock_from_yaml,
        sample_request,
        mock_registry_path
    ):
        """Test successful invocation handling."""
        from src.schemas.registry import ToolDefinition, ToolParameters
        tool_def = ToolDefinition(
            name="IBTAgent",
            description="Test tool",
            endpoint="https://test.internal/api/v1",
            capabilities=["testing"],
            parameters=ToolParameters(required=["userPrompt"], optional=[])
        )
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=1)
        mock_registry.list_tool_names.return_value = ["IBTAgent"]
        mock_registry.__iter__ = Mock(return_value=iter([tool_def]))
        mock_from_yaml.return_value = mock_registry

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "query": "What are my dental benefits?",
            "session_id": "test-session-123",
            "registry": {},
            "context": sample_request.context.model_dump(),
            "selected_tool": {
                "tool_name": "IBTAgent",
                "confidence": 9.0,
                "reasoning": "Benefit inquiry",
                "reformulated_query": "dental coverage benefits",
                "parameters": {}
            },
            "tool_result": None,
            "tool_metadata": [],
            "final_answer": ["Your dental benefits include..."],
            "error": None
        })
        mock_build_graph.return_value = mock_graph

        agent = OrchestratorAgent(registry_path=mock_registry_path)
        response = await agent.handle_invocation(sample_request)

        assert isinstance(response, InvocationResponse)
        assert response.session_id == "test-session-123"
        assert response.success is True
        assert response.response_text == ["Your dental benefits include..."]
        assert response.execution_time_ms >= 0.0

        assert len(response.metadata) == 1
        assert response.metadata[0].agent == "orchestrator"

        metadata_dict = {item.key: item.value for item in response.metadata[0].data}
        assert metadata_dict["confidence"] == 9.0
        assert metadata_dict["selectedTool"] == "IBTAgent"
        assert metadata_dict["reasoning"] == "Benefit inquiry"
        assert metadata_dict["reformulatedQuery"] == "dental coverage benefits"
```

- [ ] **Step 5: Run `test_orchestrator.py`**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_graph/test_orchestrator.py -v`
Expected: All pass.

- [ ] **Step 6: Update `test_workflow.py`**

`test_build_graph_can_be_invoked` (in `TestBuildGraph`) calls `graph.invoke(state.model_dump())` inside a `try/except Exception: pass` block specifically expecting it may fail on the real (unmocked) LLM call — since the graph's entry node is now `guardrail_check_node` (async) → `analyzer`/`intent_node` (async), a sync `graph.invoke(...)` call against a graph containing async nodes will raise a clear LangGraph error (async node in sync invoke) rather than the LLM-credentials error the test currently tolerates. Update it to use `graph.ainvoke(...)` via `asyncio.run(...)`, keeping the same "either completes or raises, both OK" structure:

```python
    def test_build_graph_can_be_invoked(self, tool_registry, mock_context):
        """Test that built graph can be invoked."""
        import asyncio
        from src.schemas.state import OrchestratorState

        graph = build_graph(tool_registry)

        state = OrchestratorState(
            query="Test query",
            session_id="test-123",
            context=mock_context
        )

        try:
            result = asyncio.run(graph.ainvoke(state.model_dump()))
            assert result is not None
        except Exception:
            # If it fails (expected due to LLM), that's OK for this test
            # We just want to verify the graph structure is valid
            pass
```

All other tests in `test_workflow.py` (`test_build_graph_returns_compiled_graph`, `test_build_graph_with_empty_registry`, `test_build_graph_with_single_tool`, `test_build_graph_registers_all_tool_nodes`, `test_build_graph_logs_tool_registration`, `test_build_graph_with_different_tool_counts`, `test_build_graph_creates_routes_for_tools`, `test_build_graph_includes_fallback_route`, `TestGuardrailNodeInGraph`'s three tests, `TestPostToolRouter`'s three tests, `TestGraphStructure`'s three tests) stay unmodified — none of them actually invoke the graph; they check `build_graph(...)` returns a non-`None` compiled graph object, or call the still-sync `guardrail_router`/`post_tool_router` functions directly.

- [ ] **Step 7: Run `test_workflow.py`**

Run: `cd orchestrator-agent && python -m pytest tests/unit/test_graph/test_workflow.py -v`
Expected: All pass.

- [ ] **Step 8: Convert the invocations route to async**

Edit `orchestrator-agent/src/api/routes/invocations.py`:

```python
"""Invocation routes for the orchestrator agent."""

from fastapi import APIRouter, Depends, Header
from typing import Optional

from src.schemas.api import InvocationRequest, InvocationResponse
from src.graph.orchestrator import OrchestratorAgent
from src.api.dependencies import get_orchestrator

router = APIRouter()


@router.post("/invocations", response_model=InvocationResponse)
async def invocations(
    payload: InvocationRequest,
    agent: OrchestratorAgent = Depends(get_orchestrator),
    authorization: Optional[str] = Header(None),
):
    """Process user query and route to appropriate tool."""
    return await agent.handle_invocation(payload, authorization=authorization)
```

- [ ] **Step 9: Extend the app lifespan**

Edit `orchestrator-agent/src/api/app.py`:

```python
"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import get_orchestrator
from src.api.error_handlers import register_exception_handlers
from src.api.routes import health, invocations
from src.executor import set_as_default_executor, shutdown_executor
from src.http_client import close_http_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: validate tools.yaml, wire the dedicated thread pool executor.
    Shutdown: close the shared http client and thread pool executor."""
    logger.info("Initializing Orchestrator Agent to validate configuration...")
    get_orchestrator()           # raises ToolRegistryError if tools.yaml is broken
    set_as_default_executor()
    logger.info("Orchestrator Agent initialized successfully.")
    yield
    logger.info("Shutting down Orchestrator Agent: closing http client and thread pool executor")
    await close_http_client()
    shutdown_executor()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Orchestrator Agent",
        description="Intelligent routing service using LangGraph for workflow orchestration",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # Register exception handlers (BCBSA format)
    register_exception_handlers(app)

    # Include routers
    app.include_router(health.router, prefix="/OrchestratorAgent/v2", tags=["Health"])
    app.include_router(invocations.router, prefix="/OrchestratorAgent/v2", tags=["Invocations"])

    return app
```

- [ ] **Step 10: Check the integration route tests and update the mock agent if needed**

Read `orchestrator-agent/tests/integration/test_api/test_routes.py`. If it overrides `get_orchestrator` with a `MagicMock` (following the same pattern as ibt-agent's `mock_hybrid_agent`), its `handle_invocation` mock needs the same `AsyncMock`/async-side-effect treatment as ibt-agent's Task 5 Step 3 — locate every `mock_orchestrator.handle_invocation.return_value = ...` or equivalent and convert it to an `AsyncMock` (either via `mock_orchestrator.handle_invocation = AsyncMock(return_value=...)`, or, if the mock object itself is constructed via `MagicMock(spec=OrchestratorAgent)` or similar, verify `unittest.mock`'s autospec correctly infers `handle_invocation` as async — if it does, no explicit `AsyncMock` wrapping is needed, `.return_value` continues to work). Apply whichever fix the file's actual mocking pattern requires, then re-run this file's tests until green.

- [ ] **Step 11: Run the integration test file**

Run: `cd orchestrator-agent && python -m pytest tests/integration/test_api/test_routes.py -v`
Expected: All pass after Step 10's fix.

- [ ] **Step 12: Add a cross-request concurrency isolation test**

Following ibt-agent's Task 5 Step 5 pattern, add a test to `orchestrator-agent/tests/integration/test_api/test_routes.py` that fires several concurrent requests (via `threading.Thread`, same pattern) with distinct `sessionId`s and a mocked `OrchestratorAgent.handle_invocation` (`AsyncMock` returning a response echoing the input `sessionId`), asserting no cross-contamination between concurrent responses. Match the file's existing fixture/override style for constructing the test client and overriding `get_orchestrator`.

- [ ] **Step 13: Run the full suite**

Run: `cd orchestrator-agent && python -m pytest tests/ -v`
Expected: All green. This is the completion criterion for all of Part B — no pre-existing failures exist in this subproject (unlike ibt-agent), so 100% pass is required here.

- [ ] **Step 14: Commit**

```bash
git add orchestrator-agent/src/graph/orchestrator.py orchestrator-agent/src/api/routes/invocations.py orchestrator-agent/src/api/app.py orchestrator-agent/tests/unit/test_graph/test_orchestrator.py orchestrator-agent/tests/unit/test_graph/test_workflow.py orchestrator-agent/tests/integration/test_api/test_routes.py
git commit -m "feat(orchestrator-agent): async OrchestratorAgent, async invocations route, executor/http-client lifespan wiring, concurrency test"
```

---

## Final verification (both subprojects)

- [ ] Run `cd ibt-agent && python -m pytest tests/ -v` — all green except `test_assume_kendra_role_success` (pre-existing, out of scope).
- [ ] Run `cd orchestrator-agent && python -m pytest tests/ -v` — all green.
- [ ] Manually sanity-check both services still start: `cd ibt-agent && python -m src.main` and, separately, `cd orchestrator-agent && python -m src.main` (Ctrl+C after confirming no startup exceptions — this exercises the new `_lifespan` wiring in both, including `set_as_default_executor()`, `get_orchestrator()`'s tools.yaml validation, and clean shutdown via `shutdown_executor()`/`close_http_client()`).
