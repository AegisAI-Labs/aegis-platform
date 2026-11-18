# ADR-010: RAG Service Provider Wiring Strategy

- **Status:** Accepted
- **Owner:** AEGIS Platform
- **Date:** 2026-11-17
- **Scope:** RAG Service dependency injection and provider initialization pattern

---

## 1. Problem Statement

The RAG Service needs to:

1. Instantiate embedding providers (OpenAI, NVIDIA) based on configuration
2. Instantiate vector stores (Qdrant) based on configuration
3. Inject these dependencies into the FastAPI application
4. Ensure providers are initialized once and reused across requests (expensive initialization)
5. Support runtime provider switching via environment variables
6. Enable testing with mocked providers without side effects

**Question:** How should we wire provider dependencies into the RAG Service and FastAPI endpoints?

**Options:**

### Option A: Singleton Pattern with @lru_cache
```python
@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    provider = get_configured_provider()
    vector_store = get_configured_vector_store()
    return RAGService(provider=provider, vector_store=vector_store)

@app.get("/api/rag/health")
async def health_check(service: RAGService = Depends(get_rag_service)):
    ...
```

**Pros:**
- Simple, single point of initialization
- Python built-in `@lru_cache` - no external dependencies
- Lazy initialization (only when first used)
- FastAPI Depends pattern integrates naturally
- Automatic singleton per cache key

**Cons:**
- Cache is per-process (not ideal in distributed settings)
- Clearing cache requires function access
- pytest fixtures can't easily override cache behavior

### Option B: Dependency Factory (Class-Based)
```python
class ServiceContainer:
    _instances: Dict[str, Any] = {}

    @classmethod
    def get_rag_service(cls) -> RAGService:
        if 'rag_service' not in cls._instances:
            provider = get_configured_provider()
            vector_store = get_configured_vector_store()
            cls._instances['rag_service'] = RAGService(provider, vector_store)
        return cls._instances['rag_service']

@app.get("/api/rag/health")
async def health_check(service: RAGService = Depends(ServiceContainer.get_rag_service)):
    ...
```

**Pros:**
- Explicit state management
- Can reset/clear instances in tests
- More testable with lifecycle control

**Cons:**
- More boilerplate code
- Manual cache management
- Thread-safety concerns with mutable dict

### Option C: FastAPI Lifespan with Context
```python
from contextlib import asynccontextmanager

rag_service: Optional[RAGService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_service
    provider = get_configured_provider()
    vector_store = get_configured_vector_store()
    rag_service = RAGService(provider, vector_store)

    yield

    # Shutdown
    await rag_service.cleanup()

app = FastAPI(lifespan=lifespan)

@app.get("/api/rag/health")
async def health_check():
    return await rag_service.health_check()
```

**Pros:**
- Clear lifecycle management
- Explicit startup/shutdown hooks
- Works well with async cleanup
- Application-level vs. request-level

**Cons:**
- Global state (requires care in testing)
- Less flexible than Depends pattern
- Breaks FastAPI dependency injection paradigm

---

## 2. Decision

**Selected: Option A (Singleton with @lru_cache)**

---

## 3. Rationale

### Why Option A

1. **FastAPI Natural:** FastAPI's `Depends` pattern is designed for this. `@lru_cache` integrates seamlessly.

2. **Minimal Boilerplate:** Single decorator, no container classes or global state management code.

3. **Provider Abstraction Respect:** Embedding providers are initialized lazily only when first needed, respecting factory pattern from Phase 1 architecture.

4. **Testability with Fixtures:** Pytest can patch `get_rag_service` function directly:
   ```python
   @pytest.fixture
   def mock_rag_service(monkeypatch):
       mock_service = MagicMock(RAGService)
       monkeypatch.setattr("app.api.rag.get_rag_service", lambda: mock_service)
       return mock_service
   ```

5. **Explicit Dependencies:** FastAPI `Depends` makes it clear what each endpoint requires. Inspection of route signature shows RAGService dependency.

6. **Configuration-Driven:** Provider selection still happens at initialization time based on `RAG_EMBEDDING_PROVIDER` env var. Swapping providers requires app restart (acceptable pattern).

### Why Not Option B

- More code for same behavior
- Manual cache invalidation error-prone
- Thread-safety concerns with mutable dict
- No real advantage for our use case

### Why Not Option C

- Loses FastAPI Depends paradigm benefits
- Global state harder to test
- Requires explicit cleanup logic
- Lifespan is for infrastructure setup (like database connections), not business-logic services

---

## 4. Implementation

### Core Dependency Function

```python
# src/aegis_platform/api/rag.py

from functools import lru_cache
from aegis_platform.rag.service import RAGService
from aegis_platform.rag.provider_factory import (
    get_configured_embedding_provider,
    get_configured_vector_store,
)

@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """
    Singleton RAG Service with configured providers.

    Configuration sourced from environment variables:
    - RAG_EMBEDDING_PROVIDER: "openai" or "nvidia" (default: "openai")
    - RAG_VECTOR_STORE_URL: Qdrant endpoint
    - RAG_VECTOR_STORE_API_KEY: Qdrant API key

    Returns:
        RAGService with configured providers

    Raises:
        ValueError: If configuration invalid or providers unavailable
    """
    embedding_provider = get_configured_embedding_provider()
    vector_store = get_configured_vector_store()
    return RAGService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
```

### Endpoint Usage

```python
@router.post("/health", response_model=HealthCheckResponse)
async def health_check(service: RAGService = Depends(get_rag_service)):
    """Check embedding provider and vector store connectivity"""
    return await service.health_check()

@router.post("/embed", response_model=EmbedResponse)
async def embed_text(
    request: EmbedRequest,
    service: RAGService = Depends(get_rag_service),
):
    """Generate embeddings for text"""
    embedding = await service.embed_text(request.text)
    return EmbedResponse(embedding=embedding.tolist())
```

### Testing Pattern

```python
# tests/integration/rag/test_rag_endpoints.py

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def mock_rag_service():
    """Mock RAG Service for testing endpoints"""
    mock = MagicMock()
    mock.health_check.return_value = {
        "status": "healthy",
        "embedding_provider": "openai",
        "vector_store": "qdrant",
    }
    return mock

@pytest.fixture
def test_client(mock_rag_service):
    """TestClient with mocked RAG Service"""
    with patch("app.api.rag.get_rag_service", return_value=mock_rag_service):
        yield TestClient(app)

def test_health_endpoint(test_client):
    response = test_client.post("/api/rag/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Configuration Handling

```python
# src/aegis_platform/rag/provider_factory.py

import os
from aegis_platform.rag.embeddings import (
    OpenAIEmbeddingProvider,
    NVIDIAEmbeddingProvider,
)
from aegis_platform.rag.vector_stores import QdrantVectorStore

def get_configured_embedding_provider() -> EmbeddingProvider:
    """Load embedding provider from configuration"""
    provider_name = os.getenv("RAG_EMBEDDING_PROVIDER", "openai").lower()

    if provider_name == "openai":
        return OpenAIEmbeddingProvider.from_env()
    elif provider_name == "nvidia":
        return NVIDIAEmbeddingProvider.from_env()
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_name}. "
            f"Supported: 'openai', 'nvidia'"
        )

def get_configured_vector_store() -> VectorStore:
    """Load vector store from configuration"""
    # Currently only Qdrant supported
    return QdrantVectorStore.from_env()
```

---

## 5. Consequences

### Positive

1. **Simplicity:** Minimal code, maximal clarity. One decorator does the job.
2. **FastAPI Idiomatic:** Follows FastAPI patterns that developers expect.
3. **Provider Abstraction:** Respects Phase 1 factory pattern and protocols.
4. **Testability:** Easy to mock with pytest fixtures and monkeypatch.
5. **Performance:** Lazy initialization, single instance per process.
6. **Configuration-Driven:** Provider switching still possible via env vars + app restart.

### Negative

1. **Cache Lifetime:** Cache persists for process lifetime. If provider needs updating without restart, requires app restart or cache clearing.
2. **Distributed Systems:** Each process/pod gets own cache. Not an issue since providers are lightweight after initialization.
3. **Testing Isolation:** Must ensure cache is cleared/mocked between tests to avoid cross-test contamination.

### Mitigation

- Clear lru_cache in test teardown if needed:
  ```python
  @pytest.fixture(autouse=True)
  def clear_cache():
      yield
      get_rag_service.cache_clear()  # Clear after test
  ```

---

## 6. Alternatives Reconsidered

### Greenfield/FastAPI AppContext (Future)
If FastAPI adds application context support similar to Flask, could use:
```python
from fastapi.context import g
g.rag_service = RAGService(...)  # Hypothetical
```
But this doesn't exist in current FastAPI, so premature.

### Dependency Override for Testing
Could use FastAPI's `app.dependency_overrides` for testing:
```python
app.dependency_overrides[get_rag_service] = lambda: mock_service
```
But less explicit than monkeypatch. Using monkeypatch is more Pythonic and pytest-idiomatic.

---

## 7. Compliance & Standards

- ✅ Aligns with FastAPI Depends pattern (FastAPI documentation: Dependency Injection)
- ✅ Uses Python standard library (@lru_cache, no external dependencies)
- ✅ Respects protocol-based abstractions from ADR-006
- ✅ Configuration-driven per ADR-004 (Multi-provider pattern)
- ✅ Supports provider-agnostic architecture from RFC-009

---

## 8. References

- **RFC-009:** RAG Architecture (Phase 2b RAG Service Layer)
- **ADR-006:** Multi-provider default selection strategy
- **ADR-004:** LLM provider abstraction pattern
- FastAPI Documentation: Dependency Injection & Dependency Overrides
- Python Documentation: functools.lru_cache

---

## 9. Future Considerations

1. **Distributed Caching:** If moving to Kubernetes with multiple pods, consider Redis-backed provider cache.
2. **Hot-Reload:** If hot-reloading providers becomes requirement, could add explicit cache invalidation endpoint.
3. **Async Initialization:** If providers require async init, could use `AsyncContextManager` pattern instead.
4. **Multi-Tenancy:** If multi-tenant scenarios emerge, could make cache key include tenant ID.

---

**Author:** AEGIS Platform Team
**Last Updated:** 2026-11-17
**Status:** Accepted
