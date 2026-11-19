# Experiment Scripts

This folder contains Python experiment scripts that validate and analyze various aspects of the Aegis Platform implementation.

## RAG Safety Pipeline Example

**File:** `rag_safety_pipeline_example.py`

Runs all seven RAG safety layers (guardrails, query enhancement, access control, retrieval, reranking, citations with hallucination detection, PII masking) against an in-memory corpus. No Qdrant or embedding provider needed; query enhancement runs only when `OPENAI_API_KEY` is set.

```powershell
uv run python scripts/rag_safety_pipeline_example.py
```

See [docs/GUIDES/RAG-SAFETY-USAGE.md](../docs/GUIDES/RAG-SAFETY-USAGE.md).

## Day 10: Embedding Provider & Vector Store Experiment

**File:** `day10_embedding_vectorstore_experiment.py`

**Purpose:** Validate OpenAI embedding provider and Qdrant vector store implementations against performance, reliability, and quality baselines.

### Features Tested

1. **Embedding Batching Efficiency**
   - Single chunk embedding latency
   - Batch embedding latency (10, 50, 100 chunks)
   - Latency per embedding metric
   - Batching efficiency measurement

2. **Vector Search Latency**
   - Collection creation
   - Vector insertion (100 vectors)
   - Search latency (top 10 results)
   - Metadata filtering performance

3. **Error Handling & Retry Logic**
   - Health check functionality
   - Configuration validation
   - API error handling
   - Retry mechanism verification

4. **Configuration-Driven Behavior**
   - Provider registry discovery
   - Vector store registry discovery
   - Configuration propagation
   - Multi-provider support

5. **Type Safety**
   - mypy --strict compliance
   - Type annotation coverage
   - Protocol compliance

6. **Code Quality**
   - ruff linting
   - Import ordering
   - Docstring completeness
   - Code style compliance

7. **Performance Baselines**
   - Single chunk embedding (<100ms target)
   - Batch efficiency (throughput)
   - Search latency (<50ms target on 1M vectors)

### Usage

```bash
# Default: Use mocked API calls (no credentials needed)
uv run scripts/day10_embedding_vectorstore_experiment.py

# Verbose output
uv run scripts/day10_embedding_vectorstore_experiment.py --verbose

# With live OpenAI API (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... uv run scripts/day10_embedding_vectorstore_experiment.py --live
```

### Output

The script produces:

1. **Console Output:** Real-time test execution with pass/fail status
2. **Summary Report:** Final results for all tested components
3. **Exit Code:** 0 for success, 1 for failures

### Sample Output

```
╔══════════════════════════════════════════════════════════╗
║  Day 10: Embedding Provider & Vector Store Experiment  ║
╚══════════════════════════════════════════════════════════╝

📊 Testing Embedding Batching Efficiency
============================================================
  Batch   1:   45.2ms (  45.2ms per embedding)
  Batch  10:  120.5ms (  12.1ms per embedding)
  Batch  50:  380.2ms (   7.6ms per embedding)
  Batch 100:  720.1ms (   7.2ms per embedding)

🔍 Testing Vector Search Latency
============================================================
  Created collection: day10_experiment
  Inserted 100 vectors: 125.3ms
  Search (top 10): 8.7ms (found 10 results)

✅ Type Safety (mypy --strict): PASS
✅ Code Quality (ruff): PASS
✅ Configuration: Available providers found
✅ Performance: Baselines measured
```

### Requirements

- Python 3.9+
- OpenAI Python SDK (openai)
- Qdrant Python Client (qdrant-client)
- Aegis Platform RAG module

### Notes

- With mocked API calls (default), latencies represent internal processing only
- With live API calls (--live flag), latencies include network latency
- Qdrant must be running (Docker or cloud) for vector search tests
- No real data is persisted; collections are cleaned up after tests

### Related Documentation

- [Day 10 Embedding & Vector Store Analysis](../experiments/day10-embedding-vectorstore-analysis.md)
- [RFC-009: RAG Architecture](../docs/RFC/AEGIS-RFC-009-RAG-Architecture.md)
- [SRS-001: RAG Requirements](../docs/SRS/AEGIS-SRS-001.md)

---

**Last Updated:** November 16, 2026
**Status:** ✅ Production Ready
