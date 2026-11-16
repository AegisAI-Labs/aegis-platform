# AEGIS-ADR-009

# Architecture Decision Record - RAG Foundation Chunking Architecture

**Status:** Accepted

**Date:** 2026-11-15

**Owner:** Kader Beevi

**Related Documents:**
- SRS: AEGIS-SRS-001 (RAG Foundation section)
- RFC: AEGIS-RFC-009 (RAG Architecture)
- EDI: AEGIS-EDI-001

---

## Context

The Aegis platform requires a Retrieval-Augmented Generation (RAG) capability to augment LLM reasoning with retrieved documents. The foundation phase must establish:

1. Flexible, production-ready text chunking for various document types
2. Abstract interfaces for embeddings and vector storage (provider-agnostic)
3. Configuration-driven dependency injection for strategy selection
4. Extensible architecture for future chunking strategies and vector store implementations

**Challenge:** Text chunking is domain-specific (code vs. markdown vs. general text requires different approaches), and vector storage providers vary significantly (Qdrant, Weaviate, Pinecone, Chroma, etc.). A monolithic implementation would lock us into specific choices.

**Constraint:** Must be production-ready with no stub implementations or future TODOs in critical paths.

---

## Options Considered

### Option A: Custom-Only Chunking

**Approach:** Implement 3 lightweight custom chunking strategies (FixedSize, Recursive, Token-based).

**Pros:**
- Full control over implementation
- No external dependencies
- Lightweight and fast

**Cons:**
- Custom implementations less battle-tested than open-source alternatives
- Maintenance burden on edge cases (Unicode, special characters)
- Duplicates work already done in LangChain
- Limited strategy variety (3 strategies insufficient for all use cases)

**Decision:** Rejected - insufficient strategy variety and maintenance risk

---

### Option B: LangChain-Only Chunking

**Approach:** Use only LangChain's text splitters (RecursiveCharacterTextSplitter, MarkdownTextSplitter, etc.).

**Pros:**
- Battle-tested implementations
- Lower maintenance burden
- Rich strategy ecosystem
- Direct integration with LangChain ecosystem

**Cons:**
- Adds external dependency (though LangChain already transitive via langgraph)
- Less control over implementation
- Possible over-engineering for simple cases

**Decision:** Accepted - Production readiness outweighs concerns. LangChain is battle-tested in production systems.

---

### Option C: Hybrid (Custom + LangChain)

**Approach:** Keep 3 custom strategies + add 2 LangChain wrappers. Users choose strategy via config.

**Pros:**
- Best of both: custom for simple cases, LangChain for complex
- Maximum flexibility
- Enables A/B testing strategies

**Cons:**
- Maintenance burden increases
- Test coverage becomes complex
- Potential confusion on which strategy to use when

**Decision:** Initially explored but rejected after user requested simplification. Hybrid adds complexity without proportional benefit.

---

## Decision

**Implement RAG Foundation using LangChain-only chunking strategies with provider-agnostic architecture.**

### Rationale

1. **Production Readiness**: LangChain's text splitters are battle-tested in production systems. No need to reinvent.

2. **Extensibility**: Abstract protocol-based architecture allows adding new strategies without modification to existing code or tests.

3. **Vendor Neutrality**: Abstract `EmbeddingProvider` and `VectorStore` protocols ensure we don't lock into specific providers. Implementations can be swapped at runtime.

4. **Configuration-Driven**: `ChunkingConfig` enables strategy selection via environment variables without code changes.

5. **Simplicity**: Two well-designed strategies (Recursive, Markdown) cover 90% of use cases. Future strategies can be added incrementally.

### Implementation Details

#### Chunking Architecture

```
ChunkingStrategy (Abstract Protocol)
├── LangChainRecursiveChunker
│   └── Wraps: LangChain RecursiveCharacterTextSplitter
├── LangChainMarkdownChunker
│   └── Wraps: LangChain MarkdownTextSplitter
└── Future: SemanticChunker, SlidingWindowChunker, etc.

ChunkingConfig (Configuration Object)
└── strategy: str (name) + chunk_size, overlap, min/max sizes

create_chunker() (Factory)
└── Instantiates correct strategy based on config
```

#### Protocol-Based Storage

```
EmbeddingProvider (Abstract Protocol)
├── No concrete implementations (Day 2)
└── Future: OpenAIEmbedder, ClaudeEmbedder, LocalEmbedder

VectorStore (Abstract Protocol)
├── No concrete implementations (Day 2)
└── Future: QdrantStore, WeaviateStore, PineconeStore
```

#### Factory Pattern

```python
# Register strategies
CHUNKING_STRATEGIES = {
    "recursive": LangChainRecursiveChunker,
    "markdown": LangChainMarkdownChunker,
}

# Create based on config
def create_chunker(config: ChunkingConfig) -> ChunkingStrategy:
    strategy_class = CHUNKING_STRATEGIES[config.strategy]
    return strategy_class(chunk_size=config.chunk_size, ...)
```

---

## Trade-offs

### Advantages

1. **Production Quality**: Using battle-tested LangChain implementations reduces bugs and edge cases.

2. **Maintainability**: One less codebase to maintain. Security updates handled by LangChain.

3. **Extensibility**: Protocol-based design allows infinite strategies without core changes.

4. **Runtime Flexibility**: Configuration enables A/B testing and adaptation without code changes.

5. **Type Safety**: Full type annotations with mypy validation. No Any or unsafe casts.

6. **Testing**: 24 comprehensive tests covering initialization, validation, chunking, configuration, metadata, edge cases, and integration scenarios.

7. **Documentation**: Clear README with usage examples, API docs, and future roadmap.

### Disadvantages

1. **Dependency**: Adds `langchain-text-splitters` dependency (though LangChain is already transitive).

2. **Less Control**: Can't optimize LangChain implementation for specific edge cases.

3. **Potential Over-engineering**: For very simple fixed-size chunking, LangChain may be overkill.

4. **Learning Curve**: Developers must understand LangChain's text splitter API.

---

## Consequences

### Immediate (Implemented)

1. ✅ Two production-ready chunking strategies available
2. ✅ Abstract protocols for embeddings and vector storage
3. ✅ Configuration-driven dependency injection working
4. ✅ 24 unit tests, all passing
5. ✅ Type checking and linting clean
6. ✅ Documentation complete

### Short Term (Day 3)

1. Implement concrete `EmbeddingProvider` classes (OpenAI, Claude)
2. Implement concrete `VectorStore` classes (Qdrant)
3. Build integration tests between chunking and embeddings
4. Create end-to-end RAG pipeline smoke tests

### Medium Term (Day 4+)

1. Add document loader for various file formats
2. Implement batch indexing pipeline
3. Explore advanced strategies (semantic chunking, sliding window)
4. Add performance optimization and caching

### Long Term (Production)

1. RAG becomes core agent capability for context-aware reasoning
2. Support for multiple vector stores and embedding providers
3. Retrieval ranking and reranking strategies
4. Evaluation metrics for chunk quality and retrieval accuracy

---

## Risk Mitigation

### Dependency Risk
- **Mitigation**: LangChain-text-splitters is small, stable package. Used in production at scale.

### Change Risk
- **Mitigation**: Protocol-based architecture allows swapping implementations if LangChain breaks.

### Complexity Risk
- **Mitigation**: Clear separation of concerns. Each component (chunking, embeddings, storage) is independent.

### Future Extensibility Risk
- **Mitigation**: Protocol-based design proven extensible. Can add strategies without core changes.

---

## Validation

### Testing

- 24 unit tests covering all acceptance criteria
- All tests passing
- Type checking clean (mypy)
- Linting clean (ruff)

### Smoke Tests

```python
# Recursive strategy works
config = ChunkingConfig(strategy="recursive", chunk_size=512)
chunker = create_chunker(config)
chunks = chunker.chunk("Text...", "doc-1")
assert len(chunks) > 0

# Markdown strategy works
config = ChunkingConfig(strategy="markdown", chunk_size=1024)
chunker = create_chunker(config)
chunks = chunker.chunk("# Title\n\nContent...", "doc-2")
assert len(chunks) > 0

# Factory errors on invalid strategy
with pytest.raises(ValueError):
    create_chunker(ChunkingConfig(strategy="invalid"))
```

### Documentation

- README with architecture, usage examples, testing instructions
- SRS with functional and non-functional requirements
- Inline code documentation with docstrings
- Clear error messages for configuration errors

---

## Alternative Futures

### If Requirement Changes

1. **Need custom strategy**: Implement `ChunkingStrategy` protocol, register in factory
2. **Need different vector store**: Implement `VectorStore` protocol, update config
3. **Need embedding provider**: Implement `EmbeddingProvider` protocol, update factory

None of these require core architecture changes.

---

## Alignment with Aegis Principles

✅ **Documentation First:** SRS-001 v1.7 (FR-013, NFR-008) defines all requirements before implementation

✅ **Extensible Design:** Protocol-based architecture proven extensible

✅ **Production Ready:** Battle-tested LangChain implementations, comprehensive testing

✅ **Type Safe:** Full type annotations, mypy clean

✅ **Testable:** 24 tests covering happy path, edge cases, integration

✅ **Maintainable:** Clear separation of concerns, single responsibility

---

## Related Decisions

- ADR-001: Adopted FastAPI for API foundation (similar philosophy: proven, production-ready)
- ADR-003: Adopted LangGraph for agent orchestration (similar: battle-tested framework)
- ADR-004: LLM provider abstraction (similar: protocol-based provider agnosticism)

This ADR extends the pattern: use battle-tested frameworks + abstract interfaces = production-ready + extensible.

---

## Approval Sign-Off

- **Decision Made By:** Architecture Review
- **Approval Date:** 2026-11-15
- **Implementation Status:** ✅ COMPLETE
- **Next Review:** Day 3 (Embedding Provider + Vector Store implementations)

---

## References

- [LangChain Text Splitters Documentation](https://python.langchain.com/docs/modules/data_connection/document_loaders/split_documents)
- [Python Protocols (PEP 544)](https://www.python.org/dev/peps/pep-0544/)
- [Factory Pattern in Python](https://refactoring.guru/design-patterns/factory-method)
- [RAG Survey (arxiv:2312.10997)](https://arxiv.org/abs/2312.10997)
