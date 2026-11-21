"""RAGService end to end with the real Qdrant client (local ":memory:" mode).

Before Day 16 the RAG API tests replaced RAGService with a fake, so defects in the real
embed -> Qdrant path (wrong embedder signature, removed ``client.search``, unstable IDs,
missing collection, ``/rag/collections`` schema) were invisible. Only the embedding model
is faked here: a deterministic bag-of-words embedder that satisfies EmbeddingProvider.
"""

import hashlib
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aegis_platform.api import rag as rag_api
from aegis_platform.rag.access_control import (
    AccessLevel,
    DocumentAccessControl,
    DocumentMetadata,
    Role,
    UserContext,
)
from aegis_platform.rag.embeddings.base import EmbeddingProvider
from aegis_platform.rag.models import (
    Document,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from aegis_platform.rag.service import DEFAULT_COLLECTION, RAGService
from aegis_platform.rag.storage.qdrant import (
    IN_MEMORY_URL,
    QdrantVectorStore,
    QdrantVectorStoreConfig,
)

DIM = 64


class HashingEmbedder:
    """Deterministic bag-of-words embedding: similar words -> similar vectors."""

    model = "hashing-test-embedder"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.requests: list[EmbeddingRequest] = []

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIM, dtype=np.float32)
        for word in text.lower().split():
            bucket = int(hashlib.sha256(word.strip(".,?!").encode()).hexdigest(), 16) % DIM
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector + 1.0 / np.sqrt(DIM)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        vectors = await self.embed_batch(request.texts)
        return EmbeddingResponse(
            embeddings=vectors,
            model=self.model,
            provider="test",
            dimension=DIM,
            usage=EmbeddingUsage(input_tokens=sum(len(t.split()) for t in request.texts)),
        )

    async def embed_text(self, text: str) -> np.ndarray:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self.batch_sizes.append(len(texts))
        return [self._vector(text) for text in texts]

    async def get_dimension(self) -> int:
        return DIM

    async def health_check(self) -> bool:
        return True


class UnhealthyStore(QdrantVectorStore):
    async def health_check(self) -> bool:
        return False


def make_service(store_cls: type[QdrantVectorStore] = QdrantVectorStore) -> RAGService:
    store = store_cls(QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM))
    return RAGService(embedding_provider=HashingEmbedder(), vector_store=store)


def doc(doc_id: str, content: str) -> Document:
    return Document(id=doc_id, title=doc_id, content=content, source="test")


DOCS = [
    doc("kb-1", "Encryption keys rotate every ninety days."),
    doc("kb-2", "The cafeteria opens at eight in the morning."),
    doc("kb-3", "Expense reports are due on the fifth of each month."),
]


def test_fake_embedder_satisfies_protocol() -> None:
    assert isinstance(HashingEmbedder(), EmbeddingProvider)


@pytest.mark.asyncio
async def test_ingest_then_search_returns_the_relevant_chunk() -> None:
    service = make_service()

    ingested = await service.ingest_documents(DOCS)
    result = await service.search_similar("when do encryption keys rotate", top_k=2)

    assert ingested["documents_stored"] == 3
    assert ingested["chunks_created"] >= 3
    top = result["results"][0]
    assert top["metadata"]["document_id"] == "kb-1"
    assert "Encryption keys" in top["text"]
    assert 0 < top["similarity_score"] <= 1


@pytest.mark.asyncio
async def test_search_before_ingestion_returns_no_results() -> None:
    result = await make_service().search_similar("anything", top_k=3)

    assert result["results"] == []


@pytest.mark.asyncio
async def test_reingest_does_not_duplicate_chunks() -> None:
    service = make_service()

    await service.ingest_documents(DOCS)
    await service.ingest_documents(DOCS)

    [collection] = (await service.list_collections())["collections"]
    first = await service.ingest_documents(DOCS[:1])
    assert collection["count"] == first["chunks_created"] + 2  # 3 docs, 1 chunk each


@pytest.mark.asyncio
async def test_ingest_more_than_100_chunks() -> None:
    service = make_service()
    documents = [doc(f"doc-{i}", f"Policy number {i} covers topic {i}.") for i in range(150)]

    result = await service.ingest_documents(documents)

    assert result["chunks_created"] == 150
    [collection] = (await service.list_collections())["collections"]
    assert collection["count"] == 150


@pytest.mark.asyncio
async def test_list_collections_matches_api_schema() -> None:
    service = make_service()
    await service.ingest_documents(DOCS)

    result = await service.list_collections()

    assert result["collections"] == [
        {"name": DEFAULT_COLLECTION, "vector_size": DIM, "count": 3, "distance": "cosine"}
    ]


@pytest.mark.asyncio
async def test_health_reports_real_model_name() -> None:
    health = await make_service().health_check()

    assert health["status"] == "healthy"
    assert health["embedding_model"] == "hashing-test-embedder"
    assert health["vector_store"] == "QdrantVectorStore"


@pytest.mark.asyncio
async def test_health_fails_when_vector_store_is_down() -> None:
    with pytest.raises(RuntimeError, match="Vector store is unreachable"):
        await make_service(UnhealthyStore).health_check()


@pytest.mark.asyncio
async def test_embedding_count_mismatch_is_an_error() -> None:
    class ShortEmbedder(HashingEmbedder):
        async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            return (await super().embed_batch(texts))[:-1]

    store = QdrantVectorStore(QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM))
    service = RAGService(embedding_provider=ShortEmbedder(), vector_store=store)

    with pytest.raises(RuntimeError, match="returned 2 vectors for 3 chunks"):
        await service.ingest_documents(DOCS)


# API level: the real service behind the /rag routes --------------------------------------


@pytest.fixture
def real_api(client: TestClient) -> Iterator[TestClient]:
    DocumentAccessControl.reset_instance()
    service = make_service()
    client.app.dependency_overrides[rag_api.get_rag_service] = lambda: service
    yield client
    client.app.dependency_overrides.clear()
    DocumentAccessControl.reset_instance()


ADMIN = {"X-User-Id": "root", "X-User-Role": "admin", "X-Organization-Id": "acme"}


def test_api_collections_endpoint_returns_200(real_api: TestClient) -> None:
    real_api.post(
        "/rag/ingest",
        json={"documents": [{"doc_id": "kb-1", "content": "Keys rotate every ninety days."}]},
        headers=ADMIN,
    )

    response = real_api.get("/rag/collections")

    assert response.status_code == 200, response.text
    assert response.json()["collections"][0]["name"] == DEFAULT_COLLECTION


def test_api_health_endpoint_returns_200(real_api: TestClient) -> None:
    response = real_api.post("/rag/health")

    assert response.status_code == 200, response.text
    assert response.json()["embedding_model"] == "hashing-test-embedder"


def test_api_embed_endpoint_returns_vector(real_api: TestClient) -> None:
    response = real_api.post("/rag/embed", json={"text": "hello world"})

    assert response.status_code == 200, response.text
    assert response.json()["dimension"] == DIM


# Tenant isolation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_org", "level", "allowed"),
    [
        (None, AccessLevel.INTERNAL, False),  # the Day 16 gap: no org header
        (None, AccessLevel.PUBLIC, True),
        ("acme", AccessLevel.INTERNAL, True),
        ("globex", AccessLevel.INTERNAL, False),
    ],
)
def test_org_scoped_documents_require_matching_org(
    user_org: str | None, level: AccessLevel, allowed: bool
) -> None:
    control = DocumentAccessControl()
    user = UserContext(user_id="u", role=Role.ANALYST, organization_id=user_org)
    doc = DocumentMetadata(doc_id="d", title="d", access_level=level, organization_id="acme")

    assert control.evaluate(user, doc).is_allowed is allowed


def test_documents_without_org_unchanged_for_users_without_org() -> None:
    control = DocumentAccessControl()
    user = UserContext(user_id="u", role=Role.ANALYST)
    doc = DocumentMetadata(doc_id="d", title="d", access_level=AccessLevel.INTERNAL)

    assert control.evaluate(user, doc).is_allowed is True


# Usage plumbing (ADR-013) ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_and_search_report_embedding_tokens_and_input_types() -> None:
    service = make_service()
    embedder = service.embedding_provider

    ingested = await service.ingest_documents(DOCS)
    searched = await service.search_similar("when do keys rotate", top_k=1)

    assert ingested["embedding_tokens"] == sum(len(d.content.split()) for d in DOCS)
    assert searched["embedding_tokens"] == 4
    assert [r.input_type for r in embedder.requests] == ["passage", "query"]


@pytest.mark.asyncio
async def test_unknown_usage_is_reported_as_none() -> None:
    class NoUsageEmbedder(HashingEmbedder):
        async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
            response = await super().embed(request)
            return EmbeddingResponse(
                embeddings=response.embeddings,
                model=response.model,
                provider=response.provider,
                dimension=response.dimension,
                usage=None,
            )

    store = QdrantVectorStore(QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM))
    service = RAGService(embedding_provider=NoUsageEmbedder(), vector_store=store)

    assert (await service.ingest_documents(DOCS))["embedding_tokens"] is None


# Replace-on-reingest (shrinking documents) ----------------------------------------------


def long_text(topic: str, sentences: int) -> str:
    return " ".join(
        f"Section {i} explains {topic} policy rule number {i}." for i in range(sentences)
    )


async def chunk_count(service: RAGService, document_id: str) -> int:
    store = service.vector_store
    results = await store.search(DEFAULT_COLLECTION, [1.0 / np.sqrt(DIM)] * DIM, limit=1000)
    return sum(1 for r in results if r.document_id == document_id)


@pytest.mark.asyncio
async def test_shorter_reingest_removes_old_tail_chunks() -> None:
    service = make_service()
    long_version = doc("kb-1", long_text("travel", 200))
    short_version = doc("kb-1", "Travel policy: book through the portal.")

    first = await service.ingest_documents([long_version, doc("kb-2", "Cafeteria opens at eight.")])
    assert first["chunks_removed"] == 0
    assert await chunk_count(service, "kb-1") > 1

    second = await service.ingest_documents([short_version])

    assert second["chunks_created"] == 1
    assert second["chunks_removed"] == first["chunks_created"] - 1 - 1  # kb-1 tail only
    assert await chunk_count(service, "kb-1") == 1
    assert await chunk_count(service, "kb-2") == 1  # other documents untouched
    results = (await service.search_similar("section policy rule number", top_k=10))["results"]
    assert all("Section" not in r["text"] for r in results)  # stale text no longer searchable


@pytest.mark.asyncio
async def test_identical_reingest_removes_nothing() -> None:
    service = make_service()
    await service.ingest_documents(DOCS)

    again = await service.ingest_documents(DOCS)

    assert again["chunks_removed"] == 0


@pytest.mark.asyncio
async def test_reingest_cleans_points_left_by_old_hash_ids() -> None:
    service = make_service()
    store = service.vector_store
    # A point written before Day 16 with a random integer ID but the same document_id.
    await store.insert(
        DEFAULT_COLLECTION,
        [[1.0 / np.sqrt(DIM)] * DIM],
        [{"text": "old copy", "document_id": "kb-1", "chunk_id": "kb-1#0"}],
        ids=["1739204551"],
    )

    result = await service.ingest_documents(DOCS[:1])

    assert result["chunks_removed"] == 1
    assert await chunk_count(service, "kb-1") == 1


@pytest.mark.asyncio
async def test_failed_reingest_keeps_previous_version() -> None:
    class FailingEmbedder(HashingEmbedder):
        async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
            raise RuntimeError("provider down")

    service = make_service()
    await service.ingest_documents([doc("kb-1", long_text("travel", 200))])
    before = await chunk_count(service, "kb-1")

    broken = RAGService(embedding_provider=FailingEmbedder(), vector_store=service.vector_store)
    with pytest.raises(RuntimeError):
        await broken.ingest_documents([doc("kb-1", "short")])

    assert await chunk_count(service, "kb-1") == before


def test_api_ingest_reports_chunks_removed(real_api: TestClient) -> None:
    body = {"documents": [{"doc_id": "kb-1", "content": long_text("travel", 200)}]}
    real_api.post("/rag/ingest", json=body, headers=ADMIN)

    body["documents"][0]["content"] = "Travel policy: book through the portal."
    response = real_api.post("/rag/ingest", json=body, headers=ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["chunks_removed"] > 0


# Configurable collection + model guard ---------------------------------------------------


@pytest.mark.asyncio
async def test_services_with_different_collections_are_isolated() -> None:
    store = QdrantVectorStore(
        QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM, embedding_model="m")
    )
    old = RAGService(embedding_provider=HashingEmbedder(), vector_store=store, collection="docs_v1")
    new = RAGService(embedding_provider=HashingEmbedder(), vector_store=store, collection="docs_v2")

    await old.ingest_documents(DOCS)

    assert (await new.search_similar("encryption keys", top_k=3))["results"] == []
    names = {c["name"] for c in (await old.list_collections())["collections"]}
    assert names == {"docs_v1"}


@pytest.mark.parametrize("name", ["", "has space", "bad/name", "x" * 256])
def test_invalid_collection_names_rejected(name: str) -> None:
    store = QdrantVectorStore(QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM))
    with pytest.raises(ValueError, match="Invalid collection name"):
        RAGService(embedding_provider=HashingEmbedder(), vector_store=store, collection=name)


@pytest.mark.asyncio
async def test_health_fails_after_switching_model_on_same_collection() -> None:
    first = QdrantVectorStore(
        QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM, embedding_model="old")
    )
    await RAGService(embedding_provider=HashingEmbedder(), vector_store=first).ingest_documents(
        DOCS
    )

    switched = QdrantVectorStore(
        QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM, embedding_model="new")
    )
    switched.client = first.client
    service = RAGService(embedding_provider=HashingEmbedder(), vector_store=switched)

    with pytest.raises(RuntimeError, match="was built with old"):
        await service.health_check()
    with pytest.raises(RuntimeError, match="was built with old"):
        await service.search_similar("encryption keys")


@pytest.mark.asyncio
async def test_health_reports_collection() -> None:
    health = await make_service().health_check()

    assert health["collection"] == DEFAULT_COLLECTION


def test_rag_collection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_COLLECTION", "documents_nemotron3_2048")
    assert rag_api.get_configured_collection() == "documents_nemotron3_2048"

    monkeypatch.delenv("RAG_COLLECTION")
    assert rag_api.get_configured_collection() == DEFAULT_COLLECTION


def test_embedding_model_passed_to_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_VECTOR_STORE_URL", IN_MEMORY_URL)  # never contact a real cluster
    store = rag_api.get_configured_vector_store(DIM, "nvidia/some-model")

    assert store.config.embedding_model == "nvidia/some-model"  # type: ignore[attr-defined]
    assert rag_api.embedding_model_for(HashingEmbedder()) == "hashing-test-embedder"
