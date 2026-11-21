"""QdrantVectorStore against the real qdrant-client in local (":memory:") mode.

These exercise the actual client API (collection_exists, create_collection, upsert,
query_points, delete) instead of mocks, so API drift such as the removed
``AsyncQdrantClient.search`` is caught by the test suite.
"""

import subprocess
import sys
import uuid

import pytest
from qdrant_client.models import Distance, VectorParams

from aegis_platform.rag.storage.qdrant import (
    IN_MEMORY_URL,
    CollectionMismatchError,
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    to_point_id,
)

DIM = 4


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM))


def vec(*values: float) -> list[float]:
    return list(values) + [0.0] * (DIM - len(values))


async def point_count(store: QdrantVectorStore, collection: str = "documents") -> int:
    stats = await store.collection_stats(collection)
    return int(stats["point_count"])


class TestPointIds:
    def test_numeric_ids_stay_integers(self):
        assert to_point_id("42") == 42

    def test_string_ids_map_to_stable_uuid5(self):
        point_id = to_point_id("doc-1#0")

        assert point_id == to_point_id("doc-1#0")
        assert uuid.UUID(str(point_id)).version == 5
        assert point_id != to_point_id("doc-1#1")

    def test_ids_are_identical_across_processes(self):
        """hash() is randomized per process; the UUIDv5 mapping must not be."""
        code = "from aegis_platform.rag.storage.qdrant import to_point_id; print(to_point_id('doc-1#0'))"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PYTHONPATH": "src"},
            ).stdout.strip()
            for seed in ("1", "2")
        }

        assert outputs == {str(to_point_id("doc-1#0"))}


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    async def test_insert_creates_missing_collection_with_configured_dimension(self, store):
        assert await store.list_collections() == []

        await store.insert("documents", [vec(1.0)], [{"text": "a"}], ids=["doc#0"])

        stats = await store.collection_stats("documents")
        assert stats["vector_size"] == DIM
        assert stats["distance"] == "cosine"
        assert stats["point_count"] == 1

    @pytest.mark.asyncio
    async def test_ensure_collection_is_idempotent(self, store):
        assert await store.ensure_collection("documents") is True
        assert await store.ensure_collection("documents") is False

    @pytest.mark.asyncio
    async def test_create_collection_never_drops_existing_data(self, store):
        await store.insert("documents", [vec(1.0)], [{}], ids=["doc#0"])

        with pytest.raises(ValueError, match="already exists"):
            await store.create_collection("documents", DIM, "cosine")

        assert await point_count(store) == 1

    @pytest.mark.asyncio
    async def test_create_and_delete_collection(self, store):
        await store.create_collection("articles", 8, "euclidean")
        assert "articles" in await store.list_collections()
        assert (await store.collection_stats("articles"))["distance"] == "euclid"

        await store.delete_collection("articles")
        assert "articles" not in await store.list_collections()

    @pytest.mark.asyncio
    async def test_health_check(self, store):
        assert await store.health_check() is True


class TestInsertAndSearch:
    @pytest.mark.asyncio
    async def test_search_ranks_by_similarity_and_returns_payload(self, store):
        await store.insert(
            "documents",
            [vec(1.0, 0.0), vec(0.0, 1.0), vec(0.7, 0.7)],
            [
                {"text": "east", "document_id": "d1", "chunk_id": "d1#0"},
                {"text": "north", "document_id": "d2", "chunk_id": "d2#0"},
                {"text": "north-east", "document_id": "d3", "chunk_id": "d3#0"},
            ],
            ids=["d1#0", "d2#0", "d3#0"],
        )

        results = await store.search("documents", vec(1.0, 0.1), limit=2)

        assert [r.text for r in results] == ["east", "north-east"]
        assert results[0].document_id == "d1"
        assert results[0].chunk_id == "d1#0"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_score_threshold_filters(self, store):
        await store.insert(
            "documents", [vec(1.0), vec(0.0, 1.0)], [{"text": "a"}, {"text": "b"}], ids=["a", "b"]
        )

        results = await store.search("documents", vec(1.0), limit=10, score_threshold=0.9)

        assert [r.text for r in results] == ["a"]

    @pytest.mark.asyncio
    async def test_search_before_any_ingestion_returns_empty(self, store):
        assert await store.search("documents", vec(1.0), limit=5) == []

    @pytest.mark.asyncio
    async def test_reingesting_same_ids_overwrites_instead_of_duplicating(self, store):
        for text in ("v1", "v2"):
            await store.insert("documents", [vec(1.0)], [{"text": text}], ids=["doc#0"])

        assert await point_count(store) == 1
        [result] = await store.search("documents", vec(1.0), limit=5)
        assert result.text == "v2"

    @pytest.mark.asyncio
    async def test_store_restart_still_addresses_same_points(self, store):
        """A new store instance (e.g. after restart) maps IDs to the same points."""
        await store.insert("documents", [vec(1.0)], [{"text": "a"}], ids=["doc#0"])
        restarted = QdrantVectorStore(store.config)
        restarted.client = store.client  # same backing database

        await restarted.insert("documents", [vec(1.0)], [{"text": "b"}], ids=["doc#0"])

        assert await point_count(store) == 1

    @pytest.mark.asyncio
    async def test_delete_and_update_use_original_ids(self, store):
        await store.insert(
            "documents",
            [vec(1.0), vec(0.0, 1.0)],
            [{"text": "a"}, {"text": "b"}],
            ids=["a#0", "b#0"],
        )

        await store.update("documents", ["a#0"], [vec(0.0, 0.0, 1.0)], [{"text": "a2"}])
        await store.delete("documents", ["b#0"])

        assert await point_count(store) == 1
        [result] = await store.search("documents", vec(0.0, 0.0, 1.0), limit=5)
        assert result.text == "a2"

    @pytest.mark.asyncio
    async def test_large_metadata_round_trips(self, store):
        metadata = {"text": "word " * 10_000, "tags": [f"tag_{i}" for i in range(100)]}

        await store.insert("documents", [vec(1.0)], [metadata], ids=["big"])

        [result] = await store.search("documents", vec(1.0), limit=1)
        assert result.metadata["tags"][-1] == "tag_99"
        assert len(result.text) == len(metadata["text"])


class TestDeleteStaleChunks:
    @pytest.mark.asyncio
    async def test_deletes_only_this_documents_points_outside_keep_set(self, store):
        payloads = [{"document_id": "a", "text": f"a{i}"} for i in range(3)] + [
            {"document_id": "b", "text": "b0"}
        ]
        await store.insert("documents", [vec(1.0)] * 4, payloads, ids=["a#0", "a#1", "a#2", "b#0"])

        removed = await store.delete_stale_chunks("documents", "a", keep_ids=["a#0"])

        assert removed == 2
        assert sorted(r.text for r in await store.search("documents", vec(1.0), limit=10)) == [
            "a0",
            "b0",
        ]

    @pytest.mark.asyncio
    async def test_empty_keep_set_removes_whole_document(self, store):
        await store.insert("documents", [vec(1.0)], [{"document_id": "a"}], ids=["a#0"])

        assert await store.delete_stale_chunks("documents", "a", keep_ids=[]) == 1
        assert await point_count(store) == 0

    @pytest.mark.asyncio
    async def test_missing_collection_is_a_noop(self, store):
        assert await store.delete_stale_chunks("documents", "a", keep_ids=["a#0"]) == 0

    @pytest.mark.asyncio
    async def test_requires_document_id(self, store):
        with pytest.raises(ValueError):
            await store.delete_stale_chunks("documents", "", keep_ids=[])


class TestCollectionGuard:
    """A collection refuses vectors from a different model/dimension/metric (ADR-012)."""

    @staticmethod
    def sibling(store: QdrantVectorStore, **overrides) -> QdrantVectorStore:
        """Another store sharing the same backing database (e.g. after a config change)."""
        config = QdrantVectorStoreConfig(
            url=IN_MEMORY_URL,
            embedding_dimension=overrides.get("dim", DIM),
            distance_metric=overrides.get("metric", "cosine"),
            embedding_model=overrides.get("model", "model-a"),
        )
        other = QdrantVectorStore(config)
        other.client = store.client
        return other

    @pytest.fixture
    def model_a(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            QdrantVectorStoreConfig(
                url=IN_MEMORY_URL, embedding_dimension=DIM, embedding_model="model-a"
            )
        )

    @pytest.mark.asyncio
    async def test_new_collection_records_model_dimension_metric(self, model_a):
        await model_a.ensure_collection("documents")

        info = await model_a.client.get_collection("documents")
        assert info.config.metadata == {
            "embedding_model": "model-a",
            "embedding_dimension": DIM,
            "distance_metric": "cosine",
        }

    @pytest.mark.asyncio
    async def test_same_configuration_is_accepted(self, model_a):
        await model_a.insert("documents", [vec(1.0)], [{"text": "a"}], ids=["a"])

        same = self.sibling(model_a)
        assert await same.verify_collection("documents") is True
        assert len(await same.search("documents", vec(1.0), limit=1)) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"model": "model-b"}, "model-b"),  # same size, different model: the silent case
            ({"dim": DIM * 2}, f"dimension {DIM * 2}"),
            ({"metric": "euclidean"}, "euclid"),
        ],
    )
    async def test_mismatch_is_refused_on_search_insert_and_cleanup(
        self, model_a, overrides, fragment
    ):
        await model_a.insert("documents", [vec(1.0)], [{"document_id": "a"}], ids=["a#0"])
        other = self.sibling(model_a, **overrides)
        vector = [0.1] * other.config.embedding_dimension

        with pytest.raises(CollectionMismatchError, match="RAG_COLLECTION") as raised:
            await other.search("documents", vector, limit=1)
        assert "model-a" in str(raised.value) and fragment in str(raised.value)

        with pytest.raises(CollectionMismatchError):
            await other.insert("documents", [vector], [{}], ids=["b#0"])
        with pytest.raises(CollectionMismatchError):
            await other.delete_stale_chunks("documents", "a", keep_ids=[])

        assert await point_count(model_a) == 1  # nothing was written or deleted

    @pytest.mark.asyncio
    async def test_legacy_collection_without_metadata_is_checked_by_dimension_only(self):
        legacy = QdrantVectorStore(
            QdrantVectorStoreConfig(url=IN_MEMORY_URL, embedding_dimension=DIM)
        )
        await legacy.client.create_collection(
            "documents", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
        )
        current = self.sibling(legacy, model="model-a")

        assert await current.verify_collection("documents") is True
        with pytest.raises(CollectionMismatchError):
            await self.sibling(legacy, dim=DIM + 1).verify_collection("documents")

    @pytest.mark.asyncio
    async def test_missing_collection_verifies_false(self, model_a):
        assert await model_a.verify_collection("documents") is False
