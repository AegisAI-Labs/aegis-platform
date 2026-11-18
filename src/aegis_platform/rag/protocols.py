"""Unified protocols module for RAG abstractions.

This module re-exports embedding and vector store protocols for convenience.
All RAG components should import protocols from this module.
"""

from aegis_platform.rag.embeddings.base import EmbeddingProvider
from aegis_platform.rag.storage.base import VectorStore, VectorStoreConfig

__all__ = [
    "EmbeddingProvider",
    "VectorStore",
    "VectorStoreConfig",
]
