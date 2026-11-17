"""Embedding providers for text-to-vector conversion.

This package contains abstract protocols, configuration classes, and concrete
implementations for generating embeddings from text using various providers.

Modules:
    base: Abstract EmbeddingProvider protocol
    openai: OpenAI embedding provider implementation
    nvidia: NVIDIA NIM API embedding provider implementation

Classes exported:
    EmbeddingProvider: Abstract protocol for embedding providers
    OpenAIEmbeddingConfig: OpenAI-specific configuration
    OpenAIEmbedding: OpenAI embedding provider implementation
    NVIDIAEmbeddingConfig: NVIDIA-specific configuration
    NVIDIAEmbeddingProvider: NVIDIA embedding provider implementation

Usage:
    from aegis_platform.rag.embeddings import (
        OpenAIEmbeddingConfig, OpenAIEmbedding,
        NVIDIAEmbeddingConfig, NVIDIAEmbeddingProvider
    )
"""

from aegis_platform.rag.embeddings.base import EmbeddingProvider
from aegis_platform.rag.embeddings.nvidia import (
    NVIDIAEmbeddingConfig,
    NVIDIAEmbeddingProvider,
)
from aegis_platform.rag.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)

__all__ = [
    "EmbeddingProvider",
    "NVIDIAEmbeddingConfig",
    "NVIDIAEmbeddingProvider",
    "OpenAIEmbedding",
    "OpenAIEmbeddingConfig",
]
