"""Abstract protocol for embedding providers.

This module defines the EmbeddingProvider protocol that all concrete
embedding implementations must follow.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from aegis_platform.rag.models import EmbeddingRequest, EmbeddingResponse


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Abstract protocol for text embedding providers.

    Implementations must support:
    - ``embed``: the primary call, returning vectors with model and token usage (ADR-013)
    - ``embed_text`` / ``embed_batch``: convenience wrappers returning bare vectors
      (queries and passages respectively)
    - Dimension information
    - Health checks for operational monitoring
    """

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed ``request.texts``; vectors are aligned with the input order.

        Raises:
            ValueError: If the request is empty or contains empty texts
            RuntimeError: If embedding fails
        """
        ...

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1D numpy array)

        Raises:
            ValueError: If text is empty or malformed
            RuntimeError: If embedding fails
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts efficiently.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (1D numpy arrays)

        Raises:
            ValueError: If texts list is empty or exceeds batch limits
            RuntimeError: If embedding fails
        """
        ...

    async def get_dimension(self) -> int:
        """Get embedding vector dimension for this provider.

        Returns:
            Number of dimensions in embedding vectors
        """
        ...

    async def health_check(self) -> bool:
        """Check if provider is available and responding.

        Returns:
            True if provider is healthy, False otherwise
        """
        ...
