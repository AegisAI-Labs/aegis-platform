"""Abstract protocol for embedding providers.

This module defines the EmbeddingProvider protocol that all concrete
embedding implementations must follow.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Abstract protocol for text embedding providers.

    Implementations must support:
    - Single text embedding
    - Batch embedding for efficiency
    - Dimension information
    - Health checks for operational monitoring
    """

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
