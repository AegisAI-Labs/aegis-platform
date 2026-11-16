"""Abstract chunking strategy protocol.

The ChunkingStrategy protocol enables flexible text segmentation with
runtime configuration. Different documents and use cases benefit from
different chunking approaches.

This design allows strategies to be selected and configured via environment
variables or dependency injection.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from aegis_platform.rag.models import Chunk


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy instantiation.

    Attributes:
        strategy: Strategy name (e.g., 'fixed-size', 'semantic', 'hierarchical')
        chunk_size: Target chunk size in characters (for fixed-size)
        overlap: Overlap between chunks in characters (for fixed-size)
        min_chunk_size: Minimum chunk size to enforce
        max_chunk_size: Maximum chunk size to enforce
        preserve_sentences: If True, don't split mid-sentence
        separator: Delimiter to use for splitting (for simple strategies)
        extra_params: Additional strategy-specific parameters
    """

    strategy: str
    chunk_size: int = 512
    overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 2048
    preserve_sentences: bool = True
    separator: str = "\n\n"
    extra_params: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be 0 <= overlap < chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("max_chunk_size must be > min_chunk_size")


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Abstract protocol for document chunking strategies.

    Implementations must provide:
    - chunk(): Split text into chunks with metadata
    - get_config(): Inspect current configuration
    - update_config(): Modify chunking parameters at runtime
    - estimate_chunks(): Predict number of chunks before chunking

    This enables flexible chunk processing:
    - Switch strategies based on document type
    - Reconfigure chunk size on-the-fly
    - Test different chunking approaches
    - Monitor chunking statistics

    All strategies must handle edge cases:
    - Empty or very small texts
    - Very large texts
    - Unicode and special characters
    - Different line endings (\\n, \\r\\n, \\r)
    """

    def chunk(self, text: str, document_id: str = "unknown") -> list[Chunk]:
        """Split text into chunks with metadata.

        Args:
            text: Text to chunk
            document_id: ID of source document for metadata

        Returns:
            List of Chunk objects with positions and sequence numbers

        Raises:
            ValueError: If text is empty or None
            RuntimeError: If chunking fails
        """
        ...

    def get_config(self) -> dict:
        """Get current chunking configuration.

        Returns:
            Dictionary of configuration parameters
        """
        ...

    def update_config(self, **kwargs) -> None:
        """Update chunking configuration at runtime.

        Args:
            **kwargs: Configuration parameters to update

        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong type
        """
        ...

    def estimate_chunks(self, text: str) -> int:
        """Estimate number of chunks before actual chunking.

        Args:
            text: Text to estimate

        Returns:
            Estimated number of chunks

        Note:
            Useful for validating performance before processing large documents.
        """
        ...
