"""LangChain-based markdown chunking strategy.

Wraps LangChain's MarkdownTextSplitter for markdown-aware chunking that
preserves document structure like headers, code blocks, and lists.

This strategy is recommended for markdown documentation and files where
preserving heading hierarchy and code blocks is important.
"""

from dataclasses import dataclass

from langchain_text_splitters import MarkdownTextSplitter

from aegis_platform.rag.chunking.strategy import ChunkingConfig, ChunkingStrategy
from aegis_platform.rag.models import Chunk


@dataclass
class LangChainMarkdownConfig(ChunkingConfig):
    """Configuration for LangChain MarkdownTextSplitter."""

    strategy: str = "langchain-markdown"


class LangChainMarkdownChunker(ChunkingStrategy):
    """Chunker using LangChain's MarkdownTextSplitter.

    This is a wrapper around LangChain's markdown-aware splitter that respects
    markdown structure (headers, code blocks, lists, etc.) when creating chunks.

    Recommended for:
    - Markdown documentation
    - README files
    - API documentation
    - Any content with markdown formatting
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        overlap: int = 128,
        min_chunk_size: int = 50,
        max_chunk_size: int = 10000,
    ):
        """Initialize LangChain MarkdownChunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to create
            max_chunk_size: Maximum chunk size allowed

        Raises:
            ValueError: If configuration is invalid
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Validate configuration
        self._validate()

        # Create underlying LangChain splitter
        self._splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be 0 <= overlap < chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("max_chunk_size must be > min_chunk_size")

    def chunk(self, text: str, document_id: str = "unknown") -> list[Chunk]:
        """Split markdown text using LangChain's markdown splitter.

        Args:
            text: Markdown text to chunk
            document_id: ID of source document

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text")

        # Use LangChain's markdown splitter to get text chunks
        text_chunks = self._splitter.split_text(text)

        # Convert to Chunk objects with metadata
        chunks = []
        pos = 0

        for chunk_id, chunk_text in enumerate(text_chunks):
            # Find position in original text
            start_pos = text.find(chunk_text, pos)
            if start_pos == -1:
                start_pos = pos

            end_pos = start_pos + len(chunk_text)
            pos = end_pos

            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"{document_id}#{chunk_id}",
                    sequence=chunk_id,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata={
                        "strategy": "langchain-markdown",
                        "document_id": document_id,
                    },
                )
            )

        if not chunks:
            raise ValueError("Could not create any chunks from text")

        return chunks

    def get_config(self) -> dict:
        """Get current chunking configuration."""
        return {
            "strategy": "langchain-markdown",
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
        }

    def update_config(self, **kwargs) -> None:
        """Update configuration at runtime.

        Args:
            **kwargs: Configuration parameters to update

        Raises:
            TypeError: If parameter type is wrong
            ValueError: If parameter value is invalid or unknown
        """
        for key, value in kwargs.items():
            if key == "chunk_size":
                if not isinstance(value, int):
                    raise TypeError(f"chunk_size must be int, got {type(value)}")
                if value <= 0:
                    raise ValueError("chunk_size must be positive")
                self.chunk_size = value
            elif key == "overlap":
                if not isinstance(value, int):
                    raise TypeError(f"overlap must be int, got {type(value)}")
                if value < 0 or value >= self.chunk_size:
                    raise ValueError("overlap must be 0 <= overlap < chunk_size")
                self.overlap = value
            elif key == "min_chunk_size":
                if not isinstance(value, int):
                    raise TypeError(f"min_chunk_size must be int, got {type(value)}")
                if value <= 0:
                    raise ValueError("min_chunk_size must be positive")
                self.min_chunk_size = value
            elif key == "max_chunk_size":
                if not isinstance(value, int):
                    raise TypeError(f"max_chunk_size must be int, got {type(value)}")
                if value <= self.min_chunk_size:
                    raise ValueError("max_chunk_size must be > min_chunk_size")
                self.max_chunk_size = value
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Recreate splitter with updated config
        self._validate()
        self._splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
        )

    def estimate_chunks(self, text: str) -> int:
        """Estimate number of chunks for given text.

        Args:
            text: Text to estimate

        Returns:
            Estimated number of chunks
        """
        if not text or not text.strip():
            return 0

        # Rough estimate: total_length / (chunk_size - overlap)
        effective_chunk_size = max(self.chunk_size - self.overlap, 1)
        return max(1, len(text) // effective_chunk_size)
