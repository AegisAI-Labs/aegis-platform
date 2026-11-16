"""Unit tests for LangChain-based chunking strategies.

Tests cover:
- LangChainRecursiveChunker with various document structures
- LangChainMarkdownChunker with markdown-specific content
- Configuration and runtime updates
"""

import pytest

from aegis_platform.rag.chunking import (
    ChunkingConfig,
    LangChainMarkdownChunker,
    LangChainRecursiveChunker,
    create_chunker,
    get_available_strategies,
)


class TestLangChainRecursiveChunker:
    """Test LangChainRecursiveChunker implementation."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        chunker = LangChainRecursiveChunker(chunk_size=512, overlap=50)
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_init_invalid_chunk_size(self):
        """Test initialization with invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            LangChainRecursiveChunker(chunk_size=-1, overlap=50)

    def test_init_invalid_overlap(self):
        """Test initialization with overlap >= chunk_size."""
        with pytest.raises(ValueError, match="overlap must be"):
            LangChainRecursiveChunker(chunk_size=512, overlap=512)

    def test_chunk_basic(self):
        """Test basic chunking with simple text."""
        chunker = LangChainRecursiveChunker(chunk_size=100, overlap=10)
        text = "A" * 250
        chunks = chunker.chunk(text, "doc1")

        assert len(chunks) > 1
        assert all(chunk.text for chunk in chunks)
        assert chunks[0].metadata["strategy"] == "langchain-recursive"

    def test_chunk_empty_text(self):
        """Test chunking empty text raises error."""
        chunker = LangChainRecursiveChunker(chunk_size=100, overlap=10)
        with pytest.raises(ValueError):
            chunker.chunk("", "doc1")

    def test_chunk_with_paragraphs(self):
        """Test chunking preserves paragraph structure."""
        chunker = LangChainRecursiveChunker(chunk_size=200, overlap=20)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text, "doc1")

        assert len(chunks) >= 1
        assert all(chunk.text for chunk in chunks)

    def test_get_config(self):
        """Test getting current configuration."""
        chunker = LangChainRecursiveChunker(chunk_size=256, overlap=32)
        config = chunker.get_config()

        assert config["strategy"] == "langchain-recursive"
        assert config["chunk_size"] == 256
        assert config["overlap"] == 32

    def test_update_config_chunk_size(self):
        """Test updating chunk size."""
        chunker = LangChainRecursiveChunker(chunk_size=512, overlap=50)
        chunker.update_config(chunk_size=256)

        assert chunker.chunk_size == 256
        config = chunker.get_config()
        assert config["chunk_size"] == 256

    def test_update_config_overlap(self):
        """Test updating overlap."""
        chunker = LangChainRecursiveChunker(chunk_size=512, overlap=50)
        chunker.update_config(overlap=100)

        assert chunker.overlap == 100

    def test_metadata_tracking(self):
        """Test metadata is properly tracked in chunks."""
        chunker = LangChainRecursiveChunker(chunk_size=200, overlap=20)
        text = "Sample text for metadata testing. " * 20
        chunks = chunker.chunk(text, "doc-123")

        for chunk in chunks:
            assert chunk.metadata["strategy"] == "langchain-recursive"
            assert chunk.metadata["document_id"] == "doc-123"
            assert chunk.chunk_id.startswith("doc-123#")


class TestLangChainMarkdownChunker:
    """Test LangChainMarkdownChunker implementation."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        chunker = LangChainMarkdownChunker(chunk_size=512, overlap=50)
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_chunk_basic(self):
        """Test basic markdown chunking."""
        chunker = LangChainMarkdownChunker(chunk_size=200, overlap=20)
        text = "# Header\n\nContent here.\n\n## Subheader\n\nMore content."
        chunks = chunker.chunk(text, "doc1")

        assert len(chunks) > 0
        assert all(chunk.text for chunk in chunks)
        assert chunks[0].metadata["strategy"] == "langchain-markdown"

    def test_chunk_with_code_blocks(self):
        """Test chunking preserves code blocks."""
        chunker = LangChainMarkdownChunker(chunk_size=300, overlap=30)
        text = """# Example

```python
def hello():
    return "world"
```

More text here.
"""
        chunks = chunker.chunk(text, "doc1")

        assert len(chunks) > 0
        assert any("hello" in chunk.text or "world" in chunk.text for chunk in chunks)

    def test_chunk_empty_text(self):
        """Test chunking empty text raises error."""
        chunker = LangChainMarkdownChunker(chunk_size=100, overlap=10)
        with pytest.raises(ValueError):
            chunker.chunk("", "doc1")

    def test_get_config(self):
        """Test getting current configuration."""
        chunker = LangChainMarkdownChunker(chunk_size=256, overlap=32)
        config = chunker.get_config()

        assert config["strategy"] == "langchain-markdown"
        assert config["chunk_size"] == 256
        assert config["overlap"] == 32

    def test_update_config(self):
        """Test updating configuration."""
        chunker = LangChainMarkdownChunker(chunk_size=512, overlap=50)
        chunker.update_config(chunk_size=300, overlap=30)

        assert chunker.chunk_size == 300
        assert chunker.overlap == 30

    def test_metadata_tracking(self):
        """Test metadata is properly tracked."""
        chunker = LangChainMarkdownChunker(chunk_size=200, overlap=20)
        text = "# Title\n\n" + "Content line.\n\n" * 30
        chunks = chunker.chunk(text, "doc-456")

        for chunk in chunks:
            assert chunk.metadata["strategy"] == "langchain-markdown"
            assert chunk.metadata["document_id"] == "doc-456"
            assert chunk.chunk_id.startswith("doc-456#")


class TestChunkingFactory:
    """Test factory function for creating chunkers."""

    def test_create_recursive(self):
        """Test creating recursive chunker via factory."""
        config = ChunkingConfig(strategy="recursive", chunk_size=512)
        chunker = create_chunker(config)

        assert isinstance(chunker, LangChainRecursiveChunker)
        assert chunker.chunk_size == 512

    def test_create_markdown(self):
        """Test creating markdown chunker via factory."""
        config = ChunkingConfig(strategy="markdown", chunk_size=256)
        chunker = create_chunker(config)

        assert isinstance(chunker, LangChainMarkdownChunker)
        assert chunker.chunk_size == 256

    def test_create_invalid_strategy(self):
        """Test factory rejects invalid strategy."""
        config = ChunkingConfig(strategy="invalid-strategy", chunk_size=512)
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker(config)

    def test_factory_respects_config(self):
        """Test factory creates chunker with correct config."""
        config = ChunkingConfig(strategy="recursive", chunk_size=1024, overlap=100)
        chunker = create_chunker(config)

        assert chunker.chunk_size == 1024
        assert chunker.overlap == 100

    def test_get_available_strategies(self):
        """Test getting list of available strategies."""
        strategies = get_available_strategies()

        assert len(strategies) == 2
        assert "recursive" in strategies
        assert "markdown" in strategies


class TestChunkingIntegration:
    """Integration tests for chunking strategies."""

    def test_all_strategies_handle_same_text(self):
        """Test all strategies can process the same text."""
        text = "Sample document. " * 50

        strategies = get_available_strategies()
        assert len(strategies) > 0

        for strategy_name in strategies:
            config = ChunkingConfig(strategy=strategy_name, chunk_size=256)
            chunker = create_chunker(config)
            chunks = chunker.chunk(text, f"doc-{strategy_name}")

            assert len(chunks) > 0
            assert all(len(chunk.text) > 0 for chunk in chunks)

    def test_strategy_reconfiguration(self):
        """Test reconfiguring strategy at runtime."""
        config = ChunkingConfig(strategy="recursive", chunk_size=500)
        chunker = create_chunker(config)

        text = "A" * 1000
        chunks1 = chunker.chunk(text, "doc1")

        chunker.update_config(chunk_size=250)
        chunks2 = chunker.chunk(text, "doc2")

        # Smaller chunk size should produce more chunks
        assert len(chunks2) >= len(chunks1)
