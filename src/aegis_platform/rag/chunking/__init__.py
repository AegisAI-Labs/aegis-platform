"""Chunking strategies for RAG document processing.

This subpackage provides LangChain-based text chunking strategies:

1. LangChainRecursiveChunker: Recursive splitting handling varied text structure
2. LangChainMarkdownChunker: Markdown-aware splitting with code block preservation

All strategies implement ChunkingStrategy protocol and support:
- Runtime configuration via update_config()
- Chunk size/overlap adjustment
- Metadata tracking
- Position tracking in original text

Package structure:
- strategy.py: Abstract ChunkingStrategy protocol and ChunkingConfig
- langchain_recursive.py: LangChainRecursiveChunker (LangChain wrapper)
- langchain_markdown.py: LangChainMarkdownChunker (LangChain wrapper)
- factory.py: create_chunker() factory function

Example:
    >>> from aegis_platform.rag.chunking import ChunkingConfig, create_chunker
    >>> # Recursive splitting for varied structure
    >>> config = ChunkingConfig(strategy="recursive", chunk_size=512)
    >>> chunker = create_chunker(config)
    >>> chunks = chunker.chunk("Long text here...")
    >>>
    >>> # Markdown-aware splitting
    >>> config = ChunkingConfig(strategy="markdown", chunk_size=1024)
    >>> chunker = create_chunker(config)
    >>> chunks = chunker.chunk("# Markdown\\nContent here...")
"""

from aegis_platform.rag.chunking.factory import create_chunker, get_available_strategies
from aegis_platform.rag.chunking.langchain_markdown import LangChainMarkdownChunker
from aegis_platform.rag.chunking.langchain_recursive import LangChainRecursiveChunker
from aegis_platform.rag.chunking.strategy import ChunkingConfig, ChunkingStrategy

__all__ = [
    "ChunkingConfig",
    "ChunkingStrategy",
    "LangChainMarkdownChunker",
    "LangChainRecursiveChunker",
    "create_chunker",
    "get_available_strategies",
]
