"""Factory function for creating chunking strategies.

Provides a factory function that instantiates chunking strategies
based on configuration, enabling dependency injection and runtime
strategy selection via environment variables.
"""

from typing import cast

from aegis_platform.rag.chunking.langchain_markdown import LangChainMarkdownChunker
from aegis_platform.rag.chunking.langchain_recursive import LangChainRecursiveChunker
from aegis_platform.rag.chunking.strategy import ChunkingConfig, ChunkingStrategy

# Registry of available chunking strategies
# Uses LangChain's battle-tested text splitters for reliability and features
CHUNKING_STRATEGIES = {
    "recursive": LangChainRecursiveChunker,
    "markdown": LangChainMarkdownChunker,
}


def create_chunker(config: ChunkingConfig) -> ChunkingStrategy:
    """Factory function to create chunker based on configuration.

    Creates an instance of a LangChain-based chunking strategy.
    All strategies use LangChain's battle-tested text splitters.

    Args:
        config: ChunkingConfig with strategy name and parameters

    Returns:
        Instantiated ChunkingStrategy implementation

    Raises:
        ValueError: If strategy is not found or config is invalid

    Example:
        >>> from aegis_platform.rag.chunking import ChunkingConfig, create_chunker
        >>> config = ChunkingConfig(strategy="recursive", chunk_size=512)
        >>> chunker = create_chunker(config)
        >>> chunks = chunker.chunk("Text to chunk")

    Available Strategies:
        - recursive: RecursiveCharacterTextSplitter (handles varied text structure)
        - markdown: MarkdownTextSplitter (markdown-aware with code block preservation)
    """
    strategy_class = CHUNKING_STRATEGIES.get(config.strategy)

    if not strategy_class:
        available = ", ".join(CHUNKING_STRATEGIES.keys())
        raise ValueError(
            f"Unknown chunking strategy: {config.strategy}. " f"Available: {available}"
        )

    # All LangChain strategies use chunk_size and overlap
    return cast(
        ChunkingStrategy,
        strategy_class(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
        ),
    )


def get_available_strategies() -> list[str]:
    """Get list of available chunking strategy names.

    Returns:
        List of strategy names that can be used with create_chunker()
    """
    return list(CHUNKING_STRATEGIES.keys())
