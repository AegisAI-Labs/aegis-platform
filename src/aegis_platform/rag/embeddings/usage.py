"""Usage aggregation shared by embedding providers (ADR-013)."""

from collections.abc import Iterable

from aegis_platform.rag.models import EmbeddingUsage


def sum_usage(token_counts: Iterable[int | None]) -> EmbeddingUsage | None:
    """Sum per-call token counts; unknown (``None``) if any call did not report usage."""
    counts = list(token_counts)
    if not counts or any(count is None for count in counts):
        return None
    return EmbeddingUsage(input_tokens=sum(count for count in counts if count is not None))
