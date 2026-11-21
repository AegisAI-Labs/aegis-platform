"""NVIDIA embedding provider using NIM (NVIDIA Inference Microservice) API.

Provides high-performance embeddings using NVIDIA's Nemotron models via the
NIM API endpoint. Supports batching, async operations, error handling, and
retry logic with exponential backoff.

Models supported:
  - nvidia/nemotron-3.5-embed-1b (1536 dimensions, recommended)
  - nvidia/nemotron-4-mini-embed-1b (1024 dimensions)
  - nvidia/nemotron-4-9b-embed (1024 dimensions)

API Documentation:
  https://docs.nvidia.com/nim/large-language-models/latest/
"""

import asyncio
import logging
import os
import types
from dataclasses import dataclass
from typing import Self

import httpx
import numpy as np

from aegis_platform.rag.embeddings.base import EmbeddingProvider
from aegis_platform.rag.embeddings.usage import sum_usage
from aegis_platform.rag.models import (
    PASSAGE,
    QUERY,
    EmbeddingRequest,
    EmbeddingResponse,
    InputType,
)

logger = logging.getLogger(__name__)


@dataclass
class NVIDIAEmbeddingConfig:
    """Configuration for NVIDIA embedding provider.

    Attributes:
        api_key: NVIDIA API key (from environment or parameter)
        model: Model name/path (default: nvidia/nemotron-3.5-embed-1b)
        embedding_dimension: Expected output dimension (default: 1536)
        batch_size: Maximum embeddings per batch request (default: 1024)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 3)
        api_endpoint: NIM API base URL (default: NVIDIA official)
        send_input_type: Send ``input_type`` ("query" for searches, "passage" for ingested
            documents). Required by asymmetric retrieval models such as Nemotron embed;
            set False for symmetric models that reject the parameter.
    """

    api_key: str
    model: str = "nvidia/nemotron-3.5-embed-1b"
    embedding_dimension: int = 1536
    batch_size: int = 1024
    timeout: int = 30
    max_retries: int = 3
    api_endpoint: str = "https://integrate.api.nvidia.com/v1"
    send_input_type: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("NVIDIA API key is required and cannot be empty")

        if not self.model or not self.model.strip():
            raise ValueError("Model name is required and cannot be empty")

        if self.embedding_dimension <= 0:
            raise ValueError(
                f"Embedding dimension must be positive, got {self.embedding_dimension}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")

        if self.max_retries < 0:
            raise ValueError(f"Max retries must be non-negative, got {self.max_retries}")

    @classmethod
    def from_env(cls) -> "NVIDIAEmbeddingConfig":
        """Create configuration from environment variables.

        Environment Variables:
            RAG_EMBEDDING_KEY: NVIDIA API key (required)
            RAG_EMBEDDING_MODEL: Model name (optional, default provided)
            RAG_EMBEDDING_DIMENSION: Output dimension (optional)
            RAG_EMBEDDING_BATCH_SIZE: Batch size (optional)
            RAG_EMBEDDING_TIMEOUT: Timeout seconds (optional)
            RAG_EMBEDDING_MAX_RETRIES: Max retries (optional)
            RAG_EMBEDDING_API_ENDPOINT: API endpoint (optional)
            RAG_EMBEDDING_SEND_INPUT_TYPE: "true" (default) or "false"

        Returns:
            Configured NVIDIAEmbeddingConfig instance

        Raises:
            ValueError: If required environment variables are missing/invalid
        """
        api_key = os.getenv("RAG_EMBEDDING_KEY", "").strip()
        if not api_key:
            raise ValueError("RAG_EMBEDDING_KEY environment variable is required")

        model = os.getenv("RAG_EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b")
        embedding_dimension = int(os.getenv("RAG_EMBEDDING_DIMENSION", "1536"))
        batch_size = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "1024"))
        timeout = int(os.getenv("RAG_EMBEDDING_TIMEOUT", "30"))
        max_retries = int(os.getenv("RAG_EMBEDDING_MAX_RETRIES", "3"))
        api_endpoint = os.getenv(
            "RAG_EMBEDDING_API_ENDPOINT",
            "https://integrate.api.nvidia.com/v1",
        )

        send_input_type = os.getenv(
            "RAG_EMBEDDING_SEND_INPUT_TYPE", "true"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
        )

        return cls(
            api_key=api_key,
            model=model,
            embedding_dimension=embedding_dimension,
            batch_size=batch_size,
            timeout=timeout,
            max_retries=max_retries,
            api_endpoint=api_endpoint,
            send_input_type=send_input_type,
        )


PROVIDER_NAME = "nvidia"


@dataclass(frozen=True)
class _Batch:
    """Result of one embeddings API call."""

    vectors: list[list[float]]
    input_tokens: int | None
    model: str | None


# Client errors that can never succeed on retry.
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 413, 422})


class NVIDIAEmbeddingProvider(EmbeddingProvider):
    """NVIDIA embedding provider using NIM API.

    Provides async embedding generation with:
    - Automatic batching for efficiency
    - Exponential backoff retry logic
    - Connection pooling via httpx
    - Comprehensive error handling
    - Health check functionality

    Example:
        config = NVIDIAEmbeddingConfig(api_key="nvapi-...")
        provider = NVIDIAEmbeddingProvider(config)
        embeddings = await provider.embed(["Hello world", "Test"])
    """

    def __init__(self, config: NVIDIAEmbeddingConfig) -> None:
        """Initialize NVIDIA embedding provider.

        Args:
            config: NVIDIAEmbeddingConfig instance with all settings

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.client: httpx.AsyncClient | None = None
        self._headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"Initialized NVIDIA embedding provider: model={config.model}, "
            f"dimension={config.embedding_dimension}, batch_size={config.batch_size}"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling.

        Returns:
            httpx.AsyncClient instance
        """
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self._headers,
                http2=True,
            )

        return self.client

    @property
    def model(self) -> str:
        return self.config.model

    async def _embed_batch(self, texts: list[str], input_type: InputType = PASSAGE) -> _Batch:
        """Call NVIDIA NIM API to embed a batch of texts.

        Args:
            texts: List of texts to embed (max batch_size)
            input_type: "query" for search queries, "passage" for stored documents

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If API call fails after retries
            ValueError: If response format is invalid
        """
        if not texts:
            return _Batch(vectors=[], input_tokens=0, model=None)

        payload: dict[str, object] = {
            "input": texts,
            "model": self.config.model,
        }
        if self.config.send_input_type:
            payload["input_type"] = input_type

        client = await self._get_client()
        url = f"{self.config.api_endpoint}/embeddings"

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.post(url, json=payload)

                if response.status_code == 401:
                    raise ValueError("Invalid NVIDIA API key (401 Unauthorized)")

                if response.status_code in _NON_RETRYABLE_STATUS:
                    # Fail fast; do not include the response body (may echo inputs).
                    raise RuntimeError(
                        f"NVIDIA embedding request rejected (HTTP {response.status_code})"
                    )

                if response.status_code == 429:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Rate limited, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()

                data = response.json()

                if "data" not in data or not isinstance(data["data"], list):
                    raise ValueError(
                        f"Invalid response format: expected 'data' list, "
                        f"got {type(data.get('data'))}"
                    )

                embeddings = [
                    item["embedding"]
                    for item in sorted(data["data"], key=lambda x: x.get("index", 0))
                ]

                if not embeddings:
                    raise ValueError("No embeddings in response")

                if len(embeddings[0]) != self.config.embedding_dimension:
                    raise ValueError(
                        f"Unexpected embedding dimension: "
                        f"expected {self.config.embedding_dimension}, "
                        f"got {len(embeddings[0])}"
                    )

                logger.debug(
                    f"Successfully embedded {len(texts)} texts ({len(embeddings[0])} dimensions)"
                )

                usage = data.get("usage")
                tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                served_model = data.get("model")
                return _Batch(
                    vectors=embeddings,
                    input_tokens=tokens if isinstance(tokens, int) else None,
                    model=served_model if isinstance(served_model, str) and served_model else None,
                )

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e

                if attempt < self.config.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Embedding request failed ({_describe(e)}). "
                        f"Retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Embedding request failed after {self.config.max_retries + 1} attempts"
                    )

        raise RuntimeError(
            f"Failed to embed texts after {self.config.max_retries + 1} attempts"
            + (f" ({_describe(last_error)})" if last_error else "")
        ) from last_error

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed ``request.texts`` with the request's ``input_type``, batching as needed.

        Raises:
            ValueError: If the request is empty, a text is empty, or the response is malformed
            RuntimeError: If the API request fails
        """
        texts = request.texts
        if not texts:
            raise ValueError("Cannot embed empty text list")

        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All inputs must be strings")

        empty = [index for index, text in enumerate(texts) if not text.strip()]
        if empty:
            raise ValueError(f"Cannot embed empty texts at positions {empty[:10]}")

        logger.info(f"Embedding {len(texts)} texts (batch_size={self.config.batch_size})")

        embeddings: list[np.ndarray] = []
        token_counts: list[int | None] = []
        served_model = self.config.model

        for i in range(0, len(texts), self.config.batch_size):
            batch = await self._embed_batch(
                texts[i : i + self.config.batch_size], input_type=request.input_type
            )
            embeddings.extend(np.array(emb, dtype=np.float32) for emb in batch.vectors)
            token_counts.append(batch.input_tokens)
            served_model = batch.model or served_model

        logger.info(f"Successfully embedded {len(embeddings)} texts")

        return EmbeddingResponse(
            embeddings=embeddings,
            model=served_model,
            provider=PROVIDER_NAME,
            dimension=len(embeddings[0]),
            usage=sum_usage(token_counts),
        )

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single query text (``input_type="query"``).

        Raises:
            ValueError: If text is empty
            RuntimeError: If API request fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        response = await self.embed(EmbeddingRequest(texts=[text], input_type=QUERY))
        return response.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed passages for storage (``input_type="passage"``).

        Returns an empty list for empty input (existing behavior).
        """
        if not texts:
            return []
        response = await self.embed(EmbeddingRequest(texts=texts, input_type=PASSAGE))
        return response.embeddings

    async def get_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            Embedding dimension (1536 for Nemotron model)
        """
        return self.config.embedding_dimension

    async def health_check(self) -> bool:
        """Check provider health and connectivity.

        Performs a test embedding request to verify the NVIDIA NIM API
        is accessible and responding correctly.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            batch = await self._embed_batch(["health check"], input_type=QUERY)
            if not batch.vectors or len(batch.vectors[0]) != self.config.embedding_dimension:
                logger.warning("Health check failed: invalid embedding dimension")
                return False
            logger.debug("NVIDIA NIM API health check passed")
            return True
        except (TimeoutError, ValueError, RuntimeError) as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client connection pool."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
            logger.debug("NVIDIA embedding provider client closed")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()


def _describe(error: Exception) -> str:
    """Error summary without URLs or response bodies."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__
