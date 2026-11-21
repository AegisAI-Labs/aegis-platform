"""Unit tests for OpenAI EmbeddingProvider implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from openai import RateLimitError

from aegis_platform.rag.embeddings.base import EmbeddingProvider
from aegis_platform.rag.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)
from aegis_platform.rag.embeddings.usage import sum_usage
from aegis_platform.rag.models import EmbeddingRequest, EmbeddingUsage


class TestOpenAIEmbeddingConfig:
    """Test OpenAIEmbeddingConfig validation."""

    def test_valid_config(self):
        """Test valid configuration is accepted."""
        config = OpenAIEmbeddingConfig(api_key="sk-test123")
        config.validate()  # Should not raise

    def test_config_empty_api_key(self):
        """Test empty API key is rejected."""
        config = OpenAIEmbeddingConfig(api_key="")
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            config.validate()

    def test_config_invalid_batch_size_zero(self):
        """Test batch size of 0 is rejected."""
        config = OpenAIEmbeddingConfig(api_key="sk-test", batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be between"):
            config.validate()

    def test_config_invalid_batch_size_over_limit(self):
        """Test batch size over 2048 is rejected."""
        config = OpenAIEmbeddingConfig(api_key="sk-test", batch_size=2049)
        with pytest.raises(ValueError, match="batch_size must be between"):
            config.validate()

    def test_config_invalid_timeout(self):
        """Test non-positive timeout is rejected."""
        config = OpenAIEmbeddingConfig(api_key="sk-test", timeout=0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            config.validate()

    def test_config_invalid_max_retries(self):
        """Test negative max_retries is rejected."""
        config = OpenAIEmbeddingConfig(api_key="sk-test", max_retries=-1)
        with pytest.raises(ValueError, match="max_retries cannot be negative"):
            config.validate()

    def test_config_invalid_dimension(self):
        """Test dimension mismatch is rejected."""
        config = OpenAIEmbeddingConfig(api_key="sk-test", dimension=768)
        with pytest.raises(ValueError, match="text-embedding-3-small always returns 1536"):
            config.validate()

    def test_config_defaults(self):
        """Test configuration defaults are sensible."""
        config = OpenAIEmbeddingConfig(api_key="sk-test")
        assert config.model == "text-embedding-3-small"
        assert config.dimension == 1536
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.batch_size == 100


class TestOpenAIEmbeddingInitialization:
    """Test OpenAIEmbedding initialization."""

    def test_init_valid_config(self):
        """Test initialization with valid config."""
        config = OpenAIEmbeddingConfig(api_key="sk-test123")
        provider = OpenAIEmbedding(config)
        assert provider.config == config
        assert provider.client is not None

    def test_init_invalid_config_raises(self):
        """Test initialization with invalid config raises error."""
        config = OpenAIEmbeddingConfig(api_key="")
        with pytest.raises(ValueError):
            OpenAIEmbedding(config)


def sdk_response(*vectors: list[float], shuffled: bool = False) -> MagicMock:
    """Build an OpenAI embeddings response; ``shuffled`` returns items out of order."""
    items = [MagicMock(embedding=vector, index=i) for i, vector in enumerate(vectors)]
    if shuffled:
        items.reverse()
    response = MagicMock()
    response.data = items
    response.usage = MagicMock(prompt_tokens=10)
    return response


def rate_limited() -> RateLimitError:
    return RateLimitError("Rate limited", response=MagicMock(), body={})


@pytest.fixture
def provider() -> OpenAIEmbedding:
    return OpenAIEmbedding(OpenAIEmbeddingConfig(api_key="sk-test", max_retries=3, batch_size=2))


@pytest.fixture(autouse=True)
def no_backoff_sleep():
    with patch(
        "aegis_platform.rag.embeddings.openai.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        yield sleep


class TestProtocolConformance:
    def test_is_an_embedding_provider(self, provider):
        assert isinstance(provider, EmbeddingProvider)

    def test_sdk_retries_disabled(self, provider):
        assert provider.client.max_retries == 0

    def test_model_property(self, provider):
        assert provider.model == "text-embedding-3-small"


class TestOpenAIEmbeddingTextEmbedding:
    """embed_text(str) -> np.ndarray, as RAGService calls it."""

    @pytest.mark.asyncio
    async def test_embed_text_returns_numpy_vector(self, provider):
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.return_value = sdk_response([0.1, 0.2, 0.3] * 512)

            vector = await provider.embed_text("  Hello world  ")

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (1536,)
        assert vector.dtype == np.float32
        assert create.call_args.kwargs == {
            "input": ["Hello world"],
            "model": "text-embedding-3-small",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["", "   \n\t  "])
    async def test_embed_text_rejects_empty(self, provider, text):
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await provider.embed_text(text)

    @pytest.mark.asyncio
    async def test_embed_special_characters(self, provider):
        special = "Hello! @#$%^&*() 你好 🚀 café"
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.return_value = sdk_response([0.1] * 1536)

            await provider.embed_text(special)

        assert create.call_args.kwargs["input"] == [special]


class TestOpenAIEmbeddingBatch:
    """embed_batch keeps order and length aligned with its input."""

    @pytest.mark.asyncio
    async def test_batch_splits_into_api_calls(self, provider):
        texts = ["a", "b", "c", "d", "e"]  # batch_size=2 -> 3 calls
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = [
                sdk_response([1.0], [2.0]),
                sdk_response([3.0], [4.0]),
                sdk_response([5.0]),
            ]

            vectors = await provider.embed_batch(texts)

        assert [v.tolist() for v in vectors] == [[1.0], [2.0], [3.0], [4.0], [5.0]]
        assert [c.kwargs["input"] for c in create.call_args_list] == [["a", "b"], ["c", "d"], ["e"]]

    @pytest.mark.asyncio
    async def test_more_than_100_texts_are_accepted(self):
        provider = OpenAIEmbedding(OpenAIEmbeddingConfig(api_key="sk-test"))  # batch_size=100
        texts = [f"chunk {i}" for i in range(250)]

        async def fake_create(input, model):
            return sdk_response(*([float(i)] for i in range(len(input))))

        with patch.object(provider.client.embeddings, "create", side_effect=fake_create) as create:
            vectors = await provider.embed_batch(texts)

        assert len(vectors) == 250
        assert create.call_count == 3

    @pytest.mark.asyncio
    async def test_out_of_order_api_items_are_realigned(self, provider):
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.return_value = sdk_response([1.0], [2.0], shuffled=True)

            vectors = await provider.embed_batch(["first", "second"])

        assert [v.tolist() for v in vectors] == [[1.0], [2.0]]

    @pytest.mark.asyncio
    async def test_empty_text_is_rejected_not_dropped(self, provider):
        """Dropping empties would shift every later vector onto the wrong chunk."""
        with pytest.raises(ValueError, match=r"positions \[1\]"):
            await provider.embed_batch(["a", "  ", "c"])

    @pytest.mark.asyncio
    async def test_empty_list_rejected(self, provider):
        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            await provider.embed_batch([])

    @pytest.mark.asyncio
    async def test_count_mismatch_from_api_is_an_error(self, provider):
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.return_value = sdk_response([1.0])

            with pytest.raises(RuntimeError, match="returned 1 vectors for 2 inputs"):
                await provider.embed_batch(["a", "b"])


class TestOpenAIEmbeddingRetry:
    @pytest.mark.asyncio
    async def test_rate_limit_retry_success(self, provider, no_backoff_sleep):
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = [rate_limited(), sdk_response([0.5] * 1536)]

            vector = await provider.embed_text("hello")

        assert vector.shape == (1536,)
        assert create.call_count == 2
        no_backoff_sleep.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_fast_without_raw_message(self, provider):
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = Exception("secret upstream detail")

            with pytest.raises(RuntimeError, match=r"Embedding failed \(Exception\)") as raised:
                await provider.embed_text("hello")

        assert "secret upstream detail" not in str(raised.value)
        assert create.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        provider = OpenAIEmbedding(OpenAIEmbeddingConfig(api_key="sk-test", max_retries=2))
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = rate_limited()

            with pytest.raises(RuntimeError, match="Failed to embed after 2 attempts"):
                await provider.embed_text("hello")

        assert create.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_max_retries_still_makes_one_attempt(self):
        provider = OpenAIEmbedding(OpenAIEmbeddingConfig(api_key="sk-test", max_retries=0))
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.return_value = sdk_response([0.1])

            await provider.embed_text("hello")

        assert create.call_count == 1


class TestOpenAIEmbeddingUtility:
    @pytest.mark.asyncio
    async def test_get_dimension(self, provider):
        assert await provider.get_dimension() == 1536

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.return_value = sdk_response([0.1] * 1536)

            assert await provider.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = Exception("Connection failed")

            assert await provider.health_check() is False


class TestQuotaExhausted:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            {"type": "insufficient_quota", "code": "credit_balance_exhausted"},
            {"error": {"type": "insufficient_quota"}},
        ],
    )
    async def test_quota_errors_fail_fast(self, provider, body):
        error = RateLimitError("quota", response=MagicMock(), body=body)
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = error

            with pytest.raises(RuntimeError, match="quota exhausted"):
                await provider.embed_text("hello")

        assert create.call_count == 1


class TestEmbedRequestResponse:
    """embed(EmbeddingRequest) -> EmbeddingResponse with usage (ADR-013)."""

    @pytest.mark.asyncio
    async def test_usage_is_summed_across_api_batches(self, provider):
        first, second = sdk_response([1.0], [2.0]), sdk_response([3.0])
        first.usage = MagicMock(prompt_tokens=7)
        second.usage = MagicMock(prompt_tokens=5)
        first.model = second.model = "text-embedding-3-small"
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = [first, second]

            response = await provider.embed(EmbeddingRequest(texts=["a", "b", "c"]))

        assert response.usage == EmbeddingUsage(input_tokens=12)
        assert response.provider == "openai"
        assert response.model == "text-embedding-3-small"
        assert response.dimension == 1
        assert [v.tolist() for v in response.embeddings] == [[1.0], [2.0], [3.0]]

    @pytest.mark.asyncio
    async def test_missing_usage_is_unknown_not_partial(self, provider):
        first, second = sdk_response([1.0], [2.0]), sdk_response([3.0])
        second.usage = None
        with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create:
            create.side_effect = [first, second]

            response = await provider.embed(EmbeddingRequest(texts=["a", "b", "c"]))

        assert response.usage is None

    @pytest.mark.asyncio
    async def test_wrappers_delegate_to_embed(self, provider):
        with (
            patch.object(provider, "embed", wraps=provider.embed) as embed,
            patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as create,
        ):
            create.side_effect = [sdk_response([1.0]), sdk_response([1.0], [2.0])]
            await provider.embed_text("q")
            await provider.embed_batch(["a", "b"])

        assert [c.args[0].input_type for c in embed.call_args_list] == ["query", "passage"]


def test_sum_usage() -> None:
    assert sum_usage([3, 4]) == EmbeddingUsage(input_tokens=7)
    assert sum_usage([3, None]) is None
    assert sum_usage([]) is None
