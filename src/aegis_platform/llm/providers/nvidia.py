from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from ..base import LLMProvider
from ..errors import (
    ProviderAuthenticationError,
    ProviderRateLimited,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
)
from ..models import LLMRequest, LLMResponse, LLMUsage


class NvidiaProvider(LLMProvider):
    """Nvidia LLM provider implementation."""

    DEFAULT_BASE_URL = (
        "https://integrate.api.nvidia.com/v1"  # Example default base URL for Nvidia API
    )

    def __init__(self, api_key: str, model: str, timeout_seconds: float):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.DEFAULT_BASE_URL,
            timeout=timeout_seconds,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        try:
            response = await self.client.chat.completions.create(
                model=request.model or self.model,
                messages=self._messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )

        except OpenAIError as exc:
            raise self._map_error(exc) from exc

        if not response.choices:
            raise ProviderRequestError("NVIDIA returned no completion choices.")

        choice = response.choices[0]

        usage = None

        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider="nvidia",
            usage=usage,
        )

    async def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[str]:
        try:
            response = await self.client.chat.completions.create(
                model=request.model or self.model,
                messages=self._messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:
                    yield delta.content

        except OpenAIError as exc:
            raise self._map_error(exc) from exc

    @staticmethod
    def _messages(request: LLMRequest) -> list[ChatCompletionMessageParam]:
        """Convert Aegis messages to the SDK's typed chat-message format."""
        # Initialize the list of messages to be sent to the Nvidia API.

        messages: list[ChatCompletionMessageParam] = []
        for message in request.messages:
            if message.role == "system":
                messages.append({"role": "system", "content": message.content})
            elif message.role == "developer":
                messages.append({"role": "developer", "content": message.content})
            elif message.role == "user":
                messages.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                messages.append({"role": "assistant", "content": message.content})
            else:
                raise ValueError(f"Unsupported OpenAI message role: {message.role}")
        return messages

    @staticmethod
    def _map_error(exc: OpenAIError) -> Exception:
        """Map SDK errors to Aegis provider-neutral exceptions."""

        if isinstance(exc, AuthenticationError):
            return ProviderAuthenticationError("NVIDIA authentication failed.")

        if isinstance(exc, RateLimitError):
            return ProviderRateLimited("NVIDIA rate limit exceeded.")

        if isinstance(exc, APITimeoutError):
            return ProviderTimeout("NVIDIA request timed out.")

        if isinstance(exc, APIConnectionError):
            return ProviderUnavailable("NVIDIA provider is unavailable.")

        return ProviderRequestError(f"NVIDIA request failed ({type(exc).__name__}).")
