from collections.abc import AsyncIterator

from openai import AsyncOpenAI
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


class OpenAIProvider(LLMProvider):
    """
    OpenAI implementation of the Aegis LLMProvider contract.

     OpenAI-specific SDK objects must not leak outside this adapter.
    """

    provider_name = "openai"

    @staticmethod
    def _messages(request: LLMRequest) -> list[ChatCompletionMessageParam]:
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

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
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

        except Exception as exc:
            raise self._map_error(exc) from exc

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
            provider=self.provider_name,
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

        except Exception as exc:
            raise self._map_error(exc) from exc

    @staticmethod
    def _map_error(exc: Exception) -> Exception:
        """
        Translate provider-specific exceptions into Aegis
        provider-neutral exceptions.
        """

        error_name = type(exc).__name__.lower()

        if "authentication" in error_name:
            return ProviderAuthenticationError("OpenAI authentication failed.")

        if "rate" in error_name:
            return ProviderRateLimited("OpenAI rate limit exceeded.")

        if "timeout" in error_name:
            return ProviderTimeout("OpenAI request timed out.")

        if "connection" in error_name:
            return ProviderUnavailable("OpenAI provider is unavailable.")

        return ProviderRequestError(f"OpenAI request failed: {exc}")
