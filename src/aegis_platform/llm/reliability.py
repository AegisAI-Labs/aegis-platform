"""Bounded provider retries and streaming-safe failure handling."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """max_attempts includes the initial attempt."""

    max_attempts: int = 3
    timeout_seconds: float = 30.0
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.initial_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")


class RetryingLLMProvider:
    """Decorate an existing provider; no cross-provider failover is performed."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        policy: RetryPolicy | None = None,
        retryable_exceptions: tuple[type[Exception], ...] = (TimeoutError,),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self._provider = provider
        self._policy = policy or RetryPolicy()
        self._retryable = tuple(dict.fromkeys((*retryable_exceptions, TimeoutError)))
        self._sleep = sleep
        self._random_value = random_value

    @property
    def provider_name(self) -> str:
        return str(getattr(self._provider, "provider_name", "unknown"))

    def _delay(self, failed_attempt: int) -> float:
        cap = min(
            self._policy.initial_delay_seconds * (2 ** (failed_attempt - 1)),
            self._policy.max_delay_seconds,
        )
        if not self._policy.jitter or cap == 0:
            return cap
        return cap * min(max(self._random_value(), 0.0), 1.0)

    def _retry_allowed(self, exc: Exception, attempt: int, deadline: float) -> bool:
        return (
            attempt < self._policy.max_attempts
            and isinstance(exc, self._retryable)
            and time.monotonic() < deadline
        )

    async def _wait(self, delay: float, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("LLM request deadline exceeded")
        await self._sleep(min(delay, remaining))

    async def _generate_with_retries(self, request: LLMRequest) -> LLMResponse:
        deadline = time.monotonic() + self._policy.timeout_seconds
        last_error: Exception | None = None

        for attempt in range(1, self._policy.max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LLM request deadline exceeded") from last_error
            try:
                async with asyncio.timeout(remaining):
                    return await self._provider.generate(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if not self._retry_allowed(exc, attempt, deadline):
                    if attempt > 1:
                        logger.warning(
                            "llm_request_failed",
                            extra={
                                "provider": self.provider_name,
                                "attempts": attempt,
                                "error_type": type(exc).__name__,
                            },
                        )
                    raise
                delay = self._delay(attempt)
                logger.warning(
                    "llm_retry",
                    extra={
                        "provider": self.provider_name,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "delay_seconds": round(delay, 4),
                    },
                )
                await self._wait(delay, deadline)

        if last_error:
            raise last_error
        raise RuntimeError("Retry loop exited unexpectedly")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self._generate_with_retries(request)

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def iterate() -> AsyncIterator[str]:
            deadline = time.monotonic() + self._policy.timeout_seconds
            last_error: Exception | None = None
            for attempt in range(1, self._policy.max_attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("LLM stream deadline exceeded") from last_error
                yielded = False
                try:
                    async with asyncio.timeout(remaining):
                        async for chunk in self._provider.stream(request):
                            yielded = True
                            yield chunk
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    if yielded or not self._retry_allowed(exc, attempt, deadline):
                        if attempt > 1:
                            logger.warning(
                                "llm_request_failed",
                                extra={
                                    "provider": self.provider_name,
                                    "attempts": attempt,
                                    "error_type": type(exc).__name__,
                                    "streaming": True,
                                    "partial_output_delivered": yielded,
                                },
                            )
                        raise
                    delay = self._delay(attempt)
                    logger.warning(
                        "llm_retry",
                        extra={
                            "provider": self.provider_name,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "error_type": type(exc).__name__,
                            "delay_seconds": round(delay, 4),
                            "streaming": True,
                        },
                    )
                    await self._wait(delay, deadline)

        return iterate()
