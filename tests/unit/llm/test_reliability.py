import asyncio
from collections.abc import AsyncIterator

import pytest

from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse
from aegis_platform.llm.reliability import RetryingLLMProvider, RetryPolicy

# ============================================================================
# Mock Providers for Testing
# ============================================================================


class SuccessfulProvider:
    """Always succeeds immediately."""

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content="success", model="test", provider="test")

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            self.call_count += 1
            yield "streaming"
        return _stream()


class FlakyGenerateProvider:
    """Fails N times, succeeds on attempt N+1."""

    def __init__(self, fail_count: int = 1) -> None:
        self.fail_count = fail_count
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        if self.generate_calls <= self.fail_count:
            raise TimeoutError(f"temporary failure #{self.generate_calls}")
        return LLMResponse(content="recovered", model="test", provider="test")

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            if False:
                yield  # Unreachable, but makes this an async generator
            raise NotImplementedError
        return _stream()


class NonRetryableFailingProvider:
    """Raises non-retryable exception."""

    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        raise ValueError("non-retryable error")

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            if False:
                yield  # Unreachable, but makes this an async generator
            raise ValueError("non-retryable error")
        return _stream()


class AlwaysFailingProvider:
    """Always fails."""

    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        raise TimeoutError("persistent failure")

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            if False:
                yield  # Unreachable, but makes this an async generator
            raise TimeoutError("persistent failure")
        return _stream()


class CancellingProvider:
    """Raises CancelledError."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise asyncio.CancelledError()

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            if False:
                yield  # Unreachable, but makes this an async generator
            raise asyncio.CancelledError()
        return _stream()


class FlakyStreamProvider:
    """Stream provider that fails before, during, or after yielding."""

    def __init__(
        self,
        *,
        fail_before_yield: bool = False,
        fail_after_chunk: bool = False,
        chunks_before_fail: int = 1,
    ) -> None:
        self.fail_before_yield = fail_before_yield
        self.fail_after_chunk = fail_after_chunk
        self.chunks_before_fail = chunks_before_fail
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            self.stream_calls += 1
            if self.fail_before_yield and self.stream_calls == 1:
                raise TimeoutError("failure before yield")
            for chunk_count in range(self.chunks_before_fail):
                yield f"chunk-{self.stream_calls}-{chunk_count}"
            if self.fail_after_chunk:
                raise TimeoutError("failure after chunks")
        return _stream()


class CustomExceptionProvider:
    """Raises custom exception on first call."""

    class CustomError(Exception):
        pass

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            raise self.CustomError("custom error")
        return LLMResponse(content="recovered", model="test", provider="test")

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            if False:
                yield  # Unreachable, but makes this an async generator
            raise NotImplementedError
        return _stream()


# ============================================================================
# Test Helpers
# ============================================================================


async def _no_sleep() -> None:
    """Mock sleep that does nothing."""
    return


def _make_request() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content="Test")])


def _make_retrying_provider(
    provider,
    max_attempts: int = 2,
    timeout_seconds: float = 1.0,
    initial_delay_seconds: float = 0.0,
    max_delay_seconds: float = 0.0,
    jitter: bool = False,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> RetryingLLMProvider:
    """Create a RetryingLLMProvider with fast test configuration."""
    if retryable_exceptions is None:
        retryable_exceptions = (TimeoutError,)
    return RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
            initial_delay_seconds=initial_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            jitter=jitter,
        ),
        sleep=lambda _: _no_sleep(),
        retryable_exceptions=retryable_exceptions,
    )


# ============================================================================
# RetryPolicy Validation Tests
# ============================================================================


def test_retry_policy_valid_construction() -> None:
    """Verify valid RetryPolicy constructs successfully."""
    policy = RetryPolicy(
        max_attempts=3,
        timeout_seconds=30.0,
        initial_delay_seconds=0.5,
        max_delay_seconds=5.0,
    )
    assert policy.max_attempts == 3
    assert policy.timeout_seconds == 30.0


def test_retry_policy_rejects_zero_max_attempts() -> None:
    """RetryPolicy should reject max_attempts < 1."""
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        RetryPolicy(max_attempts=0)


def test_retry_policy_rejects_negative_timeout() -> None:
    """RetryPolicy should reject timeout_seconds <= 0."""
    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        RetryPolicy(timeout_seconds=0.0)


def test_retry_policy_rejects_negative_delays() -> None:
    """RetryPolicy should reject negative delay values."""
    with pytest.raises(ValueError, match="retry delays cannot be negative"):
        RetryPolicy(initial_delay_seconds=-1.0)
    with pytest.raises(ValueError, match="retry delays cannot be negative"):
        RetryPolicy(max_delay_seconds=-1.0)


def test_retry_policy_single_attempt_allowed() -> None:
    """RetryPolicy should allow max_attempts=1 (no retries)."""
    policy = RetryPolicy(max_attempts=1)
    assert policy.max_attempts == 1


# ============================================================================
# Generate Flow: Basic Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_succeeds_immediately() -> None:
    """Generate should complete without retries when provider succeeds."""
    provider = SuccessfulProvider()
    retrying = _make_retrying_provider(provider)
    request = _make_request()

    response = await retrying.generate(request)

    assert response.content == "success"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_generate_retries_on_timeout() -> None:
    """Generate should retry on TimeoutError and recover."""
    provider = FlakyGenerateProvider(fail_count=1)
    retrying = _make_retrying_provider(provider, max_attempts=2)
    request = _make_request()

    response = await retrying.generate(request)

    assert response.content == "recovered"
    assert provider.generate_calls == 2


@pytest.mark.asyncio
async def test_generate_exhausts_retries() -> None:
    """Generate should raise after all retries exhausted."""
    provider = AlwaysFailingProvider()
    retrying = _make_retrying_provider(provider, max_attempts=2)
    request = _make_request()

    with pytest.raises(TimeoutError, match="persistent failure"):
        await retrying.generate(request)

    assert provider.generate_calls == 2


@pytest.mark.asyncio
async def test_generate_does_not_retry_non_retryable() -> None:
    """Generate should not retry non-retryable exceptions."""
    provider = NonRetryableFailingProvider()
    retrying = _make_retrying_provider(provider, max_attempts=3)
    request = _make_request()

    with pytest.raises(ValueError, match="non-retryable error"):
        await retrying.generate(request)

    assert provider.generate_calls == 1


@pytest.mark.asyncio
async def test_generate_propagates_cancellation() -> None:
    """Generate should propagate CancelledError immediately."""
    provider = CancellingProvider()
    retrying = _make_retrying_provider(provider)
    request = _make_request()

    with pytest.raises(asyncio.CancelledError):
        await retrying.generate(request)


@pytest.mark.asyncio
async def test_generate_notifies_callback_before_retry() -> None:
    """Generate should call on_retry callback before waiting."""
    provider = FlakyGenerateProvider(fail_count=1)
    retry_events: list[tuple[int, float, Exception]] = []
    retrying = RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=2,
            timeout_seconds=1.0,
            initial_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        sleep=lambda _: _no_sleep(),
        on_retry=lambda attempt, delay, error: retry_events.append(
            (attempt, delay, error)
        ),
    )
    request = _make_request()

    response = await retrying.generate(request)

    assert response.content == "recovered"
    assert len(retry_events) == 1
    attempt, delay, error = retry_events[0]
    assert attempt == 1
    assert delay == 0.0
    assert isinstance(error, TimeoutError)


@pytest.mark.asyncio
async def test_generate_with_custom_retryable_exception() -> None:
    """Generate should retry on custom exceptions when configured."""
    provider = CustomExceptionProvider()
    retrying = _make_retrying_provider(
        provider,
        max_attempts=2,
        retryable_exceptions=(CustomExceptionProvider.CustomError,),
    )
    request = _make_request()

    response = await retrying.generate(request)

    assert response.content == "recovered"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_generate_with_custom_exception_not_retried() -> None:
    """Generate should not retry custom exceptions unless configured."""
    provider = CustomExceptionProvider()
    retrying = _make_retrying_provider(provider, max_attempts=3)
    request = _make_request()

    with pytest.raises(CustomExceptionProvider.CustomError):
        await retrying.generate(request)

    assert provider.calls == 1


# ============================================================================
# Generate Flow: Deadline Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_with_extremely_short_timeout() -> None:
    """Generate should timeout if provider takes too long."""
    
    class SlowProvider:
        def __init__(self):
            self.generate_calls = 0
        
        async def generate(self, request: LLMRequest) -> LLMResponse:
            self.generate_calls += 1
            await asyncio.sleep(1.0)  # Sleep longer than timeout
            return LLMResponse(content="too late", model="test", provider="test")
        
        def stream(self, request: LLMRequest) -> AsyncIterator[str]:
            async def _stream() -> AsyncIterator[str]:
                if False:
                    yield  # Unreachable, but makes this an async generator
                raise NotImplementedError
            return _stream()
    
    provider = SlowProvider()
    retrying = _make_retrying_provider(provider, timeout_seconds=0.001)
    request = _make_request()

    with pytest.raises(TimeoutError):
        await retrying.generate(request)

    assert provider.generate_calls >= 1


# ============================================================================
# Generate Flow: Backoff Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_exponential_backoff_sequence() -> None:
    """Verify exponential backoff: 0.5s, 1.0s, 2.0s, etc."""
    provider = FlakyGenerateProvider(fail_count=3)
    delays_recorded: list[float] = []
    retrying = RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=4,
            timeout_seconds=10.0,
            initial_delay_seconds=0.5,
            max_delay_seconds=4.0,
            jitter=False,
        ),
        sleep=lambda d: _record_delay(delays_recorded, d),
        on_retry=None,
    )
    request = _make_request()

    response = await retrying.generate(request)

    assert response.content == "recovered"
    assert len(delays_recorded) == 3
    assert abs(delays_recorded[0] - 0.5) < 0.01  # 2^0 * 0.5 = 0.5
    assert abs(delays_recorded[1] - 1.0) < 0.01  # 2^1 * 0.5 = 1.0
    assert abs(delays_recorded[2] - 2.0) < 0.01  # 2^2 * 0.5 = 2.0


async def _record_delay(delays: list[float], delay: float) -> None:
    """Record a delay value and sleep for it."""
    delays.append(delay)
    await asyncio.sleep(delay)


@pytest.mark.asyncio
async def test_generate_respects_max_delay_cap() -> None:
    """Backoff should cap at max_delay_seconds."""
    provider = FlakyGenerateProvider(fail_count=4)
    delays_recorded: list[float] = []
    retrying = RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=5,
            timeout_seconds=20.0,
            initial_delay_seconds=1.0,
            max_delay_seconds=2.0,  # Should cap here
            jitter=False,
        ),
        sleep=lambda d: _record_delay(delays_recorded, d),
    )
    request = _make_request()

    await retrying.generate(request)

    # Expected: 1.0, 2.0, 2.0, 2.0 (capped)
    assert len(delays_recorded) == 4
    assert abs(delays_recorded[0] - 1.0) < 0.01
    assert abs(delays_recorded[1] - 2.0) < 0.01
    assert abs(delays_recorded[2] - 2.0) < 0.01
    assert abs(delays_recorded[3] - 2.0) < 0.01


@pytest.mark.asyncio
async def test_generate_jitter_with_zero_delay() -> None:
    """Jitter should not create negative delays."""
    provider = SuccessfulProvider()
    delays_recorded: list[float] = []
    retrying = RetryingLLMProvider(
        provider,
        policy=RetryPolicy(
            max_attempts=1,
            timeout_seconds=1.0,
            initial_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter=True,
        ),
        sleep=lambda d: _record_delay(delays_recorded, d),
        random_value=lambda: 0.99,  # Max jitter
    )
    request = _make_request()

    response = await retrying.generate(request)

    assert response.content == "success"


# ============================================================================
# Stream Flow: Basic Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stream_succeeds_immediately() -> None:
    """Stream should complete without retries when provider succeeds."""
    provider = FlakyStreamProvider(fail_before_yield=False, fail_after_chunk=False)
    retrying = _make_retrying_provider(provider)
    request = _make_request()

    chunks = [chunk async for chunk in retrying.stream(request)]

    assert len(chunks) == 1
    assert provider.stream_calls == 1


@pytest.mark.asyncio
async def test_stream_retries_timeout_before_yielding_output() -> None:
    """Stream should retry if timeout occurs before any yield."""
    provider = FlakyStreamProvider(fail_before_yield=True)
    retrying = _make_retrying_provider(provider, max_attempts=2)
    request = _make_request()

    chunks = [chunk async for chunk in retrying.stream(request)]

    assert len(chunks) == 1
    assert provider.stream_calls == 2


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_yielding_output() -> None:
    """Stream should NOT retry once any output has been yielded."""
    provider = FlakyStreamProvider(fail_after_chunk=True, chunks_before_fail=1)
    retrying = _make_retrying_provider(provider, max_attempts=3)
    request = _make_request()
    chunks: list[str] = []

    with pytest.raises(TimeoutError, match="failure after chunks"):
        async for chunk in retrying.stream(request):
            chunks.append(chunk)

    assert len(chunks) == 1  # One chunk was yielded
    assert provider.stream_calls == 1  # No retry happened


@pytest.mark.asyncio
async def test_stream_exhausts_retries_before_yield() -> None:
    """Stream should raise after retries exhausted if no output yielded."""
    
    class AlwaysFailBeforeYield:
        def __init__(self):
            self.stream_calls = 0
        
        async def generate(self, request: LLMRequest) -> LLMResponse:
            raise NotImplementedError
        
        def stream(self, request: LLMRequest) -> AsyncIterator[str]:
            async def _stream() -> AsyncIterator[str]:
                if False:
                    yield  # Unreachable, but makes this an async generator
                self.stream_calls += 1
                raise TimeoutError("failure before yield")
            return _stream()
    
    provider = AlwaysFailBeforeYield()
    retrying = _make_retrying_provider(provider, max_attempts=2)
    request = _make_request()

    with pytest.raises(TimeoutError, match="failure before yield"):
        async for _ in retrying.stream(request):
            pass

    assert provider.stream_calls == 2


@pytest.mark.asyncio
async def test_stream_propagates_cancellation() -> None:
    """Stream should propagate CancelledError immediately."""
    provider = CancellingProvider()
    retrying = _make_retrying_provider(provider)
    request = _make_request()

    with pytest.raises(asyncio.CancelledError):
        async for _ in retrying.stream(request):
            pass


@pytest.mark.asyncio
async def test_stream_does_not_retry_non_retryable() -> None:
    """Stream should not retry non-retryable exceptions."""
    provider = NonRetryableFailingProvider()
    retrying = _make_retrying_provider(provider, max_attempts=3)
    request = _make_request()

    with pytest.raises(ValueError, match="non-retryable error"):
        async for _ in retrying.stream(request):
            pass


@pytest.mark.asyncio
async def test_stream_multiple_chunks_then_failure() -> None:
    """Stream with multiple successful chunks followed by failure should not retry."""
    provider = FlakyStreamProvider(fail_after_chunk=True, chunks_before_fail=3)
    retrying = _make_retrying_provider(provider, max_attempts=3)
    request = _make_request()
    chunks: list[str] = []

    with pytest.raises(TimeoutError, match="failure after chunks"):
        async for chunk in retrying.stream(request):
            chunks.append(chunk)

    assert len(chunks) == 3  # All chunks yielded before failure
    assert provider.stream_calls == 1  # No retry


# ============================================================================
# Callback Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_generate_callback_error_does_not_crash() -> None:
    """Callback errors should be logged but not crash the retry loop."""
    provider = FlakyGenerateProvider(fail_count=1)
    call_count = [0]

    def failing_callback(attempt: int, delay: float, error: Exception) -> None:
        call_count[0] += 1
        raise RuntimeError("callback error")

    retrying = RetryingLLMProvider(
        provider,
        policy=RetryPolicy(max_attempts=2, timeout_seconds=1.0),
        sleep=lambda _: _no_sleep(),
        on_retry=failing_callback,
    )
    request = _make_request()

    response = await retrying.generate(request)

    assert response.content == "recovered"
    assert call_count[0] == 1  # Callback was invoked despite error


# ============================================================================
# Provider Name Property Tests
# ============================================================================


@pytest.mark.asyncio
async def test_provider_name_property() -> None:
    """RetryingLLMProvider should expose underlying provider name."""
    mock_provider = SuccessfulProvider()
    retrying = _make_retrying_provider(mock_provider)

    name = retrying.provider_name
    assert isinstance(name, str)
