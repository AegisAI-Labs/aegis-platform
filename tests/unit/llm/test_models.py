from dataclasses import FrozenInstanceError

import pytest

from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse, LLMUsage


def test_request_defaults_and_message() -> None:
    message = LLMMessage(role="user", content="Hello")
    request = LLMRequest(messages=[message])

    assert request.messages == [message]
    assert request.model is None
    assert request.temperature is None
    assert request.max_tokens is None

    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]


def test_response_usage_and_independent_metadata() -> None:
    usage = LLMUsage(input_tokens=2, output_tokens=3, total_tokens=5)
    first = LLMResponse(content="answer", model="example", provider="stub", usage=usage)
    second = LLMResponse(content="other", model="example", provider="stub")

    first.metadata["id"] = 1
    assert first.usage == usage
    assert second.usage is None
    assert second.metadata == {}
