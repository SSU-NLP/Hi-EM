"""Unit tests for ``hi_em.llm.OpenAIChatLLM``.

The OpenAI client is mocked — no real network calls. Smoke tests against an
actual OpenRouter / vLLM endpoint live in Phase 3-3.

``chat()`` always streams (``stream=True`` + ``stream_options.include_usage``)
so we can capture per-call TTFT/TPOT and report streaming-mode latency
metrics in evaluation reports. Tests exercise the streamed-iterator
contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hi_em.llm import OpenAIChatLLM


def _make_chunk(content: str | None = None, usage_completion_tokens: int | None = None
                ) -> MagicMock:
    """Build a single SSE chunk shaped like ``ChatCompletionChunk``."""
    chunk = MagicMock()
    if content is not None:
        delta = MagicMock(content=content)
        chunk.choices = [MagicMock(delta=delta)]
    else:
        chunk.choices = []
    if usage_completion_tokens is not None:
        chunk.usage = MagicMock(completion_tokens=usage_completion_tokens)
    else:
        chunk.usage = None
    return chunk


def _mock_stream(text: str, output_tokens: int | None = None) -> list[MagicMock]:
    """Stream that emits one chunk per character then a final usage chunk."""
    out = [_make_chunk(content=ch) for ch in text]
    if output_tokens is not None:
        out.append(_make_chunk(content=None, usage_completion_tokens=output_tokens))
    return out


@pytest.fixture
def patched_openai():
    with patch("hi_em.llm.OpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield mock_cls, client


def test_constructor_uses_explicit_args(patched_openai) -> None:
    mock_cls, _ = patched_openai
    OpenAIChatLLM(api_key="explicit-key", base_url="https://example.com/v1")
    mock_cls.assert_called_once_with(
        api_key="explicit-key", base_url="https://example.com/v1"
    )


def test_constructor_falls_back_to_env(patched_openai, monkeypatch) -> None:
    mock_cls, _ = patched_openai
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com/v1")
    OpenAIChatLLM()
    mock_cls.assert_called_once_with(
        api_key="env-key", base_url="https://env.example.com/v1"
    )


def test_chat_returns_concatenated_streamed_content(patched_openai) -> None:
    _, client = patched_openai
    client.chat.completions.create.return_value = iter(_mock_stream("hello world"))
    llm = OpenAIChatLLM(api_key="k")
    out = llm.chat([{"role": "user", "content": "hi"}], model="gpt-4o")
    assert out == "hello world"


def test_chat_passes_model_messages_and_kwargs(patched_openai) -> None:
    _, client = patched_openai
    client.chat.completions.create.return_value = iter(_mock_stream("ok"))
    llm = OpenAIChatLLM(api_key="k")
    msgs = [{"role": "user", "content": "hi"}]
    llm.chat(msgs, model="anthropic/claude-3.5", temperature=0.2, max_tokens=128)
    # Must enable streaming (we captured TTFT/TPOT per call) and forward
    # caller kwargs (temperature, max_tokens, etc.).
    client.chat.completions.create.assert_called_once_with(
        model="anthropic/claude-3.5",
        messages=msgs,
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.2,
        max_tokens=128,
    )


def test_chat_handles_empty_stream(patched_openai) -> None:
    _, client = patched_openai
    client.chat.completions.create.return_value = iter([])
    llm = OpenAIChatLLM(api_key="k")
    assert llm.chat([{"role": "user", "content": "hi"}], model="m") == ""


def test_chat_records_per_call_metrics(patched_openai) -> None:
    _, client = patched_openai
    client.chat.completions.create.return_value = iter(
        _mock_stream("abcd", output_tokens=4)
    )
    llm = OpenAIChatLLM(api_key="k")
    llm.chat([{"role": "user", "content": "hi"}], model="m")
    metrics = llm.last_call_metrics
    assert metrics is not None
    assert {"ttft_sec", "tpot_sec", "output_tokens", "gen_sec"} <= set(metrics)
    assert metrics["output_tokens"] == 4
    assert metrics["ttft_sec"] >= 0
    assert metrics["gen_sec"] >= metrics["ttft_sec"]


def test_chat_drops_caller_stream_flags(patched_openai) -> None:
    """Even if a caller passes stream=False or its own stream_options, we
    override — streaming is required for our metrics path."""
    _, client = patched_openai
    client.chat.completions.create.return_value = iter(_mock_stream("ok"))
    llm = OpenAIChatLLM(api_key="k")
    llm.chat(
        [{"role": "user", "content": "hi"}], model="m",
        stream=False, stream_options={"include_usage": False},
    )
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
