"""Unit tests for ``hi_em.llm.OpenAIChatLLM``.

The OpenAI client is mocked — no real network calls. Smoke tests against an
actual OpenRouter / vLLM endpoint live in Phase 3-3.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hi_em.llm import OpenAIChatLLM


def _mock_response(content: str | None = "ok") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


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


def test_chat_returns_first_choice_content(patched_openai) -> None:
    _, client = patched_openai
    client.chat.completions.create.return_value = _mock_response("hello world")
    llm = OpenAIChatLLM(api_key="k")
    out = llm.chat([{"role": "user", "content": "hi"}], model="gpt-4o")
    assert out == "hello world"


def test_chat_passes_model_messages_and_kwargs(patched_openai) -> None:
    _, client = patched_openai
    client.chat.completions.create.return_value = _mock_response("ok")
    llm = OpenAIChatLLM(api_key="k")
    msgs = [{"role": "user", "content": "hi"}]
    llm.chat(msgs, model="anthropic/claude-3.5", temperature=0.2, max_tokens=128)
    client.chat.completions.create.assert_called_once_with(
        model="anthropic/claude-3.5",
        messages=msgs,
        temperature=0.2,
        max_tokens=128,
    )


def test_chat_handles_none_content(patched_openai) -> None:
    _, client = patched_openai
    client.chat.completions.create.return_value = _mock_response(None)
    llm = OpenAIChatLLM(api_key="k")
    assert llm.chat([{"role": "user", "content": "hi"}], model="m") == ""
