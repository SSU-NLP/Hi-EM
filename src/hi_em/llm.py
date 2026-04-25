"""OpenAI-compatible chat LLM adapter.

Single thin wrapper that works against any OpenAI Chat Completion API endpoint
— OpenRouter, vLLM, or OpenAI itself. Backend choice rationale:
``memory/project_llm_backend.md`` (project memory).

Config sources (precedence):
    1. Constructor args (``api_key``, ``base_url``).
    2. Env vars ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``.

The model name and sampling kwargs are passed at call time — no defaults are
baked in, so the caller stays in control of which model is used.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


class OpenAIChatLLM:
    """Stateless wrapper around ``openai.OpenAI().chat.completions.create``."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> str:
        """Return the first choice's text content (empty string if ``None``)."""
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
