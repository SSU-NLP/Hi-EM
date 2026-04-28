"""OpenAI-compatible chat LLM adapter.

Single thin wrapper that works against any OpenAI Chat Completion API endpoint
— OpenRouter, vLLM, or OpenAI itself. Backend choice rationale:
``memory/project_llm_backend.md`` (project memory).

Config sources (precedence):
    1. Constructor args (``api_key``, ``base_url``).
    2. Env vars ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``.

The model name and sampling kwargs are passed at call time — no defaults are
baked in, so the caller stays in control of which model is used.

Streaming + per-call latency metrics
------------------------------------
``chat()`` always streams under the hood (``stream=True`` with
``stream_options.include_usage=True``) so we can record:

    * ``ttft_sec`` — Time To First Token: from request submission to the
      first content chunk.
    * ``tpot_sec`` — Time Per Output Token: ``(t_end - t_first)
      / max(1, output_tokens - 1)``.
    * ``output_tokens`` — completion token count from the final usage
      chunk (vLLM / OpenAI both populate this when
      ``stream_options.include_usage=True``).
    * ``gen_sec`` — total wall-clock for the streamed response.

Metrics from the most recent call on the **current thread** are exposed
via :attr:`last_call_metrics`. The wrapper keeps these in a
``threading.local`` so concurrent ``ThreadPoolExecutor`` workers each see
their own latest call. Callers read the property right after ``chat()``
returns; no API change to the return type.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from openai import OpenAI


class OpenAIChatLLM:
    """Stateless wrapper with streaming + per-thread latency metrics."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        self._tls = threading.local()

    @property
    def last_call_metrics(self) -> dict[str, float] | None:
        """Metrics from the most recent ``chat`` call **on this thread**.

        Keys: ``ttft_sec``, ``tpot_sec``, ``output_tokens``, ``gen_sec``.
        Returns ``None`` before the first call.
        """
        return getattr(self._tls, "last_metrics", None)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> str:
        """Return the first choice's text content (empty string if ``None``).

        Always streams; populates :attr:`last_call_metrics` after return.
        """
        # Force streaming — caller may have passed stream=False, override.
        kwargs.pop("stream", None)
        kwargs.pop("stream_options", None)

        t0 = time.perf_counter()
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        text_parts: list[str] = []
        t_first: float | None = None
        last_chunk = None
        for chunk in stream:
            last_chunk = chunk
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                text_parts.append(content)
        t_end = time.perf_counter()
        text = "".join(text_parts)

        n_out = 0
        usage = getattr(last_chunk, "usage", None)
        if usage is not None:
            n_out = int(getattr(usage, "completion_tokens", 0) or 0)

        ttft = (t_first - t0) if t_first is not None else (t_end - t0)
        if n_out > 1 and t_first is not None:
            tpot = (t_end - t_first) / (n_out - 1)
        else:
            tpot = 0.0

        self._tls.last_metrics = {
            "ttft_sec": float(ttft),
            "tpot_sec": float(tpot),
            "output_tokens": int(n_out),
            "gen_sec": float(t_end - t0),
        }
        return text
