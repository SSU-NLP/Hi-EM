"""OpenAI-compatible chat LLM adapter.

Single thin wrapper that works against any OpenAI Chat Completion API endpoint
— OpenRouter, vLLM, or OpenAI itself. Backend choice rationale:
``memory/project_llm_backend.md`` (project memory).

Config sources (precedence):
    1. Constructor args (``api_key``, ``base_url``, ``stream``).
    2. Env vars ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
       ``HIEM_LLM_STREAM``.

The model name and sampling kwargs are passed at call time — no defaults are
baked in, so the caller stays in control of which model is used.

Streaming + per-call latency metrics
------------------------------------
``chat()`` defaults to streaming (``stream=True`` with
``stream_options.include_usage=True``) so we can record:

    * ``ttft_sec`` — Time To First Token: from request submission to the
      first content chunk.
    * ``tpot_sec`` — Time Per Output Token: ``(t_end - t_first)
      / max(1, output_tokens - 1)``.
    * ``output_tokens`` — completion token count from the final usage
      chunk (vLLM / OpenAI both populate this when
      ``stream_options.include_usage=True``).
    * ``gen_sec`` — total wall-clock for the streamed response.

Streaming can be disabled via ``HIEM_LLM_STREAM=false`` (or
``stream=False`` constructor arg) for backends that 500 on streamed chat
requests (current example: Crts proxy at ``api.ssunlp.co.kr``). In
non-streaming mode, ``ttft_sec`` and ``tpot_sec`` are reported as ``0.0``
since the metric requires per-chunk timing; ``output_tokens`` and
``gen_sec`` are still populated from the final ``usage`` and wall-clock.

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

import openai
from openai import OpenAI


class OpenAIChatLLM:
    """Stateless wrapper with streaming + per-thread latency metrics."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool | None = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        if stream is None:
            env_val = os.environ.get("HIEM_LLM_STREAM", "true").strip().lower()
            stream = env_val not in {"false", "0", "no", "off"}
        self._stream = bool(stream)
        self._tls = threading.local()

    @property
    def last_call_metrics(self) -> dict[str, float] | None:
        """Metrics from the most recent ``chat`` call **on this thread**.

        Keys: ``ttft_sec``, ``tpot_sec``, ``output_tokens``, ``gen_sec``.
        Returns ``None`` before the first call.
        """
        return getattr(self._tls, "last_metrics", None)

    # Retry transient 5xx / timeout from the upstream provider.
    # Crts → Together pipeline emits 503 ("service_unavailable") under
    # short-lived load spikes; the upstream itself signals
    # ``retry_after_seconds`` ≈ 1 in the error body. We back off
    # 1s, 3s, 9s (max 3 retries) before surfacing the error.
    _RETRY_BACKOFFS_SEC: tuple[float, ...] = (1.0, 3.0, 9.0)
    _RETRY_EXCEPTIONS: tuple = (
        openai.InternalServerError,  # 5xx (503/504/etc.)
        openai.APITimeoutError,
        openai.APIConnectionError,
    )

    def _create_with_retry(self, **kwargs):
        for attempt in range(len(self._RETRY_BACKOFFS_SEC) + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except self._RETRY_EXCEPTIONS:
                if attempt >= len(self._RETRY_BACKOFFS_SEC):
                    raise
                time.sleep(self._RETRY_BACKOFFS_SEC[attempt])

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> str:
        """Return the first choice's text content (empty string if ``None``).

        Streams when ``self._stream`` is true (default); populates
        :attr:`last_call_metrics` after return.
        """
        # Caller-provided stream flags are ignored — we own that decision.
        kwargs.pop("stream", None)
        kwargs.pop("stream_options", None)

        if not self._stream:
            t0 = time.perf_counter()
            resp = self._create_with_retry(
                model=model,
                messages=messages,
                **kwargs,
            )
            t_end = time.perf_counter()
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            n_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            self._tls.last_metrics = {
                "ttft_sec": 0.0,
                "tpot_sec": 0.0,
                "output_tokens": n_out,
                "gen_sec": float(t_end - t0),
            }
            return text

        t0 = time.perf_counter()
        stream = self._create_with_retry(
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
