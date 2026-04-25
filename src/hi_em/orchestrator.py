"""Per-turn orchestration of embedding + segmentation + LTM + memory window + LLM.

A single ``HiEM`` instance handles one ``conv_id``. Resuming an existing
conversation is *not* supported in Phase 3-2: the segmenter is rebuilt fresh
each session (centroids in ``<conv_id>.state.json`` are written but never
loaded back to restore segmenter state). This is acceptable because Phase 4
evaluation runs whole conversations end-to-end in one process. Restoration
will be added in a later step if Phase 5 needs it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hi_em.embedding import QueryEncoder
from hi_em.llm import OpenAIChatLLM
from hi_em.ltm import LTM
from hi_em.memory_window import select_memory_window
from hi_em.sem_core import HiEMSegmenter


class HiEM:
    """Drives one conversation: each ``handle_turn`` is a full pipeline pass."""

    def __init__(
        self,
        conv_id: str,
        encoder: QueryEncoder,
        llm: OpenAIChatLLM,
        model: str,
        ltm_root: Path | str,
        alpha: float = 1.0,
        lmda: float = 10.0,
        sigma0_sq: float = 0.01,
        k_topics: int = 3,
        k_turns_per_topic: int = 5,
        system_prompt: str | None = None,
        response_filter: Callable[[str], str] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self.conv_id = conv_id
        self._encoder = encoder
        self._llm = llm
        self._model = model
        self._ltm = LTM(ltm_root)
        self._segmenter = HiEMSegmenter(
            dim=encoder.dim, alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq
        )
        self._k_topics = k_topics
        self._k_turns_per_topic = k_turns_per_topic
        self._system_prompt = system_prompt
        self._response_filter = response_filter
        self._llm_kwargs = llm_kwargs
        self._next_turn_id = 0

    def handle_turn(
        self, user_text: str, return_debug: bool = False
    ) -> str | tuple[str, dict[str, Any]]:
        """Process one user turn and return the assistant's response.

        With ``return_debug=True``, returns ``(response, debug)`` where
        ``debug`` contains ``topic_id`` (assigned), ``is_boundary``,
        ``prefill_turns`` (list[dict] from select_memory_window), and
        ``messages`` (the exact list sent to the LLM). Used by Phase 4
        evaluation for token-counting and topic-revisit metrics.

        Pipeline (see ``plan.md`` Phase 3-2 spec)::

            1. embed user_text                         → q
            2. segmenter.assign(q)                     → (topic_id, is_boundary)
            3. ltm.update_state(topic snapshot)
            4. select_memory_window (LTM only — current turn not yet written)
            5. messages = [system?] + prefill turns + current user turn
            6. llm.chat(messages, model)               → response
            7. ltm.append_turn(user)  +  ltm.append_turn(assistant)
        """
        # 1. embed
        q = np.asarray(self._encoder.encode([user_text])[0])

        # 2. segment (mutates segmenter state)
        topic_id, is_boundary = self._segmenter.assign(q)

        # 3. snapshot topic state
        self._ltm.update_state(self.conv_id, self._snapshot_state())

        # 4. memory window (current user turn intentionally absent from LTM here)
        prefill = select_memory_window(
            q, self._ltm, self.conv_id, self._k_topics, self._k_turns_per_topic
        )

        # 5. build messages
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend({"role": t["role"], "content": t["text"]} for t in prefill)
        messages.append({"role": "user", "content": user_text})

        # 6. LLM call (raw response returned to caller; filtered version goes to LTM)
        response = self._llm.chat(messages, model=self._model, **self._llm_kwargs)
        stored_response = (
            self._response_filter(response) if self._response_filter else response
        )

        # 7. persist user + assistant turns
        user_turn_id = self._next_turn_id
        self._ltm.append_turn(
            self.conv_id,
            self._make_turn(user_turn_id, "user", user_text, q.tolist(), topic_id, is_boundary),
        )
        self._next_turn_id += 1
        assistant_turn_id = self._next_turn_id
        self._ltm.append_turn(
            self.conv_id,
            self._make_turn(assistant_turn_id, "assistant", stored_response, None, topic_id, False),
        )
        self._next_turn_id += 1

        if return_debug:
            return response, {
                "topic_id": topic_id,
                "is_boundary": is_boundary,
                "prefill_turns": prefill,
                "messages": messages,
            }
        return response

    def preload_history(self, turns: list[dict[str, Any]]) -> None:
        """Inject pre-existing user/assistant turns into LTM without LLM calls.

        Used for benchmarks (LongMemEval, LoCoMo) where the conversation
        history is given upfront and only the final question goes through
        :meth:`handle_turn`. Only user turns are passed through the segmenter
        so that assistant text does not pollute topic centroids.

        Each input turn is a dict with at least ``{"role", "content"}``.
        Optional ``ts`` is preserved if present.
        """
        last_topic_id = 0
        for t in turns:
            role = t["role"]
            text = t["content"]
            ts = t.get("ts", datetime.now(timezone.utc).isoformat())
            if role == "user":
                q = np.asarray(self._encoder.encode([text])[0])
                topic_id, is_boundary = self._segmenter.assign(q)
                last_topic_id = topic_id
                self._ltm.append_turn(
                    self.conv_id,
                    {
                        "turn_id": self._next_turn_id,
                        "ts": ts,
                        "role": "user",
                        "text": text,
                        "embedding": q.tolist(),
                        "topic_id": topic_id,
                        "is_boundary": is_boundary,
                    },
                )
            else:  # assistant — no embedding, inherit prev user's topic
                self._ltm.append_turn(
                    self.conv_id,
                    {
                        "turn_id": self._next_turn_id,
                        "ts": ts,
                        "role": role,
                        "text": text,
                        "embedding": None,
                        "topic_id": last_topic_id,
                        "is_boundary": False,
                    },
                )
            self._next_turn_id += 1
        self._ltm.update_state(self.conv_id, self._snapshot_state())

    def _make_turn(
        self,
        turn_id: int,
        role: str,
        text: str,
        embedding: list[float] | None,
        topic_id: int,
        is_boundary: bool,
    ) -> dict[str, Any]:
        return {
            "turn_id": turn_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "text": text,
            "embedding": embedding,
            "topic_id": topic_id,
            "is_boundary": is_boundary,
        }

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "conv_id": self.conv_id,
            "n_turns": self._next_turn_id,
            "topics": [
                {
                    "topic_id": t.topic_id,
                    "centroid": t.mu.tolist(),
                    "variance": t.variance().tolist(),
                    "count": t.n,
                }
                for t in self._segmenter.topics
            ],
        }
