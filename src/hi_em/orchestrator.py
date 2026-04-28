"""Per-turn orchestration of embedding + segmentation + LTM + memory window + LLM.

A single ``HiEM`` instance handles one ``conv_id``.

Two memory regimes:

* ``use_stm=False`` (default; Phase 2 baseline) — every turn calls the
  stateless :func:`select_memory_window` over LTM (cosine top-k topics ×
  recency top-k turns).
* ``use_stm=True`` (Phase 2-Full) — STM is the working buffer + cache:

    - **Every turn lands in STM** (per spec: "단기 메모리에 모든 대화 원문
      저장"). New topic? Create a new STM entry seeded with this turn pair.
      Existing topic in STM? Append in-sync. Topic atomicity preserved.
    - LTM also gets a sync write every turn (dual-write; current = target
      per phase-2-full-design.md §0.1 "LTM 쓰기 시점: 매 턴 sync").
    - Within a round: STM membership is fixed except (a) cache miss for the
      current topic triggers a one-time promotion from LTM, (b) the just-
      finished user/assistant pair is appended to its topic in STM.
    - At round boundary (every ``round_size`` user turns = ``2*round_size``
      jsonl rows): :class:`RoundProcessor` recomputes normalized importance,
      promotes topics ≥ threshold (refreshing with full LTM contents), and
      evicts to capacity. Runs on a daemon thread by default.

Resuming an existing conversation is *not* supported here: the segmenter
is rebuilt fresh each session (centroids in ``<conv_id>.state.json`` are
written but never loaded back). Phase 4 evaluation runs whole conversations
end-to-end in one process.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hi_em.embedding import QueryEncoder
from hi_em.llm import OpenAIChatLLM
from hi_em.ltm import LTM
from hi_em.memory_window import MemoryWindow, select_memory_window
from hi_em.round_processor import RoundProcessor
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
        # ---- Phase 2-Full STM ---------------------------------------
        use_stm: bool = False,
        round_size: int = 10,
        stm_max_topics: int = 10,
        stm_max_turns: int = 200,
        promotion_threshold: float = 0.5,
        importance_alpha: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        lambda_r: float = 0.5,
        lambda_freq: float = 0.5,
        min_floor: float = 0.1,
        round_async: bool = True,
        # -------------------------------------------------------------
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

        # Phase 2-Full STM wiring
        self._use_stm = use_stm
        self._round_size = round_size
        self._round_async = round_async
        if use_stm:
            self._stm: MemoryWindow | None = MemoryWindow(
                max_topics=stm_max_topics, max_turns=stm_max_turns
            )
            self._round_processor: RoundProcessor | None = RoundProcessor(
                conv_id=conv_id,
                ltm=self._ltm,
                stm=self._stm,
                threshold=promotion_threshold,
                alpha=importance_alpha,
                lambda_r=lambda_r,
                lambda_freq=lambda_freq,
                min_floor=min_floor,
            )
        else:
            self._stm = None
            self._round_processor = None

    # ------------------------------------------------------------------
    # Public properties (debug / tests)
    # ------------------------------------------------------------------

    @property
    def stm(self) -> MemoryWindow | None:
        return self._stm

    @property
    def round_processor(self) -> RoundProcessor | None:
        return self._round_processor

    # ------------------------------------------------------------------
    # Main per-turn entry
    # ------------------------------------------------------------------

    def handle_turn(
        self, user_text: str, return_debug: bool = False
    ) -> str | tuple[str, dict[str, Any]]:
        """Process one user turn and return the assistant's response.

        With ``return_debug=True``, returns ``(response, debug)`` where
        ``debug`` includes ``topic_id``, ``is_boundary``, ``prefill_turns``,
        ``messages``, and—when ``use_stm`` is on—``stm_hit`` (bool) and
        ``round_triggered`` (bool).
        """
        # 1. embed
        q = np.asarray(self._encoder.encode([user_text])[0])

        # 2. segment (mutates segmenter state)
        topic_id, is_boundary = self._segmenter.assign(q)

        # 3. snapshot topic state
        self._ltm.update_state(self.conv_id, self._snapshot_state())

        # 4. memory window — STM-first if enabled, else stateless baseline
        stm_hit: bool | None = None
        if self._stm is not None:
            stm_hit = self._stm.has(topic_id)
            if not stm_hit:
                ltm_turns = self._ltm.load_turns(self.conv_id, topic_id=topic_id)
                if ltm_turns:
                    self._stm.promote(topic_id, ltm_turns)
            prefill = self._stm.all_turns()
        else:
            prefill = select_memory_window(
                q, self._ltm, self.conv_id, self._k_topics, self._k_turns_per_topic
            )

        # 5. build messages
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend({"role": t["role"], "content": t["text"]} for t in prefill)
        messages.append({"role": "user", "content": user_text})

        # 6. LLM call (raw to caller, filtered to LTM)
        response = self._llm.chat(messages, model=self._model, **self._llm_kwargs)
        stored_response = (
            self._response_filter(response) if self._response_filter else response
        )

        # 7. persist user + assistant turns (LTM dual-write + STM working buffer)
        user_turn = self._make_turn(
            self._next_turn_id, "user", user_text, q.tolist(), topic_id, is_boundary
        )
        self._ltm.append_turn(self.conv_id, user_turn)
        self._next_turn_id += 1
        assistant_turn = self._make_turn(
            self._next_turn_id, "assistant", stored_response, None, topic_id, False
        )
        self._ltm.append_turn(self.conv_id, assistant_turn)
        self._next_turn_id += 1
        # Per spec: every turn lands in STM. add_turn_or_promote either
        # appends to an existing cached topic (atomicity preserved — same
        # full topic + 1 turn) or seeds a new STM entry with [turn] (new
        # topic's full history at this moment is just this turn — still
        # atomic). RoundProcessor reconciles all topics with importance at
        # round boundary.
        if self._stm is not None:
            self._stm.add_turn_or_promote(topic_id, user_turn)
            self._stm.add_turn_or_promote(topic_id, assistant_turn)

        # 8. round trigger (Phase 2-Full only)
        round_triggered = self._maybe_trigger_round()

        if return_debug:
            debug: dict[str, Any] = {
                "topic_id": topic_id,
                "is_boundary": is_boundary,
                "prefill_turns": prefill,
                "messages": messages,
            }
            if self._use_stm:
                debug["stm_hit"] = stm_hit
                debug["round_triggered"] = round_triggered
            return response, debug
        return response

    # ------------------------------------------------------------------
    # History preload (benchmarks)
    # ------------------------------------------------------------------

    def preload_history(self, turns: list[dict[str, Any]]) -> None:
        """Inject pre-existing user/assistant turns into LTM without LLM calls.

        With STM enabled, runs ``ceil(loaded_pairs / round_size)`` synchronous
        :class:`RoundProcessor` rounds at the end so the mention log,
        neighbor weights, and STM accurately reflect the preloaded history
        (bug 10 fix: previously fixed at 1 round regardless of size).
        """
        user_indices = [i for i, t in enumerate(turns) if t["role"] == "user"]
        if user_indices:
            user_texts = [turns[i]["content"] for i in user_indices]
            user_embs = np.asarray(self._encoder.encode(user_texts))
        else:
            user_embs = np.empty((0, self._encoder.dim))
        emb_by_turn_idx = {idx: user_embs[k] for k, idx in enumerate(user_indices)}

        last_topic_id = 0
        for i, t in enumerate(turns):
            role = t["role"]
            text = t["content"]
            ts = t.get("ts", datetime.now(timezone.utc).isoformat())
            if role == "user":
                q = emb_by_turn_idx[i]
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

        if self._round_processor is not None:
            # ceil-divide loaded_pairs by round_size; min 1 round.
            n_pairs = self._next_turn_id // 2
            n_rounds = max(1, (n_pairs + self._round_size - 1) // self._round_size)
            for _ in range(n_rounds):
                self._round_processor.process()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_trigger_round(self) -> bool:
        """Trigger ``RoundProcessor`` exactly once when ``self._next_turn_id``
        crosses the next ``2*round_size`` boundary.

        Returns True iff a round was dispatched (sync or async).
        """
        if self._round_processor is None:
            return False
        period = 2 * self._round_size
        if self._next_turn_id == 0 or self._next_turn_id % period != 0:
            return False
        if self._round_async:
            self._round_processor.process_async()
        else:
            self._round_processor.process()
        return True

    def wait_for_round(self, timeout: float | None = None) -> None:
        """Block until any in-flight async round finishes (Phase 2-Full only)."""
        if self._round_processor is not None:
            self._round_processor.wait(timeout)

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
