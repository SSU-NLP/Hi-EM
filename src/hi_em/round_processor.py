"""Round-level async post-processing of STM.

Triggered every ``round_size`` user turns (= ``2 * round_size`` jsonl rows
because each user turn is paired with one assistant turn). Reads the LTM,
recomputes per-topic importance, promotes high-importance topics to STM
(full topic turn list, atomic), and evicts to capacity.

Phase 2-Full design §1 (P2F-3). One ``RoundProcessor`` per conversation,
constructed alongside :class:`hi_em.orchestrator.HiEM`.

Concurrency model:

* ``process()`` is the sync entry; ``process_async()`` dispatches it on a
  daemon thread and returns the ``Thread`` handle.
* A per-instance ``RLock`` serializes overlapping rounds (subsequent
  ``process_async`` calls queue behind any in-flight round).
* :class:`hi_em.memory_window.MemoryWindow` has its own per-method ``RLock``
  for fine-grained safety, so concurrent reads from ``HiEM.handle_turn`` on
  the main thread are safe even mid-round.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from hi_em.ltm import LTM
from hi_em.memory_window import MemoryWindow
from hi_em.topic_importance import compute_importance

log = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Snapshot of one round's effect on the STM."""

    round_idx: int
    importance: dict[int, float]
    promoted: list[int]
    evicted: list[int]
    stm_topics: list[int]
    stm_total_turns: int


class RoundProcessor:
    """Per-conversation round processor."""

    def __init__(
        self,
        conv_id: str,
        ltm: LTM,
        stm: MemoryWindow,
        *,
        threshold: float = 0.5,
        alpha: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        lambda_r: float = 0.5,
        lambda_freq: float = 0.5,
        min_floor: float = 0.1,
    ) -> None:
        self.conv_id = conv_id
        self._ltm = ltm
        self._stm = stm
        self._threshold = threshold
        self._alpha = alpha
        self._lambda_r = lambda_r
        self._lambda_freq = lambda_freq
        self._min_floor = min_floor

        self._round_idx: int = 0
        self._mention_log: dict[int, list[int]] = {}
        self._neighbor_weights: dict[int, dict[int, float]] = {}
        self._prev_importance: dict[int, float] = {}
        self._last_processed_turn: int = -1

        self._lock = threading.RLock()
        self._inflight: threading.Thread | None = None

    @property
    def round_idx(self) -> int:
        with self._lock:
            return self._round_idx

    @property
    def mention_log(self) -> dict[int, list[int]]:
        with self._lock:
            return {tid: list(rounds) for tid, rounds in self._mention_log.items()}

    @property
    def neighbor_weights(self) -> dict[int, dict[int, float]]:
        with self._lock:
            return {src: dict(neigh) for src, neigh in self._neighbor_weights.items()}

    # ----------------------------------------------------------------
    # Sync entry
    # ----------------------------------------------------------------

    def process(self) -> RoundResult:
        """Run one round end-to-end.

        Steps: read new turns → update mention log + neighbor weights →
        recompute importance (normalized) → promote topics ≥ threshold
        (full LTM turns, atomic) → evict to capacity (lowest importance
        first). The promote+evict block is wrapped in ``stm.lock`` so the
        main thread never observes mid-round over-capacity STM (bug 9 fix).
        """
        with self._lock:
            round_idx = self._round_idx

            all_turns = self._ltm.load_turns(self.conv_id)
            new_turns = [
                t for t in all_turns if t["turn_id"] > self._last_processed_turn
            ]
            self._update_mention_log(new_turns, round_idx)
            self._update_neighbor_weights(all_turns)
            if all_turns:
                self._last_processed_turn = max(t["turn_id"] for t in all_turns)

            state = self._ltm.load_state(self.conv_id)
            if not state or not state.get("topics"):
                self._round_idx += 1
                return RoundResult(round_idx, {}, [], [], [], 0)

            # Normalized importance per spec ("topic_importance 계산 후 정규화").
            importance = compute_importance(
                state,
                round_now=round_idx,
                mention_log=self._mention_log,
                prev_importance=self._prev_importance,
                neighbor_weights=self._neighbor_weights,
                alpha=self._alpha,
                lambda_r=self._lambda_r,
                lambda_freq=self._lambda_freq,
                min_floor=self._min_floor,
                normalize=True,
            )

            promoted: list[int] = []
            with self._stm.lock:  # atomic round transition (bug 9 fix)
                for tid, score in importance.items():
                    if score >= self._threshold:
                        topic_turns = self._ltm.load_turns(
                            self.conv_id, topic_id=tid
                        )
                        if not topic_turns:
                            continue
                        self._stm.promote(tid, topic_turns)
                        promoted.append(tid)
                evicted = self._stm.evict_to_capacity(importance)

            self._prev_importance = importance
            self._round_idx += 1

            return RoundResult(
                round_idx=round_idx,
                importance=importance,
                promoted=promoted,
                evicted=evicted,
                stm_topics=sorted(self._stm.current_topics()),
                stm_total_turns=self._stm.total_turns(),
            )

    # ----------------------------------------------------------------
    # Async entry
    # ----------------------------------------------------------------

    def process_async(self) -> threading.Thread:
        """Dispatch :meth:`process` on a daemon thread."""
        thread = threading.Thread(
            target=self._run_safely,
            name=f"round-{self.conv_id}-{self._round_idx}",
            daemon=True,
        )
        self._inflight = thread
        thread.start()
        return thread

    def wait(self, timeout: float | None = None) -> None:
        """Block until the most-recently-dispatched async round finishes."""
        t = self._inflight
        if t is not None and t.is_alive():
            t.join(timeout)

    def _run_safely(self) -> None:
        try:
            self.process()
        except Exception:  # pragma: no cover — defensive
            log.exception("RoundProcessor[%s] round failed", self.conv_id)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _update_mention_log(
        self, new_turns: list[dict[str, Any]], round_idx: int
    ) -> None:
        """Log ``round_idx`` for each topic that appeared at least once this
        round. Per-round indicator (not per-turn count) per design §0.4.
        """
        topics_this_round = {t["topic_id"] for t in new_turns}
        for tid in topics_this_round:
            history = self._mention_log.setdefault(tid, [])
            if not history or history[-1] != round_idx:
                history.append(round_idx)

    def _update_neighbor_weights(
        self, all_turns: list[dict[str, Any]]
    ) -> None:
        """Recompute adjacency from consecutive USER turns.

        Counts undirected co-adjacency, then row-normalizes so each topic's
        outgoing weights sum to 1 (bounded coupling).
        """
        user_turns = sorted(
            (t for t in all_turns if t["role"] == "user"),
            key=lambda t: t["turn_id"],
        )
        weights: dict[int, dict[int, float]] = {}
        for prev, curr in zip(user_turns, user_turns[1:]):
            a, b = prev["topic_id"], curr["topic_id"]
            if a == b:
                continue
            row_a = weights.setdefault(a, {})
            row_b = weights.setdefault(b, {})
            row_a[b] = row_a.get(b, 0.0) + 1.0
            row_b[a] = row_b.get(a, 0.0) + 1.0
        for neigh in weights.values():
            total = sum(neigh.values())
            if total > 0:
                for k in neigh:
                    neigh[k] /= total
        self._neighbor_weights = weights
