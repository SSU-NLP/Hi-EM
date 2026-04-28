"""Memory window — two implementations.

(a) :func:`select_memory_window` — Phase 2 baseline (stateless). Given the
    current query and LTM state, picks ``cosine top-k_topics × recency
    top-k_turns`` fresh on every call. Used by Phase 4 baseline measurements
    and as a fallback when STM is disabled.

(b) :class:`MemoryWindow` — Phase 2-Full STM (stateful, in-process RAM).
    Persistent across turns; ``RoundProcessor`` mutates it once per round
    based on importance.

Hard invariant (user-mandated, enforced by API shape, not assertion):
    **Topic atomicity** — a topic is either fully present in STM or fully
    absent. There is no turn-level slicing API. ``promote`` takes the full
    turn list for a topic; ``evict_lowest_importance`` removes a whole topic.

Capacity::

    max_topics : hard cap on number of distinct topics held in STM
    max_turns  : soft cap on total turn count. Eviction targets lowest-
                 importance topics one at a time. Topic atomicity wins
                 over the soft cap: if a single topic alone exceeds
                 ``max_turns``, it stays in STM (with a warning).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from hi_em.ltm import LTM

log = logging.getLogger(__name__)


def select_memory_window(
    q: np.ndarray,
    ltm: LTM,
    conv_id: str,
    k_topics: int,
    k_turns_per_topic: int,
) -> list[dict[str, Any]]:
    """Return turns to prefill, ordered by ``turn_id`` ascending."""
    state = ltm.load_state(conv_id)
    if state is None or not state.get("topics"):
        return []

    q_arr = np.asarray(q, dtype=np.float32)

    sims = [
        (float(np.dot(q_arr, np.asarray(t["centroid"], dtype=np.float32))), t["topic_id"])
        for t in state["topics"]
    ]
    sims.sort(key=lambda x: -x[0])
    selected_ids = [tid for _, tid in sims[:k_topics]]

    collected: list[dict[str, Any]] = []
    for tid in selected_ids:
        turns = sorted(ltm.load_turns(conv_id, topic_id=tid), key=lambda t: t["turn_id"])
        collected.extend(turns[-k_turns_per_topic:])

    collected.sort(key=lambda t: t["turn_id"])
    return collected


class MemoryWindow:
    """Topic-atomic short-term memory (STM).

    Each topic held in STM stores its **full** turn list (sorted by
    ``turn_id``). No partial-topic state is ever exposed.

    Thread safety: a ``threading.RLock`` guards every public mutator and
    accessor so :class:`hi_em.round_processor.RoundProcessor` can mutate the
    STM from a background thread while ``HiEM.handle_turn`` reads from the
    main thread.
    """

    def __init__(self, max_topics: int = 10, max_turns: int = 200) -> None:
        if max_topics < 1:
            raise ValueError(f"max_topics must be >= 1, got {max_topics}")
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.max_topics = max_topics
        self.max_turns = max_turns
        self._topics: dict[int, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def has(self, topic_id: int) -> bool:
        with self._lock:
            return topic_id in self._topics

    def get(self, topic_id: int) -> list[dict[str, Any]] | None:
        """Return a shallow copy of the topic's turns, or ``None`` on miss."""
        with self._lock:
            turns = self._topics.get(topic_id)
            return list(turns) if turns is not None else None

    def all_turns(self) -> list[dict[str, Any]]:
        """All STM turns across all topics, ascending by ``turn_id``."""
        with self._lock:
            flat: list[dict[str, Any]] = []
            for turns in self._topics.values():
                flat.extend(turns)
            flat.sort(key=lambda t: t["turn_id"])
            return flat

    def current_topics(self) -> set[int]:
        with self._lock:
            return set(self._topics.keys())

    def total_turns(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._topics.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "topics": len(self._topics),
                "turns": sum(len(v) for v in self._topics.values()),
                "max_topics": self.max_topics,
                "max_turns": self.max_turns,
                "topic_sizes": {tid: len(ts) for tid, ts in self._topics.items()},
            }

    # ------------------------------------------------------------------
    # Mutators (topic-atomic)
    # ------------------------------------------------------------------

    def promote(self, topic_id: int, turns: list[dict[str, Any]]) -> None:
        """Insert (or replace) a topic's full turn list.

        ``turns`` must contain **every** turn the LTM has for ``topic_id``.
        The list is copied and sorted by ``turn_id``; the caller may mutate
        their input afterward without affecting STM.

        Capacity is *not* enforced here — callers (RoundProcessor) decide
        when to evict. Promoting a topic that already exists overwrites
        the stored list (so newly-appended turns are picked up).
        """
        sorted_turns = sorted(turns, key=lambda t: t["turn_id"])
        with self._lock:
            if (
                topic_id not in self._topics
                and len(sorted_turns) > self.max_turns
            ):
                log.warning(
                    "MemoryWindow.promote: topic %d has %d turns, exceeds "
                    "max_turns=%d. Storing anyway (topic atomicity wins).",
                    topic_id, len(sorted_turns), self.max_turns,
                )
            self._topics[topic_id] = sorted_turns

    def evict_topic(self, topic_id: int) -> bool:
        """Remove a topic outright. Returns True if it was present."""
        with self._lock:
            return self._topics.pop(topic_id, None) is not None

    def maybe_append_turn(self, topic_id: int, turn: dict[str, Any]) -> bool:
        """Append a turn to a topic already in STM, keeping it sorted.

        Returns True iff the topic is currently in STM (and the turn was
        appended). Returns False if the topic is not in STM. Used by callers
        that need the explicit signal; for the common "every turn lands in
        STM" path (Phase 2-Full spec: "단기 메모리에 모든 대화 원문 저장"),
        prefer :meth:`add_turn_or_promote`.
        """
        with self._lock:
            turns = self._topics.get(topic_id)
            if turns is None:
                return False
            turns.append(turn)
            turns.sort(key=lambda t: t["turn_id"])
            return True

    def add_turn_or_promote(
        self, topic_id: int, turn: dict[str, Any]
    ) -> bool:
        """Append a turn to STM, creating a new topic entry if absent.

        Topic atomicity is preserved either way:

        * If ``topic_id`` already in STM → append the turn (the topic remains
          fully present, just one turn longer).
        * If ``topic_id`` not in STM → create a new entry with ``[turn]``;
          since this is the topic's first appearance, ``[turn]`` IS the
          topic's complete history at this moment.

        Returns True iff a new topic entry was created (caller may want to
        treat this as "topic_importance 초기화" trigger).
        """
        with self._lock:
            existing = self._topics.get(topic_id)
            if existing is None:
                self._topics[topic_id] = [turn]
                return True
            existing.append(turn)
            existing.sort(key=lambda t: t["turn_id"])
            return False

    @property
    def lock(self) -> threading.RLock:
        """Expose the internal lock so :class:`RoundProcessor` can hold it
        across promote+evict to make round end transitions atomic from the
        main thread's view (bug 9 fix)."""
        return self._lock

    def evict_lowest_importance(
        self, importance: dict[int, float]
    ) -> int | None:
        """Remove the STM topic with the smallest ``importance`` score.

        Returns the evicted ``topic_id`` (or ``None`` if STM empty).
        Topics in STM that have no entry in ``importance`` are treated
        as having score ``-inf`` so they evict first (defensive).
        """
        with self._lock:
            if not self._topics:
                return None
            target = min(
                self._topics.keys(),
                key=lambda tid: importance.get(tid, float("-inf")),
            )
            del self._topics[target]
            return target

    def evict_to_capacity(self, importance: dict[int, float]) -> list[int]:
        """Repeatedly evict lowest-importance topics until caps are met.

        ``max_topics`` is enforced strictly. ``max_turns`` is enforced as a
        soft cap: eviction stops when only one topic remains, even if that
        topic alone exceeds ``max_turns`` (topic atomicity > soft cap).

        Returns the list of evicted ``topic_id``s in eviction order.
        """
        evicted: list[int] = []
        with self._lock:
            while len(self._topics) > self.max_topics:
                tid = self.evict_lowest_importance(importance)
                if tid is None:
                    break
                evicted.append(tid)
            while (
                self.total_turns() > self.max_turns
                and len(self._topics) > 1
            ):
                tid = self.evict_lowest_importance(importance)
                if tid is None:
                    break
                evicted.append(tid)
        return evicted

    def clear(self) -> None:
        with self._lock:
            self._topics.clear()
