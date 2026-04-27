"""Memory window — two implementations.

(a) ``select_memory_window`` (Phase 2 baseline, stateless): given current
    query and LTM state, picks ``cosine top-k_topics × recency top-k_turns``
    fresh on every call. Used by Phase 4 baseline measurements.

(b) ``MemoryWindow`` class (Phase 2-Full STM, stateful, RAM cache): persistent
    across turns, importance-based promote/evict, **topic-level atomic** (no
    turn-level slicing — a topic is either fully present or fully absent).
    Capacity:
        max_topics : hard cap on number of distinct topics in cache
        max_turns  : soft cap on total turn count. Eviction targets
                     lowest-importance topics one at a time. A single topic
                     larger than ``max_turns`` is rejected on promote (rare).

Use one or the other per experiment via ``HiEM(..., use_stm=True/False)``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from hi_em.atomic_io import load_json, save_json
from hi_em.ltm import LTM


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
