"""Memory window selection.

Given the current query embedding and the LTM state of a conversation, pick
the turns to promote into the LLM prefill prefix. Phase 2 baseline policy:

    1. Compute cosine(q, topic.centroid) for every topic in the state.
    2. Pick top-``k_topics`` topics by cosine.
    3. For each selected topic, take the last ``k_turns_per_topic`` turns
       (recency by ``turn_id``).
    4. Flatten and sort by ``turn_id`` so the LLM sees turns in chronological
       order.

Importance weighting (usage / cross-reference), adaptive ``k``, and merge
policy are deferred to Step 2-4+ (see ``context/01-hi-em-design.md §A``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
