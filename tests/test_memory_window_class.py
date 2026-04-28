"""Unit tests for ``hi_em.memory_window.MemoryWindow`` (Phase 2-Full STM).

Topic-atomic invariant is enforced by API shape; these tests verify it
holds end-to-end (no turn-level slicing leaks in or out).
"""

from __future__ import annotations

import threading

import pytest

from hi_em.memory_window import MemoryWindow


def _turn(turn_id: int, topic_id: int, role: str = "user") -> dict:
    return {
        "turn_id": turn_id,
        "ts": "2026-04-27T00:00:00Z",
        "role": role,
        "text": f"t{turn_id}",
        "embedding": None,
        "topic_id": topic_id,
        "is_boundary": False,
    }


# ------------------------------------------------------------------
# Construction / validation
# ------------------------------------------------------------------


def test_invalid_max_topics_rejected() -> None:
    with pytest.raises(ValueError):
        MemoryWindow(max_topics=0, max_turns=10)


def test_invalid_max_turns_rejected() -> None:
    with pytest.raises(ValueError):
        MemoryWindow(max_topics=1, max_turns=0)


# ------------------------------------------------------------------
# Empty / miss
# ------------------------------------------------------------------


def test_empty_stm_get_returns_none() -> None:
    stm = MemoryWindow()
    assert stm.get(0) is None
    assert stm.has(0) is False
    assert stm.all_turns() == []
    assert stm.current_topics() == set()
    assert stm.total_turns() == 0


# ------------------------------------------------------------------
# Promote / get round-trip
# ------------------------------------------------------------------


def test_promote_then_get_returns_full_list() -> None:
    stm = MemoryWindow()
    turns = [_turn(0, 7), _turn(1, 7), _turn(2, 7)]
    stm.promote(7, turns)
    out = stm.get(7)
    assert out is not None
    assert [t["turn_id"] for t in out] == [0, 1, 2]


def test_promote_sorts_by_turn_id() -> None:
    stm = MemoryWindow()
    stm.promote(3, [_turn(5, 3), _turn(1, 3), _turn(2, 3)])
    out = stm.get(3)
    assert [t["turn_id"] for t in out] == [1, 2, 5]


def test_promote_returns_copy_not_alias() -> None:
    """Mutating the input list after promote must not affect STM."""
    stm = MemoryWindow()
    turns = [_turn(0, 0), _turn(1, 0)]
    stm.promote(0, turns)
    turns.append(_turn(99, 0))  # caller mutates after
    stored = stm.get(0)
    assert [t["turn_id"] for t in stored] == [0, 1]


def test_get_returns_independent_copy() -> None:
    """Mutating get() result must not affect STM."""
    stm = MemoryWindow()
    stm.promote(0, [_turn(0, 0)])
    out = stm.get(0)
    out.append(_turn(99, 0))
    assert len(stm.get(0)) == 1


# ------------------------------------------------------------------
# Promote replaces (idempotent + refresh semantics)
# ------------------------------------------------------------------


def test_promote_existing_topic_replaces_full_list() -> None:
    """Re-promoting refreshes STM turns (e.g. after new turns were appended)."""
    stm = MemoryWindow()
    stm.promote(0, [_turn(0, 0), _turn(1, 0)])
    stm.promote(0, [_turn(0, 0), _turn(1, 0), _turn(2, 0)])
    assert [t["turn_id"] for t in stm.get(0)] == [0, 1, 2]


# ------------------------------------------------------------------
# all_turns ordering
# ------------------------------------------------------------------


def test_all_turns_chronological_across_topics() -> None:
    stm = MemoryWindow()
    stm.promote(0, [_turn(0, 0), _turn(2, 0), _turn(4, 0)])
    stm.promote(1, [_turn(1, 1), _turn(3, 1)])
    flat = stm.all_turns()
    assert [t["turn_id"] for t in flat] == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------
# Eviction (topic-atomic)
# ------------------------------------------------------------------


def test_evict_topic_removes_whole_topic() -> None:
    stm = MemoryWindow()
    stm.promote(0, [_turn(0, 0), _turn(1, 0)])
    stm.promote(1, [_turn(2, 1)])
    assert stm.evict_topic(0) is True
    assert stm.has(0) is False
    assert stm.has(1) is True
    assert stm.evict_topic(0) is False  # already gone


def test_evict_lowest_importance_picks_smallest_score() -> None:
    stm = MemoryWindow()
    stm.promote(0, [_turn(0, 0)])
    stm.promote(1, [_turn(1, 1)])
    stm.promote(2, [_turn(2, 2)])
    importance = {0: 0.9, 1: 0.1, 2: 0.5}
    out = stm.evict_lowest_importance(importance)
    assert out == 1
    assert stm.current_topics() == {0, 2}


def test_evict_lowest_importance_missing_score_evicted_first() -> None:
    """A topic in STM without any score entry is treated as -inf."""
    stm = MemoryWindow()
    stm.promote(0, [_turn(0, 0)])
    stm.promote(1, [_turn(1, 1)])
    out = stm.evict_lowest_importance({0: 0.5})  # topic 1 has no score
    assert out == 1


def test_evict_lowest_importance_empty_returns_none() -> None:
    assert MemoryWindow().evict_lowest_importance({}) is None


# ------------------------------------------------------------------
# Capacity enforcement (evict_to_capacity)
# ------------------------------------------------------------------


def test_evict_to_capacity_max_topics() -> None:
    stm = MemoryWindow(max_topics=2, max_turns=100)
    stm.promote(0, [_turn(0, 0)])
    stm.promote(1, [_turn(1, 1)])
    stm.promote(2, [_turn(2, 2)])
    importance = {0: 0.1, 1: 0.5, 2: 0.9}
    evicted = stm.evict_to_capacity(importance)
    assert evicted == [0]  # lowest importance
    assert stm.current_topics() == {1, 2}


def test_evict_to_capacity_max_turns_soft() -> None:
    stm = MemoryWindow(max_topics=10, max_turns=3)
    stm.promote(0, [_turn(0, 0), _turn(1, 0)])      # 2 turns
    stm.promote(1, [_turn(2, 1), _turn(3, 1)])      # 2 turns → 4 total
    importance = {0: 0.1, 1: 0.9}
    evicted = stm.evict_to_capacity(importance)
    assert evicted == [0]
    assert stm.total_turns() == 2
    assert stm.current_topics() == {1}


def test_evict_to_capacity_topic_atomicity_overrides_max_turns() -> None:
    """A single topic larger than max_turns is kept (atomicity wins)."""
    stm = MemoryWindow(max_topics=10, max_turns=2)
    big = [_turn(i, 0) for i in range(5)]
    stm.promote(0, big)
    evicted = stm.evict_to_capacity({0: 0.0})
    assert evicted == []
    assert stm.total_turns() == 5
    assert stm.current_topics() == {0}


def test_evict_to_capacity_keeps_one_when_oversized_singleton() -> None:
    """Two topics, both individually within cap, joint > cap: evict the
    lower-importance one. Then remaining single topic is kept regardless."""
    stm = MemoryWindow(max_topics=10, max_turns=2)
    stm.promote(0, [_turn(0, 0), _turn(1, 0), _turn(2, 0)])  # 3 > 2 alone
    evicted = stm.evict_to_capacity({0: 0.0})
    # Only one topic; atomicity rule keeps it.
    assert evicted == []
    assert stm.current_topics() == {0}


def test_evict_to_capacity_oversized_promote_logs_warning(caplog) -> None:
    """Promoting a single topic > max_turns logs a warning but stores it."""
    import logging
    caplog.set_level(logging.WARNING, logger="hi_em.memory_window")
    stm = MemoryWindow(max_topics=10, max_turns=2)
    stm.promote(0, [_turn(0, 0), _turn(1, 0), _turn(2, 0)])
    assert any("exceeds max_turns" in r.message for r in caplog.records)
    assert stm.has(0)


# ------------------------------------------------------------------
# Clear
# ------------------------------------------------------------------


def test_clear_removes_everything() -> None:
    stm = MemoryWindow()
    stm.promote(0, [_turn(0, 0)])
    stm.promote(1, [_turn(1, 1)])
    stm.clear()
    assert stm.current_topics() == set()
    assert stm.total_turns() == 0


# ------------------------------------------------------------------
# add_turn_or_promote (every turn lands in STM)
# ------------------------------------------------------------------


def test_add_turn_or_promote_creates_new_topic() -> None:
    """When the topic isn't cached, the call seeds a new entry. Topic
    atomicity holds because [turn] IS the topic's full history at this
    moment."""
    stm = MemoryWindow()
    created = stm.add_turn_or_promote(7, _turn(0, 7))
    assert created is True
    assert stm.has(7)
    assert stm.get(7) == [_turn(0, 7)]


def test_add_turn_or_promote_appends_to_existing() -> None:
    stm = MemoryWindow()
    stm.promote(7, [_turn(0, 7), _turn(1, 7)])
    created = stm.add_turn_or_promote(7, _turn(2, 7))
    assert created is False
    assert [t["turn_id"] for t in stm.get(7)] == [0, 1, 2]


def test_add_turn_or_promote_keeps_sorted() -> None:
    stm = MemoryWindow()
    stm.promote(0, [_turn(5, 0)])
    stm.add_turn_or_promote(0, _turn(1, 0))   # earlier turn id
    assert [t["turn_id"] for t in stm.get(0)] == [1, 5]


def test_lock_property_returns_internal_lock() -> None:
    """RoundProcessor uses ``stm.lock`` as a context manager to atomic-bound
    promote + evict."""
    stm = MemoryWindow()
    with stm.lock:
        stm.promote(0, [_turn(0, 0)])
        stm.evict_topic(0)
    assert stm.current_topics() == set()


# ------------------------------------------------------------------
# Thread safety (smoke test)
# ------------------------------------------------------------------


def test_concurrent_promote_and_read_safe() -> None:
    """Two threads promoting/reading concurrently must not raise.

    Final STM state should contain whichever topics finished promoting,
    each in its full form (no partial promotions exposed).
    """
    stm = MemoryWindow(max_topics=100, max_turns=10000)
    n_per_thread = 50
    err: list[BaseException] = []

    def writer(start: int) -> None:
        try:
            for i in range(start, start + n_per_thread):
                stm.promote(i, [_turn(i * 10 + j, i) for j in range(3)])
        except BaseException as e:
            err.append(e)

    def reader() -> None:
        try:
            for _ in range(n_per_thread * 2):
                _ = stm.all_turns()
                _ = stm.stats()
        except BaseException as e:
            err.append(e)

    t1 = threading.Thread(target=writer, args=(0,))
    t2 = threading.Thread(target=writer, args=(n_per_thread,))
    t3 = threading.Thread(target=reader)
    for t in (t1, t2, t3):
        t.start()
    for t in (t1, t2, t3):
        t.join()

    assert err == []
    # All topics that were promoted should be intact (3 turns each).
    for tid in stm.current_topics():
        assert len(stm.get(tid)) == 3
