"""Unit tests for :class:`hi_em.round_processor.RoundProcessor`."""

from __future__ import annotations

from pathlib import Path

import pytest

from hi_em.ltm import LTM
from hi_em.memory_window import MemoryWindow
from hi_em.round_processor import RoundProcessor


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


def _state(topic_specs: list[tuple[int, int]]) -> dict:
    """topic_specs = [(topic_id, count), ...]."""
    return {
        "n_turns": sum(c for _, c in topic_specs) * 2,
        "topics": [
            {
                "topic_id": tid,
                "centroid": [1.0, 0.0],
                "variance": [0.01, 0.01],
                "count": count,
            }
            for tid, count in topic_specs
        ],
    }


@pytest.fixture
def ltm(tmp_path: Path) -> LTM:
    return LTM(tmp_path / "ltm")


# ------------------------------------------------------------------
# Empty / no-op
# ------------------------------------------------------------------


def test_empty_ltm_returns_empty_result(ltm: LTM) -> None:
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm)
    res = rp.process()
    assert res.round_idx == 0
    assert res.importance == {}
    assert res.promoted == []
    assert res.evicted == []
    assert res.stm_topics == []
    assert rp.round_idx == 1  # advanced even on no-op


def test_state_without_topics_returns_empty(ltm: LTM) -> None:
    ltm.update_state("c1", {"n_turns": 0, "topics": []})
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm)
    res = rp.process()
    assert res.importance == {}
    assert res.promoted == []


# ------------------------------------------------------------------
# Promotion (atomicity invariant)
# ------------------------------------------------------------------


def test_promotes_topic_above_threshold_full_turns(ltm: LTM) -> None:
    """Promoted topic must contain *all* LTM turns for that topic."""
    for i in range(4):
        ltm.append_turn("c1", _turn(i, topic_id=0, role="user"))
        ltm.append_turn("c1", _turn(i + 100, topic_id=0, role="assistant"))
    ltm.update_state("c1", _state([(0, 4)]))
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=0.0)
    res = rp.process()
    assert 0 in res.promoted
    stm_turns = stm.get(0)
    assert stm_turns is not None
    # Every turn in LTM for topic 0 must be in STM (no slicing).
    ltm_topic_turns = ltm.load_turns("c1", topic_id=0)
    assert len(stm_turns) == len(ltm_topic_turns)


def test_skips_topic_below_threshold(ltm: LTM) -> None:
    ltm.append_turn("c1", _turn(0, 0))
    ltm.update_state("c1", _state([(0, 1)]))
    stm = MemoryWindow()
    # threshold > log(1+1) + 1 + 1 + 0 = 0.69 + 2 = 2.69
    rp = RoundProcessor("c1", ltm, stm, threshold=100.0)
    res = rp.process()
    assert res.promoted == []
    assert stm.current_topics() == set()


# ------------------------------------------------------------------
# Mention log accumulation across rounds
# ------------------------------------------------------------------


def test_mention_log_records_each_round_once_per_topic(ltm: LTM) -> None:
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=100.0)

    # Round 0: topic 0 appears twice in same round → log only once
    ltm.append_turn("c1", _turn(0, 0))
    ltm.append_turn("c1", _turn(1, 0))
    ltm.update_state("c1", _state([(0, 2)]))
    rp.process()

    # Round 1: topic 0 again
    ltm.append_turn("c1", _turn(2, 0))
    ltm.update_state("c1", _state([(0, 3)]))
    rp.process()

    log = rp.mention_log
    assert log[0] == [0, 1]  # one entry per round


def test_only_new_turns_drive_mention_log(ltm: LTM) -> None:
    """Topics that didn't appear this round must not be re-logged."""
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=100.0)

    ltm.append_turn("c1", _turn(0, 0))
    ltm.update_state("c1", _state([(0, 1)]))
    rp.process()  # round 0: topic 0

    # Round 1: only topic 1 appears
    ltm.append_turn("c1", _turn(1, 1))
    ltm.update_state("c1", _state([(0, 1), (1, 1)]))
    rp.process()

    log = rp.mention_log
    assert log[0] == [0]      # topic 0 not re-logged in round 1
    assert log[1] == [1]


# ------------------------------------------------------------------
# Neighbor weights
# ------------------------------------------------------------------


def test_neighbor_weights_from_consecutive_user_topics(ltm: LTM) -> None:
    # User-turn topic sequence: 0 → 1 → 0 → 2  (three transitions)
    schedule = [(0, 0), (1, 1), (2, 0), (3, 2)]
    for tid, topic_id in schedule:
        ltm.append_turn("c1", _turn(tid, topic_id, role="user"))
    ltm.update_state("c1", _state([(0, 2), (1, 1), (2, 1)]))
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=100.0)
    rp.process()

    nw = rp.neighbor_weights
    # 0↔1 once, 0↔1 again? schedule: 0→1, 1→0, 0→2 → 0:{1,1,2}, 1:{0,0}, 2:{0}
    # row-normalized 0: {1: 2/3, 2: 1/3}; 1: {0: 1.0}; 2: {0: 1.0}
    assert nw[0][1] == pytest.approx(2 / 3)
    assert nw[0][2] == pytest.approx(1 / 3)
    assert nw[1][0] == pytest.approx(1.0)
    assert nw[2][0] == pytest.approx(1.0)


def test_self_loop_topic_has_no_neighbor_entry(ltm: LTM) -> None:
    # All user turns same topic → no transitions, no neighbor weights.
    for i in range(3):
        ltm.append_turn("c1", _turn(i, 0, role="user"))
    ltm.update_state("c1", _state([(0, 3)]))
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=100.0)
    rp.process()
    assert rp.neighbor_weights == {}


# ------------------------------------------------------------------
# Eviction (capacity)
# ------------------------------------------------------------------


def test_evict_when_max_topics_exceeded(ltm: LTM) -> None:
    # Three topics, each one turn — promote all, then evict to max_topics=2.
    for tid in range(3):
        ltm.append_turn("c1", _turn(tid, tid, role="user"))
    ltm.update_state("c1", _state([(0, 1), (1, 1), (2, 1)]))
    stm = MemoryWindow(max_topics=2, max_turns=100)
    rp = RoundProcessor("c1", ltm, stm, threshold=0.0)
    res = rp.process()
    assert len(res.promoted) == 3
    assert len(res.evicted) == 1
    assert len(stm.current_topics()) == 2


# ------------------------------------------------------------------
# Round counter and idempotence
# ------------------------------------------------------------------


def test_round_idx_advances_each_call(ltm: LTM) -> None:
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm)
    assert rp.round_idx == 0
    rp.process()
    assert rp.round_idx == 1
    rp.process()
    assert rp.round_idx == 2


def test_re_promote_refreshes_with_new_turns(ltm: LTM) -> None:
    """Round 1 promotes topic 0 with 2 turns; round 2 has 4 turns — STM
    must reflect the larger list (atomicity: full topic only)."""
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=0.0)

    for i in range(2):
        ltm.append_turn("c1", _turn(i, 0))
    ltm.update_state("c1", _state([(0, 2)]))
    rp.process()
    assert len(stm.get(0)) == 2

    for i in range(2, 4):
        ltm.append_turn("c1", _turn(i, 0))
    ltm.update_state("c1", _state([(0, 4)]))
    rp.process()
    assert len(stm.get(0)) == 4


# ------------------------------------------------------------------
# Async dispatch
# ------------------------------------------------------------------


def test_process_async_runs_round(ltm: LTM) -> None:
    ltm.append_turn("c1", _turn(0, 0))
    ltm.update_state("c1", _state([(0, 1)]))
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=0.0)
    t = rp.process_async()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert rp.round_idx == 1
    assert stm.has(0)


def test_async_then_sync_serialize(ltm: LTM) -> None:
    """Sync process called while async is running must wait via the lock."""
    for i in range(3):
        ltm.append_turn("c1", _turn(i, 0))
    ltm.update_state("c1", _state([(0, 3)]))
    stm = MemoryWindow()
    rp = RoundProcessor("c1", ltm, stm, threshold=0.0)
    rp.process_async()
    rp.wait(timeout=2.0)
    rp.process()
    assert rp.round_idx == 2
