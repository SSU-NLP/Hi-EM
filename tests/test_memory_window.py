"""Unit tests for ``hi_em.memory_window.select_memory_window``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hi_em.ltm import LTM
from hi_em.memory_window import select_memory_window


def _unit(*coords: float) -> list[float]:
    v = np.asarray(coords, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _turn(turn_id: int, topic_id: int, embedding: list[float]) -> dict:
    return {
        "turn_id": turn_id,
        "ts": "2026-04-25T12:00:00Z",
        "role": "user",
        "text": f"t{turn_id}",
        "embedding": embedding,
        "topic_id": topic_id,
        "is_boundary": False,
    }


@pytest.fixture
def ltm(tmp_path: Path) -> LTM:
    return LTM(tmp_path / "ltm")


def test_empty_state_returns_empty(ltm: LTM) -> None:
    q = np.asarray(_unit(1.0, 0.0))
    assert select_memory_window(q, ltm, "missing", 3, 5) == []


def test_state_with_no_topics(ltm: LTM) -> None:
    ltm.update_state("c1", {"n_turns": 0, "topics": []})
    q = np.asarray(_unit(1.0, 0.0))
    assert select_memory_window(q, ltm, "c1", 3, 5) == []


def test_single_topic_returns_its_turns(ltm: LTM) -> None:
    e = _unit(1.0, 0.0)
    for tid in range(3):
        ltm.append_turn("c1", _turn(tid, topic_id=0, embedding=e))
    ltm.update_state(
        "c1",
        {"n_turns": 3, "topics": [{"topic_id": 0, "centroid": e}]},
    )
    out = select_memory_window(np.asarray(e), ltm, "c1", k_topics=1, k_turns_per_topic=10)
    assert [t["turn_id"] for t in out] == [0, 1, 2]


def test_top_k_topics_selected_by_cosine(ltm: LTM) -> None:
    # topic 0 along (1,0); topic 1 along (0,1); topic 2 along (-1,0)
    e0, e1, e2 = _unit(1.0, 0.0), _unit(0.0, 1.0), _unit(-1.0, 0.0)
    ltm.append_turn("c1", _turn(0, 0, e0))
    ltm.append_turn("c1", _turn(1, 1, e1))
    ltm.append_turn("c1", _turn(2, 2, e2))
    ltm.update_state("c1", {
        "n_turns": 3,
        "topics": [
            {"topic_id": 0, "centroid": e0},
            {"topic_id": 1, "centroid": e1},
            {"topic_id": 2, "centroid": e2},
        ],
    })
    # query along (1, 0.1): closest = topic 0, then topic 1, then topic 2
    q = np.asarray(_unit(1.0, 0.1))
    out = select_memory_window(q, ltm, "c1", k_topics=2, k_turns_per_topic=10)
    topic_ids = {t["topic_id"] for t in out}
    assert topic_ids == {0, 1}


def test_recency_truncation(ltm: LTM) -> None:
    e = _unit(1.0, 0.0)
    for tid in range(5):
        ltm.append_turn("c1", _turn(tid, 0, e))
    ltm.update_state("c1", {"n_turns": 5, "topics": [{"topic_id": 0, "centroid": e}]})
    out = select_memory_window(np.asarray(e), ltm, "c1", k_topics=1, k_turns_per_topic=2)
    # last 2 turns by turn_id
    assert [t["turn_id"] for t in out] == [3, 4]


def test_k_topics_exceeds_available(ltm: LTM) -> None:
    e0, e1 = _unit(1.0, 0.0), _unit(0.0, 1.0)
    ltm.append_turn("c1", _turn(0, 0, e0))
    ltm.append_turn("c1", _turn(1, 1, e1))
    ltm.update_state("c1", {
        "n_turns": 2,
        "topics": [
            {"topic_id": 0, "centroid": e0},
            {"topic_id": 1, "centroid": e1},
        ],
    })
    q = np.asarray(_unit(1.0, 0.0))
    out = select_memory_window(q, ltm, "c1", k_topics=10, k_turns_per_topic=10)
    assert len(out) == 2


def test_output_sorted_by_turn_id_across_topics(ltm: LTM) -> None:
    # Interleave topic 0 and topic 1 turns so flattened order isn't trivially
    # produced by per-topic concat.
    e0, e1 = _unit(1.0, 0.0), _unit(0.0, 1.0)
    schedule = [(0, 0, e0), (1, 1, e1), (2, 0, e0), (3, 1, e1), (4, 0, e0)]
    for tid, topic_id, e in schedule:
        ltm.append_turn("c1", _turn(tid, topic_id, e))
    ltm.update_state("c1", {
        "n_turns": 5,
        "topics": [
            {"topic_id": 0, "centroid": e0},
            {"topic_id": 1, "centroid": e1},
        ],
    })
    q = np.asarray(_unit(1.0, 1.0))  # equally close to both
    out = select_memory_window(q, ltm, "c1", k_topics=2, k_turns_per_topic=10)
    ids = [t["turn_id"] for t in out]
    assert ids == sorted(ids), f"expected ascending turn_id, got {ids}"
    assert ids == [0, 1, 2, 3, 4]


def test_only_selected_topic_turns_returned(ltm: LTM) -> None:
    """Turns belonging to non-selected topics must not leak through."""
    e0, e1 = _unit(1.0, 0.0), _unit(0.0, 1.0)
    ltm.append_turn("c1", _turn(0, 0, e0))
    ltm.append_turn("c1", _turn(1, 1, e1))
    ltm.append_turn("c1", _turn(2, 0, e0))
    ltm.update_state("c1", {
        "n_turns": 3,
        "topics": [
            {"topic_id": 0, "centroid": e0},
            {"topic_id": 1, "centroid": e1},
        ],
    })
    q = np.asarray(_unit(1.0, 0.0))  # closest to topic 0 only
    out = select_memory_window(q, ltm, "c1", k_topics=1, k_turns_per_topic=10)
    assert all(t["topic_id"] == 0 for t in out)
    assert [t["turn_id"] for t in out] == [0, 2]
