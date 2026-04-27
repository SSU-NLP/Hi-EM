"""Unit tests for ``hi_em.ltm.LTM`` (file-backed long-term memory)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hi_em.ltm import LTM


@pytest.fixture
def ltm(tmp_path: Path) -> LTM:
    return LTM(tmp_path / "ltm")


def _turn(turn_id: int, topic_id: int = 0, text: str = "hi") -> dict:
    return {
        "turn_id": turn_id,
        "ts": "2026-04-25T12:00:00Z",
        "role": "user",
        "text": text,
        "embedding": [0.1, 0.2, 0.3],
        "topic_id": topic_id,
        "is_boundary": turn_id == 0,
    }


def test_load_missing_returns_empty_or_none(ltm: LTM) -> None:
    assert ltm.load_turns("nope") == []
    assert ltm.load_state("nope") is None
    assert ltm.list_conversations() == []


def test_append_then_load_preserves_order(ltm: LTM) -> None:
    for i in range(3):
        ltm.append_turn("c1", _turn(i, text=f"t{i}"))
    turns = ltm.load_turns("c1")
    assert [t["turn_id"] for t in turns] == [0, 1, 2]
    assert [t["text"] for t in turns] == ["t0", "t1", "t2"]


def test_load_turns_filters_by_topic(ltm: LTM) -> None:
    ltm.append_turn("c1", _turn(0, topic_id=0))
    ltm.append_turn("c1", _turn(1, topic_id=1))
    ltm.append_turn("c1", _turn(2, topic_id=0))
    only_topic_0 = ltm.load_turns("c1", topic_id=0)
    assert [t["turn_id"] for t in only_topic_0] == [0, 2]


def test_update_state_overwrites(ltm: LTM) -> None:
    ltm.update_state("c1", {"n_turns": 1, "topics": []})
    ltm.update_state("c1", {"n_turns": 5, "topics": [{"topic_id": 0, "count": 5}]})
    state = ltm.load_state("c1")
    assert state == {"n_turns": 5, "topics": [{"topic_id": 0, "count": 5}]}


def test_state_and_turns_independent(ltm: LTM) -> None:
    ltm.append_turn("c1", _turn(0))
    assert ltm.load_state("c1") is None  # state not written yet
    ltm.update_state("c1", {"n_turns": 1, "topics": []})
    assert len(ltm.load_turns("c1")) == 1


def test_conversations_isolated(ltm: LTM) -> None:
    ltm.append_turn("c1", _turn(0))
    ltm.append_turn("c2", _turn(0))
    assert len(ltm.load_turns("c1")) == 1
    assert len(ltm.load_turns("c2")) == 1


def test_list_conversations(ltm: LTM) -> None:
    ltm.append_turn("conv-a", _turn(0))
    ltm.update_state("conv-b", {"n_turns": 0, "topics": []})
    ltm.append_turn("conv-c", _turn(0))
    ltm.update_state("conv-c", {"n_turns": 1, "topics": []})
    assert ltm.list_conversations() == ["conv-a", "conv-b", "conv-c"]


def test_root_dir_created_lazily(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "ltm"
    assert not nested.exists()
    LTM(nested)
    assert nested.is_dir()
