"""Unit tests for ``hi_em.orchestrator.HiEM``.

Encoder is faked (lookup-based) and LLM is mocked so tests run without
network access or model downloads. End-to-end smoke tests against a real
OpenRouter / vLLM endpoint live in Phase 3-3.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from hi_em.orchestrator import HiEM


class FakeEncoder:
    """Deterministic encoder: text → preset L2-normalized vector."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.device = "cpu"
        self._lookup: dict[str, np.ndarray] = {}

    def register(self, text: str, vec: list[float]) -> None:
        v = np.asarray(vec, dtype=np.float32)
        self._lookup[text] = (v / np.linalg.norm(v)).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._lookup[t] for t in texts])


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = "ok"
    return llm


def _hi_em(tmp_path: Path, encoder: FakeEncoder, llm: MagicMock, **kwargs) -> HiEM:
    return HiEM(
        conv_id="c1",
        encoder=encoder,
        llm=llm,
        model="test-model",
        ltm_root=tmp_path / "ltm",
        **kwargs,
    )


def test_single_turn_writes_user_and_assistant(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.return_value = "hello"
    hi = _hi_em(tmp_path, enc, llm)

    out = hi.handle_turn("hi")
    assert out == "hello"

    turns = hi._ltm.load_turns("c1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert [t["turn_id"] for t in turns] == [0, 1]
    assert turns[0]["text"] == "hi"
    assert turns[1]["text"] == "hello"
    assert turns[0]["embedding"] is not None
    assert turns[1]["embedding"] is None  # assistant turn carries no embedding


def test_topic_change_marks_boundary(tmp_path: Path) -> None:
    """Segmenter convention: first turn is_boundary=False (no prior topic).
    Boundary fires only when topic_id changes from prev turn.
    """
    enc = FakeEncoder()
    enc.register("a", [1.0, 0.0, 0.0, 0.0])
    enc.register("b", [0.0, 1.0, 0.0, 0.0])  # orthogonal → likely new topic
    hi = _hi_em(tmp_path, enc, _llm())
    hi.handle_turn("a")
    hi.handle_turn("b")
    turns = hi._ltm.load_turns("c1")
    user_turns = [t for t in turns if t["role"] == "user"]
    assert user_turns[0]["is_boundary"] is False  # first turn convention
    assert user_turns[1]["is_boundary"] is True   # topic switched


def test_state_snapshot_after_turn(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    hi = _hi_em(tmp_path, enc, _llm())
    hi.handle_turn("hi")
    state = hi._ltm.load_state("c1")
    assert state["conv_id"] == "c1"
    assert state["n_turns"] == 0  # snapshot taken before user turn appended
    assert len(state["topics"]) == 1
    assert state["topics"][0]["count"] == 1


def test_messages_have_only_current_user_on_first_turn(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    hi = _hi_em(tmp_path, enc, llm)
    hi.handle_turn("hi")
    msgs = llm.chat.call_args.args[0]
    assert msgs == [{"role": "user", "content": "hi"}]


def test_system_prompt_prepended(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    hi = _hi_em(tmp_path, enc, llm, system_prompt="You are helpful.")
    hi.handle_turn("hi")
    msgs = llm.chat.call_args.args[0]
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[-1] == {"role": "user", "content": "hi"}


def test_second_turn_includes_first_in_prefill(tmp_path: Path) -> None:
    enc = FakeEncoder()
    # Same topic embedding for both turns → first turn ends up in window.
    enc.register("u1", [1.0, 0.0, 0.0, 0.0])
    enc.register("u2", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.side_effect = ["a1", "a2"]
    hi = _hi_em(tmp_path, enc, llm, k_topics=1, k_turns_per_topic=10)

    hi.handle_turn("u1")
    hi.handle_turn("u2")

    msgs_second = llm.chat.call_args_list[1].args[0]
    contents = [m["content"] for m in msgs_second]
    # First turn (user u1 + assistant a1) should be in prefill, then u2.
    assert "u1" in contents
    assert "a1" in contents
    assert contents[-1] == "u2"
    assert msgs_second[-1] == {"role": "user", "content": "u2"}


def test_topic_revisit_brings_old_turn_back(tmp_path: Path) -> None:
    """A → B → A pattern: third turn's prefill must contain turn 0."""
    enc = FakeEncoder()
    enc.register("a1", [1.0, 0.0, 0.0, 0.0])
    enc.register("b1", [0.0, 1.0, 0.0, 0.0])
    enc.register("a2", [1.0, 0.0, 0.0, 0.0])  # same direction as a1
    llm = _llm()
    llm.chat.side_effect = ["resp_a1", "resp_b1", "resp_a2"]
    # k_topics=1: only the closest topic is included → must pick A on turn 3.
    hi = _hi_em(tmp_path, enc, llm, k_topics=1, k_turns_per_topic=10)

    hi.handle_turn("a1")
    hi.handle_turn("b1")
    hi.handle_turn("a2")

    msgs_third = llm.chat.call_args_list[2].args[0]
    contents = [m["content"] for m in msgs_third]
    assert "a1" in contents
    assert "resp_a1" in contents
    assert "b1" not in contents
    assert "resp_b1" not in contents
    assert contents[-1] == "a2"


def test_llm_kwargs_forwarded(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    hi = _hi_em(tmp_path, enc, llm, temperature=0.3, max_tokens=64)
    hi.handle_turn("hi")
    kwargs = llm.chat.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 64


def test_response_filter_strips_before_storage_but_returns_raw(tmp_path: Path) -> None:
    """response_filter applies to LTM-stored text only; caller still gets raw response."""
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.return_value = "<think>internal reasoning</think>real answer"

    def strip_think(s: str) -> str:
        import re
        return re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL).strip()

    hi = _hi_em(tmp_path, enc, llm, response_filter=strip_think)
    raw = hi.handle_turn("hi")

    # caller sees raw response (so they can display thinking if they want)
    assert raw == "<think>internal reasoning</think>real answer"

    # LTM stores filtered version (so next-turn prefill stays compact)
    assistant_turn = [t for t in hi._ltm.load_turns("c1") if t["role"] == "assistant"][0]
    assert assistant_turn["text"] == "real answer"


def test_preload_history_writes_user_and_assistant_turns(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("u1", [1.0, 0.0, 0.0, 0.0])
    enc.register("u2", [0.0, 1.0, 0.0, 0.0])  # different topic
    llm = _llm()
    hi = _hi_em(tmp_path, enc, llm)

    hi.preload_history([
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ])

    # No LLM calls during preload
    assert llm.chat.call_count == 0

    turns = hi._ltm.load_turns("c1")
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert [t["text"] for t in turns] == ["u1", "a1", "u2", "a2"]

    # User turns get embeddings; assistants don't.
    assert turns[0]["embedding"] is not None
    assert turns[1]["embedding"] is None
    assert turns[2]["embedding"] is not None
    assert turns[3]["embedding"] is None

    # Assistant inherits the previous user's topic_id.
    assert turns[1]["topic_id"] == turns[0]["topic_id"]
    assert turns[3]["topic_id"] == turns[2]["topic_id"]

    # Two distinct topics formed (u1, u2 are orthogonal).
    state = hi._ltm.load_state("c1")
    assert len(state["topics"]) == 2


def test_preload_history_then_handle_turn_uses_history(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("seed", [1.0, 0.0, 0.0, 0.0])
    enc.register("query", [1.0, 0.0, 0.0, 0.0])  # same topic as seed
    llm = _llm()
    llm.chat.return_value = "answer"
    hi = _hi_em(tmp_path, enc, llm, k_topics=1, k_turns_per_topic=10)

    hi.preload_history([
        {"role": "user", "content": "seed"},
        {"role": "assistant", "content": "seed_reply"},
    ])
    hi.handle_turn("query")

    msgs = llm.chat.call_args.args[0]
    contents = [m["content"] for m in msgs]
    assert "seed" in contents
    assert "seed_reply" in contents
    assert contents[-1] == "query"


def test_handle_turn_return_debug(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("a1", [1.0, 0.0, 0.0, 0.0])
    enc.register("a2", [1.0, 0.0, 0.0, 0.0])  # same topic as a1
    llm = _llm()
    llm.chat.side_effect = ["r1", "r2"]
    hi = _hi_em(tmp_path, enc, llm, k_topics=1, k_turns_per_topic=10)

    hi.handle_turn("a1")
    response, debug = hi.handle_turn("a2", return_debug=True)
    assert response == "r2"
    assert debug["topic_id"] == 0  # reused (same direction as a1)
    # prefill should contain the prior 'a1' / 'r1' pair (selected by cosine)
    prefill_texts = [t["text"] for t in debug["prefill_turns"]]
    assert "a1" in prefill_texts
    # messages = prefill + current user
    assert debug["messages"][-1] == {"role": "user", "content": "a2"}


def test_ltm_files_created_at_root(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    hi = _hi_em(tmp_path, enc, _llm())
    hi.handle_turn("hi")
    root = tmp_path / "ltm"
    assert (root / "c1.jsonl").exists()
    assert (root / "c1.state.json").exists()
    state = json.loads((root / "c1.state.json").read_text())
    assert "topics" in state
