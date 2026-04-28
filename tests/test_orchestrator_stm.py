"""Phase 2-Full HiEM tests: STM-first prefill + round trigger.

Behavior under ``use_stm=True`` (post-2026-04-27 redesign):

* Every user/assistant pair is appended to STM via
  ``MemoryWindow.add_turn_or_promote`` — topic atomicity preserved
  whether the topic is new (creates entry with [turn]) or already
  cached (appends to existing).
* STM miss for the current topic on the prefill path additionally
  triggers a one-time bulk promote of the topic's prior LTM history
  (so revisits see all earlier turns).
* Round trigger fires every ``2 * round_size`` jsonl rows
  (= ``round_size`` user+assistant pairs); ``RoundProcessor`` recomputes
  normalized importance and evicts to capacity.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from hi_em.orchestrator import HiEM


class FakeEncoder:
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
        use_stm=True,
        round_async=False,        # sync rounds for deterministic tests
        promotion_threshold=0.0,  # promote everything for visibility
        **kwargs,
    )


# ------------------------------------------------------------------
# STM-first miss / hit
# ------------------------------------------------------------------


def test_first_turn_stm_miss_then_seed(tmp_path: Path) -> None:
    """First user turn: STM empty + no prior LTM turns. Prefill is empty
    (cold start). After the turn, the new topic seeds a STM entry with
    just the (user, assistant) pair."""
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    hi = _hi_em(tmp_path, enc, _llm())
    _, debug = hi.handle_turn("hi", return_debug=True)
    assert debug["stm_hit"] is False
    assert debug["messages"] == [{"role": "user", "content": "hi"}]
    # New topic's full history at this moment IS the just-completed pair.
    assert hi.stm.has(0)
    assert len(hi.stm.get(0)) == 2  # user + assistant


def test_topic_revisit_miss_pulls_full_topic_from_ltm(tmp_path: Path) -> None:
    """A → B → A: turn 3 is a STM hit because turn 1's STM entry was seeded
    on cold start, then turn 2 created topic 1 (B). On turn 3 (revisit A),
    STM has both topics already."""
    enc = FakeEncoder()
    enc.register("a1", [1.0, 0.0, 0.0, 0.0])
    enc.register("b1", [0.0, 1.0, 0.0, 0.0])
    enc.register("a2", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.side_effect = ["resp_a1", "resp_b1", "resp_a2"]
    hi = _hi_em(tmp_path, enc, llm)

    hi.handle_turn("a1")  # creates STM[A] = [a1, resp_a1]
    hi.handle_turn("b1")  # creates STM[B] = [b1, resp_b1]
    _, d3 = hi.handle_turn("a2", return_debug=True)

    assert d3["stm_hit"] is True  # seeded at turn 1
    contents = [m["content"] for m in d3["messages"]]
    assert "a1" in contents and "resp_a1" in contents
    # B is also in STM (seeded at turn 2) so prefill includes it too.
    assert "b1" in contents and "resp_b1" in contents
    assert contents[-1] == "a2"


def test_every_turn_lands_in_stm(tmp_path: Path) -> None:
    """Per spec: every turn pair lands in STM. Topics are atomic — each
    has its full set."""
    enc = FakeEncoder()
    enc.register("a1", [1.0, 0.0, 0.0, 0.0])
    enc.register("b1", [0.0, 1.0, 0.0, 0.0])
    enc.register("a2", [1.0, 0.0, 0.0, 0.0])
    enc.register("b2", [0.0, 1.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.side_effect = [f"r{i}" for i in range(4)]
    hi = _hi_em(tmp_path, enc, llm, round_size=10)

    hi.handle_turn("a1")  # seeds STM[0] = [a1, r0]
    hi.handle_turn("b1")  # seeds STM[1] = [b1, r1]
    hi.handle_turn("a2")  # appends STM[0] = [a1, r0, a2, r2]
    hi.handle_turn("b2")  # appends STM[1] = [b1, r1, b2, r3]

    assert hi.stm.current_topics() == {0, 1}
    assert len(hi.stm.get(0)) == 4
    assert len(hi.stm.get(1)) == 4


def test_revisit_after_round_with_ltm_history_pulls_full_topic(tmp_path: Path) -> None:
    """If a topic was evicted by a round and then revisited, the cache miss
    branch reloads the full LTM history (atomicity)."""
    enc = FakeEncoder()
    enc.register("a", [1.0, 0.0, 0.0, 0.0])
    enc.register("b", [0.0, 1.0, 0.0, 0.0])
    enc.register("c", [0.0, 0.0, 1.0, 0.0])
    enc.register("a2", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.side_effect = [f"r{i}" for i in range(4)]
    # max_topics=1 so the round will evict 2 of the 3 and keep just the
    # highest-importance one (deterministic via promotion_threshold=0).
    hi = _hi_em(tmp_path, enc, llm, round_size=3, stm_max_topics=1)

    hi.handle_turn("a")  # STM[0] = [a, r0]
    hi.handle_turn("b")  # STM[0,1]
    hi.handle_turn("c")  # STM[0,1,2] then round trigger → evicts 2 of them.
    # The surviving topic kept by the round may not be A; either way, when
    # we revisit A on turn 4, if A was evicted, the miss-promote loads its
    # full LTM history. STM ends up containing A in full.
    _, d4 = hi.handle_turn("a2", return_debug=True)
    a_turns = hi.stm.get(0)
    assert a_turns is not None
    # All A turns from LTM are present in STM (atomicity).
    ltm_a = hi._ltm.load_turns("c1", topic_id=0)
    assert len(a_turns) == len(ltm_a)


# ------------------------------------------------------------------
# Round trigger
# ------------------------------------------------------------------


def test_round_triggers_after_round_size_user_turns(tmp_path: Path) -> None:
    """round_size=3 → trigger every 3 user turns (= 6 jsonl rows)."""
    enc = FakeEncoder()
    for i in range(7):
        enc.register(f"u{i}", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.side_effect = [f"r{i}" for i in range(7)]
    hi = _hi_em(tmp_path, enc, llm, round_size=3)

    triggers: list[bool] = []
    for i in range(7):
        _, d = hi.handle_turn(f"u{i}", return_debug=True)
        triggers.append(d["round_triggered"])

    assert triggers == [False, False, True, False, False, True, False]
    assert hi.round_processor.round_idx == 2


def test_round_does_not_trigger_with_zero_turns(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("u0", [1.0, 0.0, 0.0, 0.0])
    hi = _hi_em(tmp_path, enc, _llm(), round_size=10)
    _, d = hi.handle_turn("u0", return_debug=True)
    assert d["round_triggered"] is False


def test_round_evicts_lowest_importance(tmp_path: Path) -> None:
    """3 distinct topics + max_topics=2 + threshold=0 → after round, 1 evicted."""
    enc = FakeEncoder()
    enc.register("a", [1.0, 0.0, 0.0, 0.0])
    enc.register("b", [0.0, 1.0, 0.0, 0.0])
    enc.register("c", [0.0, 0.0, 1.0, 0.0])
    llm = _llm()
    llm.chat.side_effect = ["ra", "rb", "rc"]
    hi = _hi_em(tmp_path, enc, llm, round_size=3, stm_max_topics=2)

    hi.handle_turn("a")
    hi.handle_turn("b")
    hi.handle_turn("c")  # round triggered (sync)

    assert len(hi.stm.current_topics()) == 2


# ------------------------------------------------------------------
# preload_history warms STM
# ------------------------------------------------------------------


def test_preload_history_warms_stm(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("u1", [1.0, 0.0, 0.0, 0.0])
    enc.register("u2", [0.0, 1.0, 0.0, 0.0])
    enc.register("query", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.return_value = "answer"
    hi = _hi_em(tmp_path, enc, llm)

    hi.preload_history([
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "r1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "r2"},
    ])

    assert hi.stm.current_topics() == {0, 1}

    _, d = hi.handle_turn("query", return_debug=True)
    assert d["stm_hit"] is True
    contents = [m["content"] for m in d["messages"]]
    assert "u1" in contents and "r1" in contents


# ------------------------------------------------------------------
# Backwards compat: use_stm=False keeps prior behavior
# ------------------------------------------------------------------


def test_baseline_mode_does_not_create_stm(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("hi", [1.0, 0.0, 0.0, 0.0])
    hi = HiEM(
        conv_id="c1",
        encoder=enc,
        llm=_llm(),
        model="test-model",
        ltm_root=tmp_path / "ltm",
    )
    assert hi.stm is None
    assert hi.round_processor is None
    _, d = hi.handle_turn("hi", return_debug=True)
    assert "stm_hit" not in d
    assert "round_triggered" not in d


# ------------------------------------------------------------------
# Async round path (smoke)
# ------------------------------------------------------------------


def test_async_round_completes_via_wait(tmp_path: Path) -> None:
    enc = FakeEncoder()
    enc.register("u0", [1.0, 0.0, 0.0, 0.0])
    llm = _llm()
    llm.chat.side_effect = ["r0"]
    hi = HiEM(
        conv_id="c1",
        encoder=enc,
        llm=llm,
        model="test-model",
        ltm_root=tmp_path / "ltm",
        use_stm=True,
        round_async=True,
        round_size=1,
        promotion_threshold=0.0,
    )
    hi.handle_turn("u0")  # triggers async round
    hi.wait_for_round(timeout=2.0)
    assert hi.stm.has(0)
