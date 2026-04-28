"""Tests for the read-only :meth:`HiEM.eval_query` evaluation path.

The contract: ``eval_query`` must return a sensible response while leaving
segmenter state, STM contents, and the LTM jsonl unchanged. This is what
the LoCoMo conv-level cache relies on — building per-sample Hi-EM state
once and answering ~200 test questions against the frozen state.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hi_em.embedding import QueryEncoder
from hi_em.orchestrator import HiEM


class _DummyEncoder:
    """Deterministic stand-in for QueryEncoder. Avoids loading bge."""

    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        out = []
        for t in texts:
            seed = (sum(ord(c) for c in t) * 7919) % (2**31)
            r = np.random.default_rng(seed)
            v = r.normal(size=self.dim).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-9)
            out.append(v)
        return np.asarray(out)


class _RecordingLLM:
    """Captures last call args without actually hitting any endpoint."""

    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.calls: list[dict] = []

    def chat(self, messages, model, **kwargs):
        self.calls.append({"messages": list(messages), "model": model, "kwargs": kwargs})
        return self.response


def _make_history(n_pairs: int = 6) -> list[dict]:
    history = []
    for i in range(n_pairs):
        history.append({"role": "user", "content": f"user turn {i}"})
        history.append({"role": "assistant", "content": f"assistant turn {i}"})
    return history


@pytest.fixture
def stm_hiem(tmp_path: Path) -> HiEM:
    enc = _DummyEncoder()
    llm = _RecordingLLM()
    hi = HiEM(
        conv_id="conv-test",
        encoder=enc,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        model="dummy",
        ltm_root=tmp_path / "ltm",
        alpha=1.0, lmda=10.0, sigma0_sq=0.01,
        use_stm=True,
        round_size=3,
        stm_max_topics=5,
        stm_max_turns=200,
        promotion_threshold=0.0,  # promote everything for the test
        importance_alpha=(1.0, 1.0, 1.0, 1.0),
        lambda_r=0.5, lambda_freq=0.5, min_floor=0.1,
        round_async=False,
    )
    hi.preload_history(_make_history(6))
    return hi


def _snapshot_state(hi: HiEM) -> dict:
    """Capture mutation-sensitive fields of HiEM."""
    seg = hi._segmenter  # type: ignore[attr-defined]
    return {
        "next_turn_id": hi._next_turn_id,  # type: ignore[attr-defined]
        "prev_k": seg.prev_k,
        "counts": seg.counts.copy(),
        "n_topics": len(seg.topics),
        "topic_centroids": [t.mu.copy() for t in seg.topics],
        "topic_counts": [t.n for t in seg.topics],
        "stm_topics": (
            {tid: list(turns) for tid, turns in hi._stm._topics.items()}
            if hi._stm is not None else None
        ),
        "ltm_jsonl_size": (Path(hi._ltm.root) / f"{hi.conv_id}.jsonl").stat().st_size,
    }


def test_eval_query_returns_response(stm_hiem: HiEM) -> None:
    response = stm_hiem.eval_query("did you mean turn 0?")
    assert response == "ok"
    assert stm_hiem._llm.calls, "LLM was not called"  # type: ignore[attr-defined]


def test_eval_query_does_not_mutate_segmenter(stm_hiem: HiEM) -> None:
    before = _snapshot_state(stm_hiem)
    stm_hiem.eval_query("a query that should not move centroids")
    after = _snapshot_state(stm_hiem)
    assert before["next_turn_id"] == after["next_turn_id"]
    assert before["prev_k"] == after["prev_k"]
    assert np.array_equal(before["counts"], after["counts"])
    assert before["n_topics"] == after["n_topics"]
    for b, a in zip(before["topic_centroids"], after["topic_centroids"]):
        assert np.array_equal(b, a)
    assert before["topic_counts"] == after["topic_counts"]


def test_eval_query_does_not_mutate_stm(stm_hiem: HiEM) -> None:
    before = _snapshot_state(stm_hiem)
    stm_hiem.eval_query("does this leak into STM?")
    after = _snapshot_state(stm_hiem)
    assert before["stm_topics"] is not None and after["stm_topics"] is not None
    assert set(before["stm_topics"]) == set(after["stm_topics"])
    for tid, turns in before["stm_topics"].items():
        assert len(turns) == len(after["stm_topics"][tid])


def test_eval_query_does_not_grow_ltm(stm_hiem: HiEM) -> None:
    before = _snapshot_state(stm_hiem)
    for _ in range(3):
        stm_hiem.eval_query("question")
    after = _snapshot_state(stm_hiem)
    assert before["ltm_jsonl_size"] == after["ltm_jsonl_size"], (
        "eval_query must not append to the LTM jsonl"
    )


def test_eval_query_idempotent_under_repeated_calls(stm_hiem: HiEM) -> None:
    """Two calls with the same text → same prefill turns and same messages
    sent to the LLM."""
    _, debug1 = stm_hiem.eval_query("repeat me", return_debug=True)
    _, debug2 = stm_hiem.eval_query("repeat me", return_debug=True)
    ids1 = [t["turn_id"] for t in debug1["prefill_turns"]]
    ids2 = [t["turn_id"] for t in debug2["prefill_turns"]]
    assert ids1 == ids2
    assert debug1["topic_id"] == debug2["topic_id"]
    # The LLM-bound message stream should also be identical.
    assert debug1["messages"] == debug2["messages"]


def test_eval_query_falls_back_to_stateless_window(tmp_path: Path) -> None:
    """Without STM (stateless Phase-2 baseline), eval_query uses
    ``select_memory_window`` instead of STM-merged prefill."""
    enc = _DummyEncoder()
    llm = _RecordingLLM()
    hi = HiEM(
        conv_id="conv-stateless",
        encoder=enc,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        model="dummy",
        ltm_root=tmp_path / "ltm",
        alpha=1.0, lmda=10.0, sigma0_sq=0.01,
        k_topics=2, k_turns_per_topic=2,
        use_stm=False,
    )
    hi.preload_history(_make_history(4))
    response, debug = hi.eval_query("any question", return_debug=True)
    assert response == "ok"
    assert "stm_hit" not in debug
    assert isinstance(debug["prefill_turns"], list)
