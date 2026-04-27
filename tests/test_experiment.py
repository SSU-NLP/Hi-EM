"""Unit tests for ``hi_em.experiment`` lifecycle + atomic_io."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hi_em.atomic_io import append_jsonl, load_json, load_jsonl, save_json
from hi_em.experiment import (
    EXPERIMENT_JSON_SCHEMA_VERSION,
    SUMMARY_JSON_SCHEMA_VERSION,
    ExperimentMeta,
    Session,
    create_experiment,
    experiment_dir,
    find_resumable_experiment,
    make_experiment_id,
    mark_experiment_complete,
    mark_round_complete,
    round_dir,
    sanity_check_summary,
    save_session,
)


# --- atomic_io ----------------------------------------------------------

def test_save_json_atomic_no_partial(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_json(p, {"a": 1})
    assert load_json(p) == {"a": 1}
    # No leftover .tmp on success
    assert not (tmp_path / "x.json.tmp").exists()


def test_save_json_handles_lone_surrogate(tmp_path: Path) -> None:
    """LLM outputs sometimes contain lone surrogates that crash strict utf-8."""
    p = tmp_path / "y.json"
    save_json(p, {"text": "hello \udc00 world"})
    # Must not raise; '?' replacement happens at encode time.
    loaded = load_json(p)
    assert "hello" in loaded["text"]


def test_jsonl_append_and_load(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    assert load_jsonl(p) == []  # missing file → []
    append_jsonl(p, {"i": 1})
    append_jsonl(p, {"i": 2})
    rows = load_jsonl(p)
    assert [r["i"] for r in rows] == [1, 2]


# --- experiment id / dir layout ----------------------------------------

def test_make_experiment_id_no_model_by_default() -> None:
    eid = make_experiment_id("persistence", "oracle", "method=hi-em",
                              timestamp="20260427T120000")
    assert eid == "20260427T120000_persistence_oracle_method=hi-em"
    # Slash → __ for path safety.
    eid2 = make_experiment_id("openai/gpt-4", timestamp="20260427T120000")
    assert "/" not in eid2 and "openai__gpt-4" in eid2


def test_create_experiment_writes_immutable_meta(tmp_path: Path) -> None:
    meta = ExperimentMeta(
        experiment_id="20260427T120000_test",
        session_id=None,
        config={"method": "hi-em", "alpha": 1.0},
        seeds={"data_seed": 42, "sampling_seed": None},
        created_at="2026-04-27T12:00:00Z",
    )
    exp_dir = create_experiment(meta, root=tmp_path)
    assert exp_dir == tmp_path / "experiments" / "20260427T120000_test"
    saved = load_json(exp_dir / "experiment.json")
    assert saved["schema_version"] == EXPERIMENT_JSON_SCHEMA_VERSION
    assert saved["config"]["method"] == "hi-em"
    assert saved["seeds"]["data_seed"] == 42


def test_create_experiment_idempotent_resume(tmp_path: Path) -> None:
    meta = ExperimentMeta(
        experiment_id="x", session_id=None, config={"a": 1},
        seeds={}, created_at="...",
    )
    create_experiment(meta, root=tmp_path)
    # Second call with same id returns same dir, doesn't overwrite.
    create_experiment(meta, root=tmp_path)
    assert load_json(tmp_path / "experiments" / "x" / "experiment.json")["config"]["a"] == 1


def test_create_experiment_rejects_id_collision(tmp_path: Path) -> None:
    m1 = ExperimentMeta(
        experiment_id="x", session_id=None, config={"a": 1},
        seeds={}, created_at="...",
    )
    create_experiment(m1, root=tmp_path)
    m2 = ExperimentMeta(
        experiment_id="y", session_id=None, config={}, seeds={}, created_at="...",
    )
    # Path collision via filesystem reuse: same dir, different exp_id → error.
    save_json(tmp_path / "experiments" / "x" / "experiment.json",
              {**m1.to_dict(), "experiment_id": "y"})  # tamper (y wins, last in dict)
    with pytest.raises(ValueError, match="different experiment_id"):
        create_experiment(m1, root=tmp_path)


# --- round lifecycle ---------------------------------------------------

def test_mark_round_complete_writes_summary_then_checkpoint(tmp_path: Path) -> None:
    meta = ExperimentMeta(
        experiment_id="x", session_id=None, config={}, seeds={}, created_at="...",
    )
    create_experiment(meta, root=tmp_path)

    summary = {"n_processed": 30, "primary_metric": 0.7, "error_rate": 0.0}
    mark_round_complete("x", 1, summary, root=tmp_path)

    rd = round_dir("x", 1, root=tmp_path)
    s = load_json(rd / "summary.json")
    assert s["schema_version"] == SUMMARY_JSON_SCHEMA_VERSION
    assert s["primary_metric"] == 0.7

    ckpt = load_json(rd / "checkpoint.json")
    assert ckpt["complete"] is True
    assert ckpt["round"] == 1


def test_find_resumable_returns_highest_complete_round(tmp_path: Path) -> None:
    meta = ExperimentMeta(
        experiment_id="x", session_id=None, config={}, seeds={}, created_at="...",
    )
    create_experiment(meta, root=tmp_path)
    assert find_resumable_experiment("x", root=tmp_path) is None  # nothing yet

    mark_round_complete("x", 1, {"n_processed": 1, "primary_metric": 1.0}, root=tmp_path)
    mark_round_complete("x", 2, {"n_processed": 1, "primary_metric": 1.0}, root=tmp_path)
    assert find_resumable_experiment("x", root=tmp_path) == 2

    mark_round_complete("x", 3, {"n_processed": 1, "primary_metric": 1.0}, root=tmp_path)
    assert find_resumable_experiment("x", root=tmp_path) == 3


def test_find_resumable_ignores_round_with_only_summary(tmp_path: Path) -> None:
    """SKILL §9.7: summary.json without checkpoint.json → not complete."""
    meta = ExperimentMeta(experiment_id="x", session_id=None, config={},
                          seeds={}, created_at="...")
    create_experiment(meta, root=tmp_path)
    rd = round_dir("x", 1, root=tmp_path)
    rd.mkdir(parents=True)
    save_json(rd / "summary.json", {"primary_metric": 0.5})
    # No checkpoint.json — must not resume from here.
    assert find_resumable_experiment("x", root=tmp_path) is None


def test_completed_experiment_returns_none(tmp_path: Path) -> None:
    meta = ExperimentMeta(experiment_id="x", session_id=None, config={},
                          seeds={}, created_at="...")
    create_experiment(meta, root=tmp_path)
    mark_round_complete("x", 1, {"n_processed": 1, "primary_metric": 1.0}, root=tmp_path)
    mark_experiment_complete("x", total_rounds=1, root=tmp_path)
    # SKILL §5: completed experiments are not resumed.
    assert find_resumable_experiment("x", root=tmp_path) is None


# --- sanity_check_summary (SKILL §9.8) ---------------------------------

def test_sanity_check_clean() -> None:
    s = {"n_processed": 30, "primary_metric": 0.7, "error_rate": 0.01}
    assert sanity_check_summary(s) == []


def test_sanity_check_zero_items() -> None:
    warns = sanity_check_summary({"n_processed": 0})
    assert any("zero items" in w for w in warns)


def test_sanity_check_zero_metric() -> None:
    s = {"n_processed": 30, "primary_metric": 0.0}
    warns = sanity_check_summary(s)
    assert any("0% primary metric" in w for w in warns)


def test_sanity_check_big_metric_jump() -> None:
    prev = {"primary_metric": 0.8}
    s = {"n_processed": 30, "primary_metric": 0.4}  # drop > 0.3
    warns = sanity_check_summary(s, prev=prev)
    assert any("jumped" in w for w in warns)


def test_sanity_check_high_error_rate() -> None:
    s = {"n_processed": 30, "primary_metric": 0.5, "error_rate": 0.2}
    warns = sanity_check_summary(s)
    assert any("error_rate" in w for w in warns)


# --- session ----------------------------------------------------------

def test_session_layout(tmp_path: Path) -> None:
    sess = Session(
        session_id="20260427_phase4_main",
        purpose="HP sweep",
        common_config={"model": "Qwen/Qwen3-8B"},
    )
    sess.add_experiment("20260427T120000_persistence", overrides={"alpha": 1.0})
    sess.add_experiment("20260427T120500_freq-shift", overrides={"alpha": 10.0})
    save_session(sess, root=tmp_path)

    saved = load_json(tmp_path / "sessions" / "20260427_phase4_main" / "session.json")
    assert saved["common_config"]["model"] == "Qwen/Qwen3-8B"
    assert len(saved["experiments"]) == 2
    assert saved["experiments"][0]["overrides"]["alpha"] == 1.0
