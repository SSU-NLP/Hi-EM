"""Unit tests for ``hi_em.topic_importance``."""

from __future__ import annotations

import json
import math
from pathlib import Path

from hi_em.topic_importance import compute_importance, load_importance_config


def _state(*counts: int) -> dict:
    return {
        "conv_id": "x",
        "n_turns": sum(counts),
        "topics": [{"topic_id": i, "count": c, "centroid": [], "variance": []}
                   for i, c in enumerate(counts)],
    }


# --- config loader -------------------------------------------------------

def test_load_default_config_from_repo() -> None:
    cfg = load_importance_config()
    assert "alpha" in cfg and len(cfg["alpha"]) == 4
    for k in ("lambda_r", "lambda_freq", "min_floor"):
        assert k in cfg
    # JSON comment must be stripped.
    assert "_comment" not in cfg


def test_load_missing_path_returns_defaults(tmp_path: Path) -> None:
    cfg = load_importance_config(tmp_path / "does-not-exist.json")
    assert cfg == {
        "alpha": [1.0, 1.0, 1.0, 1.0],
        "lambda_r": 0.5,
        "lambda_freq": 0.5,
        "min_floor": 0.1,
    }


def test_load_custom_config(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "topic_importance": {
            "alpha": [2.0, 0.5, 1.0, 0.0],
            "lambda_r": 1.0,
            "lambda_freq": 1.0,
            "min_floor": 0.0,
        }
    }))
    cfg = load_importance_config(p)
    assert cfg["alpha"] == [2.0, 0.5, 1.0, 0.0]
    assert cfg["lambda_r"] == 1.0


def test_load_partial_section_uses_section_defaults_for_missing(tmp_path: Path) -> None:
    """Partial config — missing keys fall back to module defaults."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"topic_importance": {"lambda_r": 2.0}}))
    cfg = load_importance_config(p)
    assert cfg["lambda_r"] == 2.0
    assert cfg["alpha"] == [1.0, 1.0, 1.0, 1.0]   # default
    assert cfg["min_floor"] == 0.1                 # default


# --- compute_importance --------------------------------------------------

def test_empty_state_returns_empty_dict() -> None:
    assert compute_importance({"topics": []}, round_now=0, mention_log={}) == {}


def test_floor_applied_when_all_terms_zero() -> None:
    # No mentions → all 4 terms ~0; min_floor must apply.
    state = _state(0)  # n_t=0 → log(1)=0
    out = compute_importance(state, round_now=5, mention_log={}, min_floor=0.25)
    # The recency term default = exp(-0.5 * 0) = 1 since last_round defaults to round_now.
    # So this test verifies the floor specifically when α₃=0.
    out_no_recency = compute_importance(
        state, round_now=5, mention_log={}, alpha=(1.0, 1.0, 0.0, 1.0), min_floor=0.25
    )
    assert out_no_recency[0] == 0.25


def test_turn_count_strengthens(monkeypatch) -> None:
    """α₁ * log(1+n_t): more turns → higher score (strict mono)."""
    state = _state(1, 5, 20)
    out = compute_importance(
        state, round_now=0, mention_log={},
        alpha=(1.0, 0.0, 0.0, 0.0), min_floor=0.0,
    )
    # log(2) < log(6) < log(21)
    assert out[0] < out[1] < out[2]
    assert math.isclose(out[2], math.log(21), rel_tol=1e-9)


def test_recency_decay() -> None:
    """α₃ * exp(-λ_r·Δround): older last mention → lower score."""
    state = _state(1, 1)
    mention_log = {0: [9], 1: [3]}  # topic 0 recent, topic 1 stale
    out = compute_importance(
        state, round_now=10, mention_log=mention_log,
        alpha=(0.0, 0.0, 1.0, 0.0), lambda_r=0.5, min_floor=0.0,
    )
    assert out[0] > out[1]
    assert math.isclose(out[0], math.exp(-0.5 * 1), rel_tol=1e-9)
    assert math.isclose(out[1], math.exp(-0.5 * 7), rel_tol=1e-9)


def test_frequency_ema_higher_for_recent_repeats() -> None:
    """α₂ EMA(mentions): same total, but recent grouping > old."""
    state = _state(1, 1)
    out = compute_importance(
        state, round_now=10,
        mention_log={0: [9, 8, 7], 1: [0, 1, 2]},
        alpha=(0.0, 1.0, 0.0, 0.0), lambda_freq=2.0, min_floor=0.0,
    )
    assert out[0] > out[1]


def test_frequency_zero_when_no_mentions() -> None:
    state = _state(1)
    out = compute_importance(
        state, round_now=5, mention_log={},
        alpha=(0.0, 1.0, 0.0, 0.0), min_floor=0.0,
    )
    assert out[0] == 0.0


def test_neighbor_coupling_uses_prev_importance() -> None:
    """α₄ Σⱼ wᵢⱼ prev_I[j]: only fires when prev + weights both given."""
    state = _state(1, 1)
    prev = {0: 1.0, 1: 2.0}
    weights = {0: {1: 0.5}, 1: {0: 0.5}}
    out = compute_importance(
        state, round_now=0, mention_log={},
        prev_importance=prev, neighbor_weights=weights,
        alpha=(0.0, 0.0, 0.0, 1.0), min_floor=0.0,
    )
    # topic 0 borrows from topic 1: 0.5 * 2.0 = 1.0
    # topic 1 borrows from topic 0: 0.5 * 1.0 = 0.5
    assert math.isclose(out[0], 1.0)
    assert math.isclose(out[1], 0.5)


def test_neighbor_coupling_zero_without_prev_importance() -> None:
    state = _state(1)
    out = compute_importance(
        state, round_now=0, mention_log={},
        alpha=(0.0, 0.0, 0.0, 1.0), min_floor=0.0,
    )
    assert out[0] == 0.0


def test_combined_uniform_alpha() -> None:
    """Smoke: all 4 forces active together with uniform α=1."""
    state = _state(3, 1)
    out = compute_importance(
        state, round_now=2,
        mention_log={0: [0, 1, 2], 1: [2]},
        prev_importance={0: 0.5, 1: 0.1},
        neighbor_weights={0: {1: 1.0}, 1: {0: 1.0}},
        alpha=(1.0, 1.0, 1.0, 1.0), lambda_r=0.5, lambda_freq=2.0,
        min_floor=0.0,
    )
    # Sanity: both > 0, recently-repeated topic 0 dominates.
    assert out[0] > 0 and out[1] > 0
    assert out[0] > out[1]
