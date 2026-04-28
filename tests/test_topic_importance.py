"""Unit tests for ``hi_em.topic_importance``."""

from __future__ import annotations

import json
import math
from pathlib import Path

from hi_em.topic_importance import (
    _ema_frequency,
    compute_importance,
    load_importance_config,
)


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
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"topic_importance": {"lambda_r": 2.0}}))
    cfg = load_importance_config(p)
    assert cfg["lambda_r"] == 2.0
    assert cfg["alpha"] == [1.0, 1.0, 1.0, 1.0]
    assert cfg["min_floor"] == 0.1


# --- _ema_frequency (bounded EMA) ----------------------------------------

def test_ema_frequency_bounded_above_by_one() -> None:
    """True EMA over per-round 0/1 indicator stays ≤ 1.0 even when topic is
    mentioned every round (regression: old impl was unbounded sum)."""
    rounds = list(range(0, 100))   # mentioned every round 0..99
    val = _ema_frequency(rounds, round_now=99, half_life=2.0)
    assert val <= 1.0 + 1e-9
    assert val > 0.5


def test_ema_frequency_zero_for_empty() -> None:
    assert _ema_frequency([], round_now=10, half_life=1.0) == 0.0


# --- compute_importance --------------------------------------------------

def test_empty_state_returns_empty_dict() -> None:
    assert compute_importance({"topics": []}, round_now=0, mention_log={}) == {}


def test_no_mentions_no_recency_boost() -> None:
    """Bug 6 regression: a topic with empty mention_log must NOT receive
    a free recency=1 boost. With α₃ acting alone and no mentions, score
    must be 0 (then floored)."""
    state = _state(0)
    out = compute_importance(
        state, round_now=5, mention_log={},
        alpha=(0.0, 0.0, 1.0, 0.0), min_floor=0.0,
        normalize=False,
    )
    assert out[0] == 0.0


def test_floor_applied_when_all_terms_zero() -> None:
    state = _state(0)
    out = compute_importance(
        state, round_now=5, mention_log={}, min_floor=0.25,
        normalize=False,
    )
    assert out[0] == 0.25


def test_turn_count_strengthens() -> None:
    """α₁ * log(1+n_t): more turns → higher raw score."""
    state = _state(1, 5, 20)
    out = compute_importance(
        state, round_now=0, mention_log={},
        alpha=(1.0, 0.0, 0.0, 0.0), min_floor=0.0,
        normalize=False,
    )
    assert out[0] < out[1] < out[2]
    assert math.isclose(out[2], math.log(21), rel_tol=1e-9)


def test_recency_decay_with_mentions() -> None:
    """α₃ * exp(-λ_r·Δround): older last mention → lower score."""
    state = _state(1, 1)
    mention_log = {0: [9], 1: [3]}
    out = compute_importance(
        state, round_now=10, mention_log=mention_log,
        alpha=(0.0, 0.0, 1.0, 0.0), lambda_r=0.5, min_floor=0.0,
        normalize=False,
    )
    assert out[0] > out[1]
    assert math.isclose(out[0], math.exp(-0.5 * 1), rel_tol=1e-9)
    assert math.isclose(out[1], math.exp(-0.5 * 7), rel_tol=1e-9)


def test_frequency_ema_higher_for_recent_repeats() -> None:
    state = _state(1, 1)
    out = compute_importance(
        state, round_now=10,
        mention_log={0: [9, 8, 7], 1: [0, 1, 2]},
        alpha=(0.0, 1.0, 0.0, 0.0), lambda_freq=2.0, min_floor=0.0,
        normalize=False,
    )
    assert out[0] > out[1]


def test_frequency_zero_when_no_mentions() -> None:
    state = _state(1)
    out = compute_importance(
        state, round_now=5, mention_log={},
        alpha=(0.0, 1.0, 0.0, 0.0), min_floor=0.0,
        normalize=False,
    )
    assert out[0] == 0.0


def test_neighbor_coupling_uses_prev_importance() -> None:
    state = _state(1, 1)
    prev = {0: 1.0, 1: 2.0}
    weights = {0: {1: 0.5}, 1: {0: 0.5}}
    out = compute_importance(
        state, round_now=0, mention_log={},
        prev_importance=prev, neighbor_weights=weights,
        alpha=(0.0, 0.0, 0.0, 1.0), min_floor=0.0,
        normalize=False,
    )
    assert math.isclose(out[0], 1.0)
    assert math.isclose(out[1], 0.5)


def test_neighbor_coupling_zero_without_prev_importance() -> None:
    state = _state(1)
    out = compute_importance(
        state, round_now=0, mention_log={},
        alpha=(0.0, 0.0, 0.0, 1.0), min_floor=0.0,
        normalize=False,
    )
    assert out[0] == 0.0


def test_combined_uniform_alpha() -> None:
    """All 4 forces active with uniform α=1; recently-repeated topic dominates."""
    state = _state(3, 1)
    out = compute_importance(
        state, round_now=2,
        mention_log={0: [0, 1, 2], 1: [2]},
        prev_importance={0: 0.5, 1: 0.1},
        neighbor_weights={0: {1: 1.0}, 1: {0: 1.0}},
        alpha=(1.0, 1.0, 1.0, 1.0), lambda_r=0.5, lambda_freq=2.0,
        min_floor=0.0,
        normalize=False,
    )
    assert out[0] > out[1]


# --- Normalization -------------------------------------------------------

def test_normalization_scales_peak_to_one() -> None:
    """When normalize=True, the peak topic should become 1.0."""
    state = _state(1, 5, 20)
    out = compute_importance(
        state, round_now=0, mention_log={},
        alpha=(1.0, 0.0, 0.0, 0.0), min_floor=0.0,
        normalize=True,
    )
    assert math.isclose(max(out.values()), 1.0)
    # Rank preserved: more turns → higher normalized score.
    assert out[0] < out[1] < out[2]


def test_normalization_min_floor_still_applies() -> None:
    """A topic whose normalized score is below min_floor gets clipped up."""
    state = _state(100, 1)  # 1st topic dominates, 2nd very low
    out = compute_importance(
        state, round_now=0, mention_log={},
        alpha=(1.0, 0.0, 0.0, 0.0), min_floor=0.3,
        normalize=True,
    )
    assert out[1] >= 0.3
    assert math.isclose(out[0], 1.0)


def test_normalization_all_zero_raw_returns_floor() -> None:
    """With normalize=True and all zero raw (no terms active), output = floor."""
    state = _state(0, 0)
    out = compute_importance(
        state, round_now=0, mention_log={},
        alpha=(1.0, 1.0, 1.0, 1.0), min_floor=0.2,
        normalize=True,
    )
    assert out == {0: 0.2, 1: 0.2}
