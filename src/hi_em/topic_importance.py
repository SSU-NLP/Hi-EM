"""Topic importance — 4 작용 (강화·빈도·망각·연결).

phase-2-full-design.md §0.4 정의:

    I_t = α₁ log(1 + n_t)
        + α₂ EMA(mention_freq, λ_freq)
        + α₃ exp(-λ_r · Δround_last)
        + α₄ Σⱼ wᵢⱼ · prev_I[j]

가중치/상수는 ``configs/hiem.json``의 ``topic_importance`` 섹션에서 관리. CLI/
생성자 인자가 명시되면 그게 우선. 본 모듈의 ``compute_importance``는
**config-agnostic pure function** — caller가 dict를 풀어서 명시 인자로 전달.

Bug fixes (2026-04-27, codex review reaction):
    - Bug 5: ``_ema_frequency`` was an unbounded sum of decayed indicators;
      now multiplies by ``(1 - decay)`` to give a true EMA bounded in [0, 1].
    - Bug 6: a topic with no mentions previously defaulted ``last_round`` to
      ``round_now`` giving recency = exp(0) = 1 (false fresh). Fixed: set
      ``s3 = 0`` when ``mention_log[tid]`` is empty.
    - User spec ("topic_importance 계산 후 정규화") — added row-normalization:
      after computing raw scores, divide by max so the largest topic = 1.0 and
      others scale proportionally. Floor still applied. Threshold-based
      promotion is interpreted relative to the normalized scale (default 0.5
      = "at least half as important as the current peak").

Codex review carryover (phase-2-full-design.md §7-pre):
    - 재귀(α₄)는 1단계만 평가 (직전 round prev_importance를 그대로 사용,
      반복 수렴 안 시도).
    - quiet-but-important topic이 starve하지 않도록 ``min_floor`` lower bound.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from hi_em.config import get_section


def load_importance_config(path: Path | str | None = None) -> dict[str, Any]:
    """Return ``topic_importance`` section from ``configs/hiem.json``."""
    return get_section("topic_importance", path)


def _ema_frequency(
    rounds_with_mention: list[int], round_now: int, half_life: float
) -> float:
    """True EMA of per-round mention indicator with half-life ``half_life``.

    Each round contributes a 0/1 indicator; EMA = (1 - decay) Σ decay^Δ
    is bounded in [0, 1]. ``half_life`` is in rounds.
    """
    if not rounds_with_mention or half_life <= 0:
        return 0.0
    decay = math.exp(-math.log(2) / half_life)  # per-round retention
    return (1.0 - decay) * sum(
        decay ** (round_now - r) for r in rounds_with_mention
    )


def compute_importance(
    state: dict[str, Any],
    round_now: int,
    mention_log: dict[int, list[int]],
    prev_importance: dict[int, float] | None = None,
    neighbor_weights: dict[int, dict[int, float]] | None = None,
    alpha: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    lambda_r: float = 0.5,
    lambda_freq: float = 0.5,
    min_floor: float = 0.1,
    normalize: bool = True,
) -> dict[int, float]:
    """Compute per-topic importance scores.

    Args:
        state: ``ltm.load_state(conv_id)`` return value (has ``topics`` list).
        round_now: 현재 round index (0-based).
        mention_log: ``{topic_id: [round_idx, round_idx, ...]}``.
        prev_importance: 직전 round의 importance (재귀 1단계용).
        neighbor_weights: ``{topic_id: {other_topic_id: weight, ...}}``.
        alpha: (α₁, α₂, α₃, α₄) — 4 작용 가중치.
        lambda_r: recency decay rate (높을수록 빨리 잊음).
        lambda_freq: frequency EMA half-life (round 단위).
        min_floor: starvation floor (post-normalization).
        normalize: if True, scale raw scores by max so peak topic = 1.0
            (then re-apply min_floor). Set False for unit tests that need
            raw values.

    Returns:
        ``{topic_id: importance}``. All scores are ≥ ``min_floor``; with
        ``normalize=True``, scores are in ``[min_floor, 1.0]``.
    """
    if not state.get("topics"):
        return {}

    a1, a2, a3, a4 = alpha
    raw: dict[int, float] = {}

    for t in state["topics"]:
        tid = t["topic_id"]
        n_t = t.get("count", 0)
        rounds = mention_log.get(tid, [])

        # 1) 강화 (turn 수)
        s1 = a1 * math.log(1 + n_t)

        # 2) 강화 (빈도) — bounded EMA in [0, 1]
        s2 = a2 * _ema_frequency(rounds, round_now, lambda_freq)

        # 3) 망각 (recency) — only meaningful when topic was actually mentioned
        if rounds:
            last_round = max(rounds)
            s3 = a3 * math.exp(-lambda_r * (round_now - last_round))
        else:
            s3 = 0.0

        # 4) 연결 (인접 + 통합) — 1-step recursion via prev_importance
        s4 = 0.0
        if prev_importance and neighbor_weights and tid in neighbor_weights:
            for j, w in neighbor_weights[tid].items():
                s4 += w * prev_importance.get(j, 0.0)
        s4 *= a4

        raw[tid] = s1 + s2 + s3 + s4

    if normalize:
        peak = max(raw.values()) if raw else 0.0
        if peak > 0:
            scaled = {tid: v / peak for tid, v in raw.items()}
        else:
            scaled = {tid: 0.0 for tid in raw}
    else:
        scaled = raw

    return {tid: max(min_floor, v) for tid, v in scaled.items()}
