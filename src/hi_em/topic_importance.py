"""Topic importance — 4 작용 (강화·빈도·망각·연결).

phase-2-full-design.md §0.4 정의:

    I_t = α₁ log(1 + n_t)
        + α₂ EMA(mention_freq, λ_freq)
        + α₃ exp(-λ_r · Δround_last)
        + α₄ Σⱼ wᵢⱼ · prev_I[j]

가중치/상수는 ``configs/hiem.json``의 ``topic_importance`` 섹션에서 관리. CLI/
생성자 인자가 명시되면 그게 우선. 본 모듈의 ``compute_importance``는
**config-agnostic pure function** — caller가 dict를 풀어서 명시 인자로 전달.

Codex review 반영 (phase-2-full-design.md §7-pre):
    - 재귀(α₄)는 1단계만 평가 (직전 round prev_importance를 그대로 사용,
      반복 수렴 안 시도). Phase 4 결과로 수렴 알고리즘 필요 시 재검토.
    - quiet-but-important topic이 starve하지 않도록 ``min_floor``
      lower bound 적용.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from hi_em.config import get_section


def load_importance_config(path: Path | str | None = None) -> dict[str, Any]:
    """Return ``topic_importance`` section from ``configs/hiem.json``.

    Convenience wrapper over :func:`hi_em.config.get_section`. Missing file
    or missing section both fall back to module defaults baked into
    :mod:`hi_em.config`.
    """
    return get_section("topic_importance", path)


def _ema_frequency(rounds_with_mention: list[int], round_now: int, half_life: float) -> float:
    """EMA of per-round mention indicator with half-life ``half_life`` (rounds)."""
    if not rounds_with_mention or half_life <= 0:
        return 0.0
    decay = math.log(2) / half_life
    return sum(math.exp(-decay * (round_now - r)) for r in rounds_with_mention)


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
) -> dict[int, float]:
    """Compute per-topic importance scores.

    Args:
        state: ``ltm.load_state(conv_id)`` return value (has ``topics`` list).
        round_now: 현재 round index (0-based).
        mention_log: ``{topic_id: [round_idx, round_idx, ...]}`` — 어느 round
            에 그 topic이 등장했는지 누적 기록.
        prev_importance: 직전 round의 importance (재귀 1단계용). ``None``이면
            연결항(α₄)이 0.
        neighbor_weights: ``{topic_id: {other_topic_id: weight, ...}}`` — 인접
            관계 가중치. 미제공 시 연결항 0.
        alpha: (α₁, α₂, α₃, α₄) — 4 작용 가중치.
        lambda_r: recency decay rate (높을수록 빨리 잊음).
        lambda_freq: frequency EMA half-life (round 단위).
        min_floor: starvation floor.

    Returns:
        ``{topic_id: importance}``. 모든 score는 ``min_floor`` 이상.
    """
    if not state.get("topics"):
        return {}

    a1, a2, a3, a4 = alpha
    out: dict[int, float] = {}

    for t in state["topics"]:
        tid = t["topic_id"]
        n_t = t.get("count", 0)

        # 1) 강화 (turn 수)
        s1 = a1 * math.log(1 + n_t)

        # 2) 강화 (빈도) — EMA over per-round mention indicator
        rounds = mention_log.get(tid, [])
        s2 = a2 * _ema_frequency(rounds, round_now, lambda_freq)

        # 3) 망각 (recency) — 마지막 등장 round
        last_round = max(rounds) if rounds else round_now
        s3 = a3 * math.exp(-lambda_r * (round_now - last_round))

        # 4) 연결 (인접 + 통합) — prev_importance에 weight를 곱한 합 (1단계 재귀)
        s4 = 0.0
        if prev_importance and neighbor_weights and tid in neighbor_weights:
            for j, w in neighbor_weights[tid].items():
                s4 += w * prev_importance.get(j, 0.0)
        s4 *= a4

        out[tid] = max(min_floor, s1 + s2 + s3 + s4)

    return out
