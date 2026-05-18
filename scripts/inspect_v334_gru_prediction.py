#!/usr/bin/env python3
"""v3.3.4 의 EventRNN(GRU) 이 예측한 '다음 턴 임베딩' 을 직접 확인.

한 LoCoMo conversation 의 user 턴을 HiEMSegmenterV334 에 순차 투입하면서,
각 턴 직전에 *그 턴이 배정될 토픽* 이 들고 있던 예측 임베딩을 꺼내
실제 턴 임베딩과 비교한다:

  * pred_mode = centroid (n < rnn_min_history) | GRU (n >= rnn_min_history)
  * cos(pred, actual)      = 1 - PE   (그 토픽이 이 턴을 얼마나 잘 예측했나)
  * cos(centroid, actual)  = GRU 모드일 때 평균 centroid 와의 대조군
  * pred 벡터 앞 6개 dim + L2 norm (예측 벡터 자체를 눈으로)

마지막에 centroid-mode vs GRU-mode 평균 cos 를 비교해, GRU 예측이
centroid 평균보다 실제로 나은지 한눈에 본다.

Usage:
    uv run python scripts/inspect_v334_gru_prediction.py \
        [--conv 0] [--turns 40] [--topic K]   # --topic 지정 시 그 토픽만
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.locomo_loader import _build_haystack_sessions  # noqa: E402
from hi_em.sem_core_v334_rnn_var import HiEMSegmenterV334  # noqa: E402

DATA = REPO_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=int, default=0)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--topic", type=int, default=-1, help="이 토픽만 출력 (-1=전체)")
    # v3.3.4 default HP (orchestrator/run_experiment 기본과 동일)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--lmda", type=float, default=10.0)
    ap.add_argument("--rnn-min-history", type=int, default=2)
    args = ap.parse_args()

    sample = json.loads(DATA.read_text())[args.conv]
    sessions = _build_haystack_sessions(sample["conversation"])
    flat = [t for sess in sessions for t in sess][: args.turns]

    enc = QueryEncoder()
    embs = np.asarray(enc.encode([t["content"] for t in flat]))

    seg = HiEMSegmenterV334(
        dim=embs.shape[1],
        alpha=args.alpha,
        lmda=args.lmda,
        rnn_min_history=args.rnn_min_history,
    )

    print(
        f"conv={args.conv} sample_id={sample.get('sample_id')}  "
        f"v3.3.4 (alpha={args.alpha} lmda={args.lmda} "
        f"rnn_min_history={args.rnn_min_history})"
    )
    print(
        "pred = 그 토픽이 이 턴 직전에 들고 있던 '다음 턴' 예측 임베딩\n"
        "cos = cos(pred, actual)=1-PE · cosμ = cos(centroid, actual) 대조군\n"
    )
    print(
        f"{'#':>3} {'topic':>5} {'mode':>8} {'n':>3} {'cos':>7} {'cosμ':>7} "
        f"{'PE':>6}  pred[:6] (‖pred‖)"
    )
    print("-" * 118)

    gru_cos: list[float] = []
    cen_cos: list[float] = []
    for i, (t, s) in enumerate(zip(flat, embs)):
        # 이 턴이 어느 토픽에 갈지: 모든 활성 토픽 중 가장 높은 score.
        # 굳이 점수 재현하지 않고, assign 직전 토픽별 예측을 스냅샷한 뒤
        # assign 결과 k 의 스냅샷을 사용 (assign 이 곧 update 도 수행).
        snap_pred = {k: tp.predict_next().copy() for k, tp in enumerate(seg.topics)}
        snap_n = {k: tp.n for k, tp in enumerate(seg.topics)}
        snap_mu = {k: tp.mu.copy() for k, tp in enumerate(seg.topics)}

        k, is_bnd = seg.assign(s)

        if args.topic >= 0 and k != args.topic:
            continue

        if k in snap_pred and snap_n[k] > 0:
            pred = snap_pred[k]
            n_before = snap_n[k]
            mode = "GRU" if n_before >= args.rnn_min_history else "centroid"
            cos = float(np.dot(pred, s))
            mu = snap_mu[k]
            cosmu = float(np.dot(mu, s)) if np.linalg.norm(mu) > 0 else float("nan")
            pe = 1.0 - cos
            if mode == "GRU":
                gru_cos.append(cos)
                if not np.isnan(cosmu):
                    cen_cos.append(cosmu)
            head = " ".join(f"{x:+.3f}" for x in pred[:6])
            nrm = float(np.linalg.norm(pred))
            print(
                f"{i:>3} {k:>5} {mode:>8} {n_before:>3} {cos:>7.3f} "
                f"{cosmu:>7.3f} {pe:>6.3f}  [{head}] ({nrm:.3f})"
            )
        else:
            # 새로 생성된 토픽 (이 턴이 첫 멤버) → 예측 없음
            print(
                f"{i:>3} {k:>5} {'NEW':>8} {0:>3} {'-':>7} {'-':>7} "
                f"{'-':>6}  (이 턴이 토픽 {k} 의 첫 멤버 — 예측 대상 아님)"
            )

    print("-" * 118)
    if gru_cos:
        print(
            f"GRU-mode 턴 {len(gru_cos)}개  평균 cos(GRU pred, actual) = "
            f"{np.mean(gru_cos):.4f}"
        )
    if cen_cos:
        print(
            f"같은 턴들의 평균 cos(centroid, actual)         = "
            f"{np.mean(cen_cos):.4f}   "
            f"(GRU 가 centroid 보다 {'높음' if gru_cos and cen_cos and np.mean(gru_cos) > np.mean(cen_cos) else '낮음/동등'})"
        )
    if not gru_cos:
        print("GRU-mode 턴 없음 — 토픽들이 rnn_min_history 만큼 안 쌓임 "
              "(--turns 늘리거나 --topic 으로 길게 유지되는 토픽 지정).")


if __name__ == "__main__":
    main()
