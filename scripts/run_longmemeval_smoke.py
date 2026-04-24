#!/usr/bin/env python3
"""Phase 2.5 smoke test — Hi-EM segmenter on LongMemEval oracle.

Purpose
    Verify that Hi-EM (option A + persistence HP) approximately recovers
    the session structure of LongMemEval oracle conversations. This is
    the early sanity check required by plan.md before Phase 2 (LTM +
    Memory window) is designed.

Input
    benchmarks/LongMemEval/data/longmemeval_oracle.json
    - 500 question entries
    - Each entry has 1–6 evidence sessions
    - Each session = list of turns {role, content, has_answer}

Scene vector
    Per design rule (context/01-hi-em-design.md §1), only **user**
    utterances are embedded. Assistant responses are ignored as scene
    inputs.

Ground-truth proxy
    In LongMemEval there is no per-turn topic label. We treat the
    session boundary (crossing ``haystack_session_ids``) as a topic
    boundary — a weak but natural proxy (each session is assumed to be
    one coherent topic).

Metrics
    session-boundary recall   : P(Hi-EM predicts boundary | true session shift)
    within-session purity avg : within each session, the fraction of user
                                turns assigned to the single dominant Hi-EM
                                topic. 1.0 = perfect.
    avg topics per session    : total distinct Hi-EM topics / total sessions.
                                ~1.0 = good (no over-segmentation inside a
                                session).

HP regimes
    persistence   α=1,  λ=10, σ₀²=0.01  (design-target for Claude-style chat)
    freq-shift    α=10, λ=1,  σ₀²=0.1   (TopiOCQA-tuned winner — comparison)

Also: cosine-θ sweep baseline for reference.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.sem_core import HiEMSegmenter  # noqa: E402


# --- Data --------------------------------------------------------------

def load_oracle() -> list[dict]:
    path = REPO_ROOT / "benchmarks" / "LongMemEval" / "data" / "longmemeval_oracle.json"
    return json.loads(path.read_text())


def user_turns_with_session_idx(entry: dict) -> list[tuple[int, str]]:
    """Return [(session_idx, content)] for user turns only, in session order."""
    out: list[tuple[int, str]] = []
    for si, session in enumerate(entry["haystack_sessions"]):
        for turn in session:
            if turn.get("role") == "user":
                out.append((si, turn.get("content", "")))
    return out


# --- Metrics -----------------------------------------------------------

def smoke_metrics(
    oracle: list[dict],
    per_entry_emb: list[np.ndarray],
    per_entry_turns: list[list[tuple[int, str]]],
    alpha: float,
    lmda: float,
    sigma0_sq: float,
) -> dict:
    sb_num = sb_den = 0          # session-boundary recall
    sb_fp = sb_tn = 0            # false positives / true negatives
    purity_sum = 0.0
    purity_count = 0
    tot_topics = 0
    tot_sessions = 0
    tot_turns = 0
    tot_boundaries_pred = 0
    tot_entries = 0

    for entry, emb, turns in zip(oracle, per_entry_emb, per_entry_turns):
        if len(turns) < 2:
            continue
        tot_entries += 1
        seg = HiEMSegmenter(
            dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq,
        )
        assignments = [seg.assign(s)[0] for s in emb]
        gt_shift = [turns[i][0] != turns[i - 1][0] for i in range(1, len(turns))]
        pred_shift = [assignments[i] != assignments[i - 1] for i in range(1, len(assignments))]

        for g, p in zip(gt_shift, pred_shift):
            if g:
                sb_den += 1
                if p:
                    sb_num += 1
            else:
                if p:
                    sb_fp += 1
                else:
                    sb_tn += 1

        sess_topics: dict[int, list[int]] = defaultdict(list)
        for (si, _), k in zip(turns, assignments):
            sess_topics[si].append(k)
        for ks in sess_topics.values():
            cnt = Counter(ks)
            dom = cnt.most_common(1)[0][1]
            purity_sum += dom / len(ks)
            purity_count += 1

        tot_turns += len(turns)
        tot_topics += len(set(assignments))
        tot_sessions += len(sess_topics)
        tot_boundaries_pred += sum(pred_shift)

    return {
        "entries_scored": tot_entries,
        "total_user_turns": tot_turns,
        "total_sessions": tot_sessions,
        "gt_session_boundaries": sb_den,
        "hi_em_boundaries": tot_boundaries_pred,
        "session_boundary_recall": sb_num / sb_den if sb_den else None,
        "false_positive_rate": sb_fp / (sb_fp + sb_tn) if (sb_fp + sb_tn) else None,
        "within_session_purity_avg": purity_sum / purity_count if purity_count else None,
        "avg_topics_per_session": tot_topics / tot_sessions if tot_sessions else None,
    }


def cosine_baseline_metrics(
    oracle: list[dict],
    per_entry_emb: list[np.ndarray],
    per_entry_turns: list[list[tuple[int, str]]],
) -> dict:
    """Best-θ cosine baseline: shift if cos(s_i, s_{i-1}) < θ."""
    best = {"F1": -1.0}
    for thr in np.arange(0.3, 0.95, 0.025):
        tp = fp = fn = tn = 0
        for emb, turns in zip(per_entry_emb, per_entry_turns):
            if len(turns) < 2:
                continue
            for i in range(1, len(turns)):
                gt = turns[i][0] != turns[i - 1][0]
                cos = float(np.dot(emb[i], emb[i - 1]))
                pred = cos < thr
                if gt and pred: tp += 1
                elif gt and not pred: fn += 1
                elif not gt and pred: fp += 1
                else: tn += 1
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        if F > best["F1"]:
            best = {
                "threshold": float(thr), "P": P, "R": R, "F1": F,
                "FPR": fp / (fp + tn) if (fp + tn) else None,
            }
    return best


# --- Report ------------------------------------------------------------

def _fmt(v, d=3):
    return f"{v:.{d}f}" if v is not None else "n/a"


def write_report(out_path: Path, *, oracle, turn_stats, encode_info, cos, pers, freq):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# Phase 2.5 — LongMemEval oracle smoke test

실행: `python scripts/run_longmemeval_smoke.py` (device={encode_info['device']})

## 데이터
- 500 question entries (oracle)
- 총 user turns 임베딩: {encode_info['n_user_turns']}
- 평균 user turn 길이 (chars): {turn_stats['avg_chars']:.0f}
- 최대 user turn 길이: {turn_stats['max_chars']}
- bge-base 512-token cap 초과 추정 (>2000 chars): {turn_stats['long_turns']}개

## 인코딩
- 모델: {encode_info['model']} (dim={encode_info['dim']})
- 전체 시간: {encode_info['encode_sec']:.1f}s
- 턴당: {encode_info['ms_per_turn']:.1f} ms

## Ground truth (weak proxy)
session 전환을 topic 전환의 proxy로 사용. LongMemEval는 turn-level topic label이 없으므로 강한 GT가 아니라 **sanity check**임.

## Cosine baseline
- best θ={_fmt(cos['threshold'])} → P={_fmt(cos['P'])} R={_fmt(cos['R'])} F1={_fmt(cos['F1'])}
- FPR (boundary 오탐율): {_fmt(cos['FPR'])}

## Hi-EM — Persistence HP (α=1, λ=10, σ₀²=0.01)  [design-target]
- session-boundary recall : {_fmt(pers['session_boundary_recall'])}
- false positive rate     : {_fmt(pers['false_positive_rate'])}
- within-session purity   : {_fmt(pers['within_session_purity_avg'])}
- avg topics per session  : {_fmt(pers['avg_topics_per_session'], 2)}
- total Hi-EM boundaries  : {pers['hi_em_boundaries']}  (vs {pers['gt_session_boundaries']} gt)

## Hi-EM — Freq-shift HP (α=10, λ=1, σ₀²=0.1)  [참조, TopiOCQA 튜닝값]
- session-boundary recall : {_fmt(freq['session_boundary_recall'])}
- false positive rate     : {_fmt(freq['false_positive_rate'])}
- within-session purity   : {_fmt(freq['within_session_purity_avg'])}
- avg topics per session  : {_fmt(freq['avg_topics_per_session'], 2)}

## 해석 기준
- 이상적: session-boundary recall ≥ 0.7, within-session purity ≥ 0.8, avg topics/session ≈ 1.0
- Persistence HP가 freq-shift HP보다 우수해야 '대화 persistence 가정'이 실증적으로 뒷받침됨.
- 두 HP 모두 부실하면 옵션 A 자체 재검토 — plan.md Phase 2.5 FAIL 경로 (= Phase 1-4와 동일): `06-decision-log.md` append + 옵션 D escalation.
"""
    out_path.write_text(md)


def main() -> None:
    oracle = load_oracle()
    print(f"[data] {len(oracle)} oracle entries")

    per_entry_turns = [user_turns_with_session_idx(e) for e in oracle]
    all_texts = [c for ut in per_entry_turns for (_, c) in ut]
    n_turns = len(all_texts)
    lens = [len(t) for t in all_texts]
    long_turns = sum(1 for L in lens if L > 2000)  # rough ~500 tokens cap proxy
    print(f"[stats] {n_turns} user turns; avg {np.mean(lens):.0f} chars, max {max(lens)}, long(>2k):{long_turns}")

    print("[encode] loading bge-base-en-v1.5...")
    enc = QueryEncoder()
    t0 = time.perf_counter()
    emb_flat = np.asarray(enc.encode(all_texts))
    enc_sec = time.perf_counter() - t0
    print(f"  encoded {n_turns} in {enc_sec:.1f}s ({enc_sec/n_turns*1000:.1f} ms/turn) on {enc.device}")

    per_entry_emb = []
    idx = 0
    for ut in per_entry_turns:
        per_entry_emb.append(emb_flat[idx:idx + len(ut)])
        idx += len(ut)

    print("[baseline] cosine-θ sweep...")
    cos = cosine_baseline_metrics(oracle, per_entry_emb, per_entry_turns)
    print(f"  cosine best: θ={cos['threshold']:.3f} F1={cos['F1']:.3f} "
          f"(P={cos['P']:.3f} R={cos['R']:.3f}, FPR={cos['FPR']:.3f})")

    print("[Hi-EM] persistence HP (α=1, λ=10, σ₀²=0.01)...")
    pers = smoke_metrics(oracle, per_entry_emb, per_entry_turns,
                         alpha=1.0, lmda=10.0, sigma0_sq=0.01)
    print(f"  session-boundary recall: {pers['session_boundary_recall']:.3f}")
    print(f"  within-session purity  : {pers['within_session_purity_avg']:.3f}")
    print(f"  avg topics/session     : {pers['avg_topics_per_session']:.2f}")
    print(f"  false positive rate    : {pers['false_positive_rate']:.3f}")

    print("[Hi-EM] freq-shift HP (α=10, λ=1, σ₀²=0.1)...")
    freq = smoke_metrics(oracle, per_entry_emb, per_entry_turns,
                         alpha=10.0, lmda=1.0, sigma0_sq=0.1)
    print(f"  session-boundary recall: {freq['session_boundary_recall']:.3f}")
    print(f"  within-session purity  : {freq['within_session_purity_avg']:.3f}")
    print(f"  avg topics/session     : {freq['avg_topics_per_session']:.2f}")
    print(f"  false positive rate    : {freq['false_positive_rate']:.3f}")

    turn_stats = {
        "avg_chars": float(np.mean(lens)),
        "max_chars": int(max(lens)),
        "long_turns": int(long_turns),
    }
    encode_info = {
        "model": "BAAI/bge-base-en-v1.5",
        "dim": int(emb_flat.shape[1]),
        "n_user_turns": n_turns,
        "encode_sec": enc_sec,
        "ms_per_turn": enc_sec / n_turns * 1000,
        "device": enc.device,
    }

    (REPO_ROOT / "outputs" / "phase-2.5-smoke.json").write_text(json.dumps({
        "turn_stats": turn_stats,
        "encode": encode_info,
        "cosine_baseline": cos,
        "hi_em_persistence": pers,
        "hi_em_freq_shift": freq,
    }, indent=2))

    write_report(
        REPO_ROOT / "outputs" / "phase-2.5-smoke.md",
        oracle=oracle, turn_stats=turn_stats, encode_info=encode_info,
        cos=cos, pers=pers, freq=freq,
    )
    print(f"report → {REPO_ROOT / 'outputs' / 'phase-2.5-smoke.md'}")


if __name__ == "__main__":
    main()
