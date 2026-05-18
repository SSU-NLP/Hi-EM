#!/usr/bin/env python3
"""LoCoMo 대화가 Hi-EM segmenter 에 의해 실제로 어떻게 분절되는지 직접 확인.

LLM 없이 (encoder + sCRP 만으로) 한 conversation 의 앞 N 턴을 segmenter 에
순차 투입하고, 턴별 (session / speaker / topic_id / boundary / 본문) 을
그대로 출력한다. 평가/생성 없이 segmentation 동작만 들여다보는 용도.

Usage:
    uv run python scripts/inspect_locomo_segmentation.py \
        [--conv 0] [--turns 20] [--alpha 1.0 --lmda 10.0 --sigma0-sq 0.01]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.locomo_loader import _build_haystack_sessions  # noqa: E402
from hi_em.sem_core import HiEMSegmenter  # noqa: E402

DATA = REPO_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"


def inspect_questions(sample: dict, sessions: list, args) -> None:
    """전체 대화로 segmenter 를 구축한 뒤, 각 질문 원문이 predict_topic 으로
    어느 topic 에 매핑되는지 + 그 질문의 evidence 턴이 실제 어느 topic 에
    배정됐는지를 나란히 출력 (질문→토픽 매칭이 맞는지 직접 확인용)."""
    flat: list[dict] = [t for sess in sessions for t in sess]

    enc = QueryEncoder()
    turn_embs = enc.encode([t["content"] for t in flat])

    seg = HiEMSegmenter(
        dim=turn_embs.shape[1],
        alpha=args.alpha,
        lmda=args.lmda,
        sigma0_sq=args.sigma0_sq,
    )
    dia2topic: dict[str, int] = {}
    for t, e in zip(flat, turn_embs):
        tid, _ = seg.assign(e)
        if t.get("dia_id"):
            dia2topic[t["dia_id"]] = tid

    qa = [q for q in sample["qa"] if "question" in q][: args.questions]
    q_embs = enc.encode([q["question"] for q in qa])

    n_topics = len([c for c in seg.counts if c > 0])
    print(
        f"conv={args.conv} sample_id={sample.get('sample_id')}  "
        f"전체 {len(flat)}턴 → 토픽 {n_topics}개"
    )
    print(
        f"HP: alpha={args.alpha} lmda={args.lmda} sigma0_sq={args.sigma0_sq}\n"
        f"predict_topic = 질문이 매핑된 topic / evid_topics = "
        f"evidence 턴이 실제 배정된 topic (✓ = 일치)\n"
    )

    n_hit = 0
    for i, (q, qe) in enumerate(zip(qa, q_embs)):
        pred = seg.predict_topic(qe)
        ev = q.get("evidence", []) or []
        ev_topics = sorted({dia2topic[d] for d in ev if d in dia2topic})
        hit = pred in ev_topics if ev_topics else None
        if hit:
            n_hit += 1
        mark = "✓" if hit else ("✗" if hit is False else "·")
        ans = str(q.get("answer", ""))
        if len(ans) > 40:
            ans = ans[:37] + "..."
        print(
            f"[{i:>3}] cat{q.get('category')} pred_topic={pred:>3} "
            f"evid_topics={ev_topics or '-'} {mark}"
        )
        print(f"      Q: {q['question']}")
        print(f"      A: {ans}   evidence={ev}")

    scored = sum(1 for q in qa if (q.get('evidence') or []))
    print(
        f"\n질문 {len(qa)}개 중 evidence 보유 {scored}개 / "
        f"predict_topic 이 evidence topic 과 일치 {n_hit}개"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=int, default=0, help="conversation index")
    ap.add_argument("--turns", type=int, default=20, help="number of turns to inspect")
    ap.add_argument(
        "--questions",
        type=int,
        default=0,
        help="질문 모드: 전체 대화로 segmenter 를 만든 뒤 앞 N개 질문의 "
        "원문 + predict_topic 결과 출력 (>0 이면 이 모드)",
    )
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lmda", type=float, default=10.0)
    ap.add_argument("--sigma0-sq", type=float, default=0.01)
    args = ap.parse_args()

    samples = json.loads(DATA.read_text())
    sample = samples[args.conv]
    sessions = _build_haystack_sessions(sample["conversation"])

    if args.questions > 0:
        inspect_questions(sample, sessions, args)
        return

    # flatten sessions → (session_idx, turn_dict)
    flat: list[tuple[int, dict]] = []
    for si, sess in enumerate(sessions, start=1):
        for t in sess:
            flat.append((si, t))
    flat = flat[: args.turns]

    print(
        f"conv={args.conv} sample_id={sample.get('sample_id')}  "
        f"speakers: A={sample['conversation']['speaker_a']} / "
        f"B={sample['conversation']['speaker_b']}"
    )
    print(
        f"HP: alpha={args.alpha} lmda={args.lmda} sigma0_sq={args.sigma0_sq}  "
        f"(persistence default, SEM core v2)"
    )
    print(f"inspecting first {len(flat)} turns\n")

    enc = QueryEncoder()
    embs = enc.encode([t["content"] for _, t in flat])

    seg = HiEMSegmenter(
        dim=embs.shape[1],
        alpha=args.alpha,
        lmda=args.lmda,
        sigma0_sq=args.sigma0_sq,
    )

    print(f"{'#':>3} {'sess':>4} {'topic':>5} {'bnd':>3}  text")
    print("-" * 100)
    prev_sess = None
    n_bnd = 0
    for i, ((si, t), e) in enumerate(zip(flat, embs)):
        tid, is_bnd = seg.assign(e)
        n_bnd += int(is_bnd)
        if prev_sess is not None and si != prev_sess:
            print(f"    --- session {si} 시작 ---")
        prev_sess = si
        mark = "▶" if is_bnd else " "
        txt = t["content"].replace("\n", " ")
        if len(txt) > 80:
            txt = txt[:77] + "..."
        print(f"{i:>3} {si:>4} {tid:>5} {mark:>3}  {txt}")

    n_topics = len([c for c in seg.counts if c > 0])
    print(f"\n총 {len(flat)} 턴 → 토픽 {n_topics}개, boundary {n_bnd}회")


if __name__ == "__main__":
    main()
