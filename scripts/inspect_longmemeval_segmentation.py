#!/usr/bin/env python3
"""LongMemEval haystack 가 Hi-EM segmenter 로 어떻게 분절되는지 직접 확인.

LLM 없이 (encoder + sCRP 만으로) 한 question 의 haystack_sessions 를
orchestrator.preload_history 와 동일한 규칙으로 segmenter 에 투입한다:

  * ``role == "user"`` 턴만 임베딩 → ``seg.assign`` 으로 topic 배정.
  * ``role == "assistant"`` 턴은 직전 user 턴의 topic 을 상속 (임베딩 없음).

turn 모드 = 앞 N 턴의 (session / role / topic / boundary / has_answer / 본문).
question 모드 = 전체 haystack 으로 segmenter 구축 후, question 원문이
predict_topic 으로 어느 topic 에 가는지 + 정답 evidence 턴(has_answer=true)이
실제 배정된 topic 과 일치하는지.

Usage:
    uv run python scripts/inspect_longmemeval_segmentation.py \
        [--idx 0] [--turns 20] [--questions 0] \
        [--alpha 1.0 --lmda 10.0 --sigma0-sq 0.01]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.sem_core import HiEMSegmenter  # noqa: E402
from hi_em.sem_core_v334_rnn_var import HiEMSegmenterV334  # noqa: E402
from hi_em.sem_core_v335 import HiEMSegmenterV335  # noqa: E402
from hi_em.sem_core_v336 import HiEMSegmenterV336  # noqa: E402
from hi_em.sem_core_v337 import HiEMSegmenterV337  # noqa: E402
from hi_em.sem_core_v338 import HiEMSegmenterV338  # noqa: E402

DATA = REPO_ROOT / "benchmarks" / "LongMemEval" / "data" / "longmemeval_oracle.json"


def _fmt(text: str, width: int) -> str:
    """Single-line text, truncated to ``width`` chars (0 = no truncation)."""
    txt = text.replace("\n", " ")
    if width and len(txt) > width:
        txt = txt[: width - 3] + "..."
    return txt


def _flatten(entry: dict) -> list[dict]:
    """haystack_sessions → flat turn list with session index attached.

    각 turn: {sess, role, content, has_answer}.
    """
    flat: list[dict] = []
    sessions = entry["haystack_sessions"]
    for si, sess in enumerate(sessions, start=1):
        for t in sess:
            flat.append(
                {
                    "sess": si,
                    "role": t.get("role", "user"),
                    "content": t.get("content", ""),
                    "has_answer": bool(t.get("has_answer", False)),
                }
            )
    return flat


def _segment(flat: list[dict], enc: QueryEncoder, args) -> tuple[HiEMSegmenter, list[dict]]:
    """preload_history 규칙대로 segmenter 구축. flat 각 dict 에
    ``topic`` / ``is_boundary`` 를 채워 넣고 (seg, flat) 반환."""
    user_idx = [i for i, t in enumerate(flat) if t["role"] == "user"]
    user_embs = enc.encode([flat[i]["content"] for i in user_idx]) if user_idx else None
    dim = user_embs.shape[1] if user_embs is not None else enc.dim

    if args.version == "v3.3.8":
        # v3.3.8: v3.3.7 + SEM2-calibrated fresh baseline (pe_prior) +
        # non-prev topics scored by f0-likelihood.
        seg = HiEMSegmenterV338(dim=dim, alpha=args.alpha, lmda=args.lmda)
    elif args.version == "v3.3.7":
        # v3.3.7: v3.3.6 dynamics + SEM2 map_variance posterior σ² (n≥2).
        seg = HiEMSegmenterV337(dim=dim, alpha=args.alpha, lmda=args.lmda)
    elif args.version == "v3.3.6":
        # v3.3.6: SEM2-faithful event dynamics (persistence + replay).
        seg = HiEMSegmenterV336(dim=dim, alpha=args.alpha, lmda=args.lmda)
    elif args.version == "v3.3.5":
        # v3.3.5: SEM2 f_is_trained gating; sigma0_sq HP 무효 (pe_var_* 사용).
        seg = HiEMSegmenterV335(dim=dim, alpha=args.alpha, lmda=args.lmda)
    elif args.version == "v3.3.4":
        # NOTE: v3.3.4 has no sigma0_sq HP (uses pe_var_* instead);
        # --sigma0-sq is ignored here. alpha/lmda 만 적용.
        seg = HiEMSegmenterV334(dim=dim, alpha=args.alpha, lmda=args.lmda)
    else:
        seg = HiEMSegmenter(
            dim=dim, alpha=args.alpha, lmda=args.lmda, sigma0_sq=args.sigma0_sq
        )
    emb_by_idx = {idx: user_embs[k] for k, idx in enumerate(user_idx)}

    last_topic = 0
    for i, t in enumerate(flat):
        if t["role"] == "user":
            tid, is_bnd = seg.assign(emb_by_idx[i])
            last_topic = tid
            t["topic"], t["is_boundary"] = tid, is_bnd
        else:  # assistant inherits prev user topic, no boundary
            t["topic"], t["is_boundary"] = last_topic, False
    return seg, flat


def inspect_turns(entry: dict, args) -> None:
    flat = _flatten(entry)
    enc = QueryEncoder()
    seg, flat = _segment(flat, enc, args)
    pool = [t for t in flat if t["role"] == "user"] if args.user_only else flat
    shown = pool[: args.turns]

    print(
        f"idx={args.idx} qid={entry.get('question_id')} "
        f"qtype={entry.get('question_type')}"
    )
    print(
        f"segmenter={args.version}  alpha={args.alpha} lmda={args.lmda}"
        + (f" sigma0_sq={args.sigma0_sq}" if args.version == "v2"
           else f" (sigma0_sq 무효: {args.version} 는 pe_var_* 사용)")
    )
    print(
        f"haystack: {len(entry['haystack_sessions'])} sessions / "
        f"{len(flat)} turns — inspecting first {len(shown)}\n"
    )
    print(f"{'#':>3} {'sess':>4} {'role':>5} {'topic':>5} {'bnd':>3} {'ans':>3}  text")
    print("-" * 110)
    prev_sess = None
    n_bnd = 0
    for i, t in enumerate(shown):
        if prev_sess is not None and t["sess"] != prev_sess:
            print(f"    --- session {t['sess']} 시작 ---")
        prev_sess = t["sess"]
        n_bnd += int(t["is_boundary"])
        bnd = "▶" if t["is_boundary"] else " "
        ans = "★" if t["has_answer"] else " "
        txt = _fmt(t["content"], args.width)
        print(
            f"{i:>3} {t['sess']:>4} {t['role'][:5]:>5} {t['topic']:>5} "
            f"{bnd:>3} {ans:>3}  {txt}"
        )

    n_topics = len([c for c in seg.counts if c > 0])
    n_ans = sum(1 for t in shown if t["has_answer"])
    print(
        f"\n표시 {len(shown)}턴 → 토픽 {n_topics}개(전체 haystack 기준), "
        f"boundary {n_bnd}회, has_answer 턴 {n_ans}개 (★)"
    )


def inspect_questions(entry: dict, args) -> None:
    """question 원문 → predict_topic vs has_answer 턴들의 실제 topic."""
    flat = _flatten(entry)
    enc = QueryEncoder()
    seg, flat = _segment(flat, enc, args)

    ans_topics = sorted({t["topic"] for t in flat if t["has_answer"]})
    q = entry["question"]
    q_emb = enc.encode(q)
    pred = seg.predict_topic(q_emb)
    hit = pred in ans_topics if ans_topics else None
    mark = "✓" if hit else ("✗" if hit is False else "·")

    n_topics = len([c for c in seg.counts if c > 0])
    print(
        f"idx={args.idx} qid={entry.get('question_id')} "
        f"qtype={entry.get('question_type')}"
    )
    print(
        f"HP: alpha={args.alpha} lmda={args.lmda} sigma0_sq={args.sigma0_sq}\n"
        f"haystack: {len(entry['haystack_sessions'])} sessions / "
        f"{len(flat)} turns → 토픽 {n_topics}개\n"
    )
    print(f"Q: {q}")
    print(f"A: {entry.get('answer')}")
    print(f"answer_session_ids: {entry.get('answer_session_ids')}")
    print(
        f"\npredict_topic = {pred}   "
        f"has_answer 턴 실제 topic = {ans_topics or '-'}   {mark}"
    )
    print("\n--- has_answer=true 턴 (정답 근거) ---")
    for t in flat:
        if t["has_answer"]:
            txt = _fmt(t["content"], args.width)
            print(f"  sess{t['sess']} {t['role']:>9} topic={t['topic']:>3}  {txt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="question entry index")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument(
        "--questions",
        type=int,
        default=0,
        help=">0 이면 question 모드 (predict_topic vs evidence topic)",
    )
    ap.add_argument(
        "--user-only",
        action="store_true",
        help="user 턴만 출력 (assign 은 user 에서만 일어남; assistant 는 "
        "직전 user topic 상속일 뿐이라 분절 관찰엔 노이즈)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=80,
        help="턴 본문 출력 최대 길이 (0 = 자르지 않고 전문 출력)",
    )
    ap.add_argument(
        "--version",
        choices=["v2", "v3.3.4", "v3.3.5", "v3.3.6", "v3.3.7", "v3.3.8"],
        default="v2",
        help="segmenter: v2=HiEMSegmenter(SEM core), v3.3.4=HiEMSegmenterV334, "
        "v3.3.5=HiEMSegmenterV335 (SEM2 f_is_trained gating), "
        "v3.3.6=HiEMSegmenterV336 (SEM2 persistence+replay dynamics), "
        "v3.3.7=HiEMSegmenterV337 (+ SEM2 map_variance posterior σ²), "
        "v3.3.8=HiEMSegmenterV338 (+ SEM2 fresh baseline + non-prev f0)",
    )
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lmda", type=float, default=10.0)
    ap.add_argument("--sigma0-sq", type=float, default=0.01)
    args = ap.parse_args()

    entries = json.loads(DATA.read_text())
    entry = entries[args.idx]

    if args.questions > 0:
        inspect_questions(entry, args)
    else:
        inspect_turns(entry, args)


if __name__ == "__main__":
    main()
