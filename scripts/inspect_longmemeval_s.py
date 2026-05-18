#!/usr/bin/env python3
"""LongMemEval-S 의 한 질문에서 *증거(관련) 세션* 을 distractor 와 분리해 확인.

longmemeval_s 는 질문당 ~40+ 세션 (대부분 distractor) 이다. 각 entry 는
``answer_session_ids`` (증거 세션 id, 세션레벨 recall 라벨) 와 evidence turn
의 ``has_answer: true`` (턴레벨 라벨) 를 갖는다. 이 스크립트는 그 라벨로
haystack 을 증거 세션 / distractor 로 가른다.

* ``--qid`` 로 지정하면 oracle 과 같은 질문을 (s 의 idx 정렬이 oracle 과
  달라도) 정확히 찾는다. ``--idx`` 는 s_cleaned 파일 내 위치.
* 기본 출력: 전체 세션 수 / 증거 세션 id / distractor 수 + 증거 세션의
  user 턴만 (★=has_answer) — ``longmemeval_idx*_useronly.txt`` 형식.
* ``--with-distractors`` 면 distractor 세션 목록(요약)도 출력.

Usage:
    uv run python scripts/inspect_longmemeval_s.py --qid gpt4_a1b77f9c
    uv run python scripts/inspect_longmemeval_s.py --idx 0 --out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "benchmarks" / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
OUT_DIR = REPO / "outputs" / "runs" / "_misc"


def _find(entries: list[dict], idx: int | None, qid: str | None) -> tuple[int, dict]:
    if qid is not None:
        for i, e in enumerate(entries):
            if e.get("question_id") == qid:
                return i, e
        sys.exit(f"qid {qid!r} not found in {DATA.name}")
    if idx is None:
        sys.exit("provide --idx or --qid")
    if not 0 <= idx < len(entries):
        sys.exit(f"--idx out of range [0, {len(entries)})")
    return idx, entries[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None, help="position in s_cleaned")
    ap.add_argument("--qid", type=str, default=None, help="question_id (oracle-stable)")
    ap.add_argument(
        "--with-distractors",
        action="store_true",
        help="distractor 세션 요약도 출력",
    )
    ap.add_argument("--out", action="store_true", help="outputs/runs/_misc/ 에 저장")
    args = ap.parse_args()

    entries = json.loads(DATA.read_text())
    idx, e = _find(entries, args.idx, args.qid)

    sids: list[str] = e["haystack_session_ids"]
    sessions: list[list[dict]] = e["haystack_sessions"]
    dates: list[str] = e.get("haystack_dates") or [""] * len(sids)
    ev_ids = set(e.get("answer_session_ids") or [])

    ev_idx = [i for i, sid in enumerate(sids) if sid in ev_ids]
    dis_idx = [i for i in range(len(sids)) if i not in set(ev_idx)]

    L: list[str] = []
    L.append(
        f"# LongMemEval-S  s_idx={idx}  qid={e.get('question_id')}  "
        f"qtype={e.get('question_type')}"
    )
    L.append(
        f"# 전체 {len(sids)} 세션 = 증거 {len(ev_idx)} + distractor {len(dis_idx)}"
    )
    L.append(f"# question_date: {e.get('question_date')}")
    L.append("")
    L.append(f"## QUESTION: {e['question']}")
    L.append(f"## ANSWER: {e.get('answer')}")
    L.append(f"## answer_session_ids: {e.get('answer_session_ids')}")
    L.append("")
    L.append("=" * 100)
    L.append("## 증거(관련) 세션 — user 턴만, ★=has_answer")
    L.append("=" * 100)

    uid = 0
    for rank, i in enumerate(ev_idx, 1):
        L.append("")
        L.append(
            f"### 증거세션 {rank}  (haystack#{i}  id={sids[i]}  date={dates[i]})"
        )
        for t in sessions[i]:
            if t.get("role", "user") != "user":
                continue
            uid += 1
            star = " ★" if t.get("has_answer") else ""
            L.append("")
            L.append(f"[#{uid}]{star} {t.get('content', '')}")

    if args.with_distractors:
        L.append("")
        L.append("=" * 100)
        L.append(f"## distractor 세션 {len(dis_idx)}개 (요약: id · date · 첫 user턴)")
        L.append("=" * 100)
        for i in dis_idx:
            first_u = next(
                (
                    t.get("content", "")
                    for t in sessions[i]
                    if t.get("role", "user") == "user"
                ),
                "",
            )
            L.append(
                f"  #{i:>3} {sids[i]}  {dates[i]}  | "
                f"{first_u[:90].replace(chr(10), ' ')}"
            )

    text = "\n".join(L)
    print(text if not args.out else text[: text.find("=" * 100)])
    if args.out:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fn = OUT_DIR / f"longmemeval_s_{e.get('question_id')}_evidence.txt"
        fn.write_text(text)
        print(f"\nwritten {fn.relative_to(REPO)}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
