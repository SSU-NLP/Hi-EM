#!/usr/bin/env python3
"""LongMemEval-S 한 질문(qid)의 user 턴을 segmenter 로 분절하고 topic별
그룹으로 출력 (oracle 의 ``longmemeval_idx*_vXXX_bytopic.txt`` 형식 동일).

-s 는 질문당 ~49 세션 (대부분 distractor). oracle 과 idx 정렬이 다르므로
``--qid`` 로 같은 질문을 정확히 찾는다 (oracle idx374 = qid gpt4_a1b77f9c).

orchestrator.preload_history 규칙: ``role=="user"`` 턴만 임베딩→assign.
encoder = 로컬 QueryEncoder (결정적; 기존 bytopic 파일과 동일 계열).

Usage:
    uv run python scripts/inspect_longmemeval_s_bytopic.py \
        --qid gpt4_a1b77f9c --version v3.1.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DATA = REPO / "benchmarks" / "LongMemEval" / "data" / "longmemeval_s_cleaned.json"
OUT_DIR = REPO / "outputs" / "runs" / "_misc"

from hi_em.embedding import QueryEncoder  # noqa: E402


def _build_segmenter(version: str, dim: int, alpha: float, lmda: float,
                     tau: float, cos_threshold: float):
    if version == "v3.1.1":
        from hi_em.sem_core_optimize import HiEMSegmenterV3
        return HiEMSegmenterV3(
            dim=dim, alpha=alpha, lmda=lmda, tau=tau,
            cos_threshold=cos_threshold,
        ), "Bounded Cosine MAP (no dynamics)"
    if version == "v2":
        from hi_em.sem_core import HiEMSegmenter
        return HiEMSegmenter(dim=dim, alpha=alpha, lmda=lmda), "SEM core"
    if version == "v3.3.4":
        from hi_em.sem_core_v334_rnn_var import HiEMSegmenterV334
        return HiEMSegmenterV334(dim=dim, alpha=alpha, lmda=lmda), "PE-var"
    if version == "v3.3.6":
        from hi_em.sem_core_v336 import HiEMSegmenterV336
        return HiEMSegmenterV336(dim=dim, alpha=alpha, lmda=lmda), "SEM2 replay"
    if version == "v3.3.8":
        from hi_em.sem_core_v338 import HiEMSegmenterV338
        return HiEMSegmenterV338(dim=dim, alpha=alpha, lmda=lmda), "SEM2 fresh f0"
    sys.exit(f"unsupported --version {version!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=str, default="gpt4_a1b77f9c")
    ap.add_argument("--idx", type=int, default=None,
                    help="s_cleaned 위치 (qid 우선)")
    ap.add_argument("--version", type=str, default="v3.1.1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lmda", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=50.0,
                    help="v3.1.1 cosine temperature (권장 50)")
    ap.add_argument("--cos-threshold", type=float, default=0.7,
                    help="v3.1.1 fresh-topic cosine baseline (권장 0.7)")
    args = ap.parse_args()

    entries = json.loads(DATA.read_text())
    idx, entry = None, None
    if args.idx is not None and args.qid == ap.get_default("qid"):
        idx, entry = args.idx, entries[args.idx]
    else:
        for i, e in enumerate(entries):
            if e.get("question_id") == args.qid:
                idx, entry = i, e
                break
    if entry is None:
        sys.exit(f"qid {args.qid!r} not found")

    # evidence 세션 id 집합 (세션레벨 라벨)
    ev_sids = set(entry.get("answer_session_ids") or [])
    sids = entry.get("haystack_session_ids") or []

    # flatten user 턴: (gidx, sess_no(1-based), has_answer, content)
    flat = []
    for si, sess in enumerate(entry["haystack_sessions"], start=1):
        for t in sess:
            if t.get("role") == "user":
                flat.append({
                    "sess": si,
                    "has_answer": bool(t.get("has_answer", False)),
                    "content": (t.get("content", "") or "").replace("\n", " "),
                })

    enc = QueryEncoder()
    embs = enc.encode([t["content"] for t in flat])
    seg, desc = _build_segmenter(
        args.version, embs.shape[1], args.alpha, args.lmda,
        args.tau, args.cos_threshold,
    )
    for k, t in enumerate(flat):
        tid, _ = seg.assign(embs[k])
        t["topic"] = int(tid)
        t["gidx"] = k + 1

    topics = sorted({t["topic"] for t in flat})
    ev_topics = sorted({t["topic"] for t in flat if t["has_answer"]})

    lines = []
    lines.append(
        f"{args.version}  alpha={args.alpha:g} lmda={args.lmda:g} "
        f"tau={args.tau:g} cos_thr={args.cos_threshold:g} (det) — "
        f"s idx{idx} qid={args.qid} topic별 그룹 ({desc})"
    )
    lines.append(
        f"sessions={len(entry['haystack_sessions'])} user턴={len(flat)} "
        f"has_answer={sum(t['has_answer'] for t in flat)} · "
        f"topic {len(topics)}개  evidence(★) topics={ev_topics}  "
        f"evidence 세션={[sids.index(s)+1 for s in ev_sids if s in sids]}"
    )
    lines.append("=" * 100)
    for tk in topics:
        members = [t for t in flat if t["topic"] == tk]
        msess = sorted({t["sess"] for t in members})
        ev = "  ★EVIDENCE" if any(t["has_answer"] for t in members) else ""
        lines.append(
            f"\n### TOPIC {tk} (user턴 {len(members)}, 세션 {msess}{ev})"
        )
        for t in members:
            star = "★" if t["has_answer"] else " "
            lines.append(f"  [#{t['gidx']} s{t['sess']}] {star} {t['content']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vtag = args.version.replace(".", "").replace("v", "v")
    out = OUT_DIR / f"longmemeval_s_idx{idx}_{vtag}_bytopic.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))
    print(f"... ({len(topics)} topics, {len(flat)} user turns)")
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
