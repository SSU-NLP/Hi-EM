#!/usr/bin/env python3
"""Phase 4: 4-way baseline 평가 (LongMemEval).

Methods (--method):
    sliding   : 직전 K turn (history 끝에서 자르기)
    full      : 전체 history 그대로 prefill (LLM 토큰 한계는 vLLM이 처리)
    rag       : 모든 turn 임베딩 → cosine top-K (chronological 정렬 후 prefill)
    hi-em     : HiEM.preload_history(history) + handle_turn(question)

Output:
    JSONL per line: {"question_id": ..., "hypothesis": ..., "method": ..., "model": ...}

Usage:
    uv run python scripts/run_longmemeval.py --method sliding --limit 30
    uv run python scripts/run_longmemeval.py --method hi-em
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Silence HF tokenizers fork warning before any encoder import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.llm import OpenAIChatLLM  # noqa: E402
from hi_em.orchestrator import HiEM  # noqa: E402


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def flatten_history(haystack_sessions: list[list[dict]]) -> list[dict]:
    """[[turn, turn], [turn, ...], ...] → [turn, turn, turn, ...]"""
    return [t for session in haystack_sessions for t in session]


# --- Baselines -----------------------------------------------------------

def run_sliding(
    history: list[dict], question: str, llm: OpenAIChatLLM, model: str,
    k_turns: int, **llm_kwargs,
) -> str:
    selected = history[-k_turns:] if k_turns > 0 else []
    msgs = [{"role": t["role"], "content": t["content"]} for t in selected]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs)


def run_full(
    history: list[dict], question: str, llm: OpenAIChatLLM, model: str,
    **llm_kwargs,
) -> str:
    msgs = [{"role": t["role"], "content": t["content"]} for t in history]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs)


def run_rag(
    history: list[dict], question: str, llm: OpenAIChatLLM, model: str,
    encoder: QueryEncoder, k: int, **llm_kwargs,
) -> str:
    if not history:
        msgs = [{"role": "user", "content": question}]
        return llm.chat(msgs, model=model, **llm_kwargs)
    contents = [t["content"] for t in history]
    embs = np.asarray(encoder.encode(contents))           # (N, D)
    q_emb = np.asarray(encoder.encode([question])[0])     # (D,)
    sims = embs @ q_emb
    top_idx = np.argsort(-sims)[: min(k, len(history))]
    chrono = sorted(int(i) for i in top_idx)
    msgs = [{"role": history[i]["role"], "content": history[i]["content"]} for i in chrono]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs)


def run_hi_em(
    history: list[dict], question: str, llm: OpenAIChatLLM, model: str,
    encoder: QueryEncoder, ltm_root: Path, conv_id: str,
    k_topics: int, k_turns_per_topic: int, **llm_kwargs,
) -> str:
    hi = HiEM(
        conv_id=conv_id, encoder=encoder, llm=llm, model=model,
        ltm_root=ltm_root,
        k_topics=k_topics, k_turns_per_topic=k_turns_per_topic,
        response_filter=strip_think_tags,
        **llm_kwargs,
    )
    hi.preload_history(history)
    return hi.handle_turn(question)


# --- Driver --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["sliding", "full", "rag", "hi-em"])
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "benchmarks/LongMemEval/data/longmemeval_oracle.json"),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N questions (sanity)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    # method-specific HP
    parser.add_argument("--sliding-k", type=int, default=20)
    parser.add_argument("--rag-k", type=int, default=10)
    parser.add_argument("--k-topics", type=int, default=3)
    parser.add_argument("--k-turns-per-topic", type=int, default=5)
    # Hi-EM persistence HP (옵션 5 결과: cluster 보존성 우위)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lmda", type=float, default=10.0)
    parser.add_argument("--sigma0-sq", type=float, default=0.01)
    # output
    parser.add_argument("--output", required=True, help="hypothesis jsonl")
    parser.add_argument(
        "--ltm-root", default=str(REPO_ROOT / "data/ltm/longmemeval"),
        help="(Wiped per question to keep LTMs isolated; only used when method=hi-em)",
    )
    args = parser.parse_args()

    load_dotenv()
    base_url = os.environ.get("OPENAI_BASE_URL", "(SDK default)")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[fatal] OPENAI_API_KEY missing — set in .env"); sys.exit(1)
    print(f"[env] base_url={base_url}  model={args.model}")
    print(f"[args] method={args.method} limit={args.limit} data={args.data}")

    print(f"[load] {args.data}")
    questions = json.loads(Path(args.data).read_text())
    if args.limit:
        questions = questions[: args.limit]
    print(f"  {len(questions)} questions")

    needs_encoder = args.method in {"rag", "hi-em"}
    encoder = None
    if needs_encoder:
        print("[encoder] loading bge-base-en-v1.5 ...")
        encoder = QueryEncoder()
        print(f"  device={encoder.device}")

    llm = OpenAIChatLLM()
    llm_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Hi-EM: per-question LTM root for isolation
    if args.method == "hi-em":
        ltm_base = Path(args.ltm_root)
        if ltm_base.exists():
            shutil.rmtree(ltm_base)
        ltm_base.mkdir(parents=True)

    t_start = time.perf_counter()
    with out_path.open("w") as f:
        for i, entry in enumerate(questions, 1):
            qid = entry["question_id"]
            question = entry["question"]
            history = flatten_history(entry["haystack_sessions"])

            t0 = time.perf_counter()
            try:
                if args.method == "sliding":
                    hyp = run_sliding(history, question, llm, args.model,
                                      k_turns=args.sliding_k, **llm_kwargs)
                elif args.method == "full":
                    hyp = run_full(history, question, llm, args.model, **llm_kwargs)
                elif args.method == "rag":
                    hyp = run_rag(history, question, llm, args.model,
                                  encoder=encoder, k=args.rag_k, **llm_kwargs)
                elif args.method == "hi-em":
                    conv_id = qid.replace("/", "_")
                    hyp = run_hi_em(
                        history, question, llm, args.model,
                        encoder=encoder,
                        ltm_root=Path(args.ltm_root) / conv_id,
                        conv_id=conv_id,
                        k_topics=args.k_topics,
                        k_turns_per_topic=args.k_turns_per_topic,
                        alpha=args.alpha, lmda=args.lmda, sigma0_sq=args.sigma0_sq,
                        **llm_kwargs,
                    )
                else:
                    raise ValueError(args.method)
                # caller-side cleanup of <think> for non-hi-em methods too
                hyp_clean = strip_think_tags(hyp)
            except Exception as e:
                print(f"  [{i}/{len(questions)}] {qid}: ERROR {e}")
                hyp_clean = ""

            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{len(questions)}] {qid} ({entry['question_type']}) "
                  f"hist={len(history)} turns / {elapsed:.1f}s")
            f.write(json.dumps({
                "question_id": qid,
                "hypothesis": hyp_clean,
                "method": args.method,
                "model": args.model,
            }) + "\n")
            f.flush()

    print(f"\ntotal {time.perf_counter() - t_start:.1f}s → {out_path}")


if __name__ == "__main__":
    main()
