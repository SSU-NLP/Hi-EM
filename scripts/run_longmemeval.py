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
import itertools
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Silence HF tokenizers fork warning before any encoder import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.eval_logging import (  # noqa: E402
    WandbRun, aggregate_summary, count_prefill_tokens,
)
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
) -> tuple[str, list[dict], dict]:
    selected = history[-k_turns:] if k_turns > 0 else []
    msgs = [{"role": t["role"], "content": t["content"]} for t in selected]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}


def run_full(
    history: list[dict], question: str, llm: OpenAIChatLLM, model: str,
    **llm_kwargs,
) -> tuple[str, list[dict], dict]:
    msgs = [{"role": t["role"], "content": t["content"]} for t in history]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}


def run_rag(
    history: list[dict], question: str, llm: OpenAIChatLLM, model: str,
    encoder: QueryEncoder, k: int, **llm_kwargs,
) -> tuple[str, list[dict], dict]:
    if not history:
        msgs = [{"role": "user", "content": question}]
        return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}
    contents = [t["content"] for t in history]
    embs = np.asarray(encoder.encode(contents))           # (N, D), thread-safe
    q_emb = np.asarray(encoder.encode([question])[0])     # (D,)
    sims = embs @ q_emb
    top_idx = np.argsort(-sims)[: min(k, len(history))]
    chrono = sorted(int(i) for i in top_idx)
    msgs = [{"role": history[i]["role"], "content": history[i]["content"]} for i in chrono]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}


def run_hi_em(
    history: list[dict], question: str, llm: OpenAIChatLLM, model: str,
    encoder: QueryEncoder, ltm_root: Path, conv_id: str,
    k_topics: int, k_turns_per_topic: int, **llm_kwargs,
) -> tuple[str, list[dict], dict]:
    hi = HiEM(
        conv_id=conv_id, encoder=encoder, llm=llm, model=model,
        ltm_root=ltm_root,
        k_topics=k_topics, k_turns_per_topic=k_turns_per_topic,
        response_filter=strip_think_tags,
        **llm_kwargs,
    )
    # encoder serializes its own forward pass (see hi_em.embedding); segmenter +
    # LTM are per-question (fresh HiEM instance), so no cross-thread state.
    hi.preload_history(history)
    response, debug = hi.handle_turn(question, return_debug=True)
    revisit_hit = int(any(t["topic_id"] == debug["topic_id"]
                          for t in debug["prefill_turns"]))
    return response, debug["messages"], {"topic_revisit_hit": revisit_hit}


# --- Driver --------------------------------------------------------------

def main() -> None:
    # .env first so HIEM_* defaults below pick up overrides.
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["sliding", "full", "rag", "hi-em"])
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "benchmarks/LongMemEval/data/longmemeval_oracle.json"),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N questions (sanity). "
                             "With --stratify, samples N/n_qtypes per type.")
    parser.add_argument("--stratify", action="store_true",
                        help="Sample evenly across question_type. "
                             "LongMemEval oracle is type-sorted, so plain "
                             "--limit yields a single type — almost always "
                             "what you want for sanity.")
    parser.add_argument("--model",
                        default=os.environ.get("HIEM_MODEL", "Qwen/Qwen3-8B"),
                        help="Default: $HIEM_MODEL (.env) or Qwen/Qwen3-8B.")
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.environ.get("HIEM_RUN_MAX_TOKENS", "800")),
                        help="Default: $HIEM_RUN_MAX_TOKENS (.env) or 800. "
                             "Reasoning models (Qwen3) need room for <think> "
                             "+ final answer.")
    parser.add_argument("--temperature", type=float,
                        default=float(os.environ.get("HIEM_TEMPERATURE", "0.7")))
    parser.add_argument("--device",
                        default=os.environ.get("HIEM_DEVICE"),
                        help="Encoder device: cuda / mps / cpu. "
                             "None = auto (cuda → mps → cpu). "
                             "Default: $HIEM_DEVICE (.env) or auto.")
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
    # concurrency
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Concurrent question workers (LLM I/O bound). 1=sequential. "
             "Tune to vLLM endpoint capacity (default safe 8 for sanity).",
    )
    # wandb (optional — skipped if WANDB_API_KEY missing)
    parser.add_argument("--wandb-project", default="hi-em-phase4")
    parser.add_argument("--wandb-group", default=None,
                        help="Default: dataset stem + UTC timestamp")
    parser.add_argument("--no-token-count", action="store_true",
                        help="Skip prefill_tokens computation (faster, only if "
                             "tokenizer download is undesirable).")
    args = parser.parse_args()

    base_url = os.environ.get("OPENAI_BASE_URL", "(SDK default)")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[fatal] OPENAI_API_KEY missing — set in .env"); sys.exit(1)
    print(f"[env] base_url={base_url}  model={args.model}")
    print(f"[args] method={args.method} limit={args.limit} data={args.data}")

    print(f"[load] {args.data}")
    questions = json.loads(Path(args.data).read_text())
    if args.stratify:
        from collections import defaultdict as _dd
        by_type: dict[str, list] = _dd(list)
        for q in questions:
            by_type[q["question_type"]].append(q)
        if args.limit:
            k_per = max(1, args.limit // len(by_type))
            questions = [q for qs in by_type.values() for q in qs[:k_per]]
        else:
            questions = [q for qs in by_type.values() for q in qs]
        print(f"  stratified: {len(by_type)} types × ~{len(questions)//len(by_type)}/type = {len(questions)}")
    elif args.limit:
        questions = questions[: args.limit]
    print(f"  {len(questions)} questions")

    needs_encoder = args.method in {"rag", "hi-em"}
    encoder = None
    if needs_encoder:
        print("[encoder] loading bge-base-en-v1.5 ...")
        encoder = QueryEncoder(device=args.device)
        print(f"  device={encoder.device}")

    # Warm up the tokenizer in the main thread before the ThreadPool starts:
    # transformers' lazy module initialization is not thread-safe and races
    # on the first concurrent call (Python 3.13 + transformers 4.49 surfaces
    # this as 'cannot import name AutoTokenizer').
    if not args.no_token_count:
        print("[tokenizer] warmup ...")
        try:
            count_prefill_tokens([{"role": "user", "content": "warmup"}], args.model)
        except Exception as e:
            print(f"  warmup failed ({e}); disabling token count")
            args.no_token_count = True

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

    def process(entry: dict) -> dict:
        """Process one question → per-q record (thread-safe, no shared state)."""
        qid = entry["question_id"]
        question = entry["question"]
        history = flatten_history(entry["haystack_sessions"])
        t0 = time.perf_counter()
        try:
            if args.method == "sliding":
                hyp, msgs, extras = run_sliding(
                    history, question, llm, args.model,
                    k_turns=args.sliding_k, **llm_kwargs)
            elif args.method == "full":
                hyp, msgs, extras = run_full(
                    history, question, llm, args.model, **llm_kwargs)
            elif args.method == "rag":
                hyp, msgs, extras = run_rag(
                    history, question, llm, args.model,
                    encoder=encoder, k=args.rag_k, **llm_kwargs)
            elif args.method == "hi-em":
                conv_id = qid.replace("/", "_")
                hyp, msgs, extras = run_hi_em(
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
            hyp_clean = strip_think_tags(hyp)
            elapsed = time.perf_counter() - t0
            tokens = (count_prefill_tokens(msgs, args.model)
                      if not args.no_token_count else None)
            rec: dict[str, object] = {
                "question_id": qid,
                "hypothesis": hyp_clean,
                "question_type": entry["question_type"],
                "history_n_turns": len(history),
                "prefill_n_msgs": len(msgs),
                "latency_sec": elapsed,
                "error": None,
                **extras,
            }
            if tokens is not None:
                rec["prefill_tokens"] = tokens
            return rec
        except Exception as e:
            return {
                "question_id": qid,
                "hypothesis": "",
                "question_type": entry["question_type"],
                "history_n_turns": len(history),
                "latency_sec": time.perf_counter() - t0,
                "error": str(e),
            }

    # wandb (optional, no-op if WANDB_API_KEY missing)
    from datetime import datetime, timezone
    group = args.wandb_group or (
        f"{Path(args.data).stem}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    config = {
        "method": args.method, "model": args.model, "dataset": args.data,
        "limit": args.limit, "workers": args.workers,
        "sliding_k": args.sliding_k, "rag_k": args.rag_k,
        "k_topics": args.k_topics, "k_turns_per_topic": args.k_turns_per_topic,
        "alpha": args.alpha, "lmda": args.lmda, "sigma0_sq": args.sigma0_sq,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
    }
    sidecar = Path(str(out_path) + ".wandb-run-id")
    wb = WandbRun(project=args.wandb_project, name=args.method,
                  group=group, config=config, sidecar_path=sidecar)
    if wb.enabled:
        print(f"[wandb] project={args.wandb_project} group={group} name={args.method}")

    t_start = time.perf_counter()
    write_lock = threading.Lock()
    counter = itertools.count(1)
    total = len(questions)
    per_q_records: list[dict] = []

    def write_result(rec: dict) -> None:
        with write_lock:
            i = next(counter)
            tag = ("ERROR " + rec["error"]) if rec.get("error") else "ok"
            tok = f" tok={rec['prefill_tokens']}" if "prefill_tokens" in rec else ""
            print(f"  [{i}/{total}] {rec['question_id']} ({rec['question_type']}) "
                  f"hist={rec['history_n_turns']} msgs={rec.get('prefill_n_msgs', '?')}{tok} "
                  f"/ {rec['latency_sec']:.1f}s  {tag}")
            f.write(json.dumps({
                "question_id": rec["question_id"],
                "hypothesis": rec["hypothesis"],
                "method": args.method,
                "model": args.model,
            }) + "\n")
            f.flush()
            per_q_records.append(rec)
            log_payload = {k: v for k, v in rec.items()
                           if k in {"prefill_n_msgs", "prefill_tokens",
                                    "latency_sec", "history_n_turns",
                                    "topic_revisit_hit"}}
            wb.log(log_payload, step=i)

    with out_path.open("w") as f:
        if args.workers <= 1:
            for entry in questions:
                write_result(process(entry))
        else:
            print(f"[concurrency] workers={args.workers}")
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(process, e) for e in questions]
                for fut in as_completed(futures):
                    write_result(fut.result())

    runtime = time.perf_counter() - t_start
    print(f"\ntotal {runtime:.1f}s → {out_path}")
    summary = aggregate_summary(per_q_records)
    summary["total_runtime_sec"] = runtime
    wb.summary(**summary)
    if wb.enabled:
        print(f"[wandb] summary: {summary}")
    wb.finish()


if __name__ == "__main__":
    main()
