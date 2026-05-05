#!/usr/bin/env python3
"""Phase 4-Re: research-experiment-infrastructure 적용 단일 entry.

기존 ``run_longmemeval.py`` + ``judge_longmemeval.py`` 의 두 단계를 ``round`` 단위로
묶어 atomic checkpoint + resume 가능하게 한다. 호출 패턴:

    # 단일 experiment
    uv run python scripts/run_experiment.py \\
        --method hi-em \\
        --data benchmarks/LongMemEval/data/longmemeval_oracle.json \\
        --questions-per-round 50

    # session 안에서
    uv run python scripts/run_experiment.py \\
        --method hi-em \\
        --session 20260427_phase4_main \\
        --questions-per-round 50

자동 동작:
    - 첫 실행: ``results/experiments/{exp_id}/`` 생성 + experiment.json 작성
    - 중단 후 재실행: 같은 ``--exp-id`` 또는 자동 생성 id 재사용 시 마지막 완료 라운드부터 resume
    - 라운드 끝마다 summary.json (informational) → checkpoint.json (commit) atomic
    - sanity_check_summary 결과 stdout 경고
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Silence HF tokenizers fork warning before encoder import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.atomic_io import append_jsonl, load_json, load_jsonl, save_json  # noqa: E402
from hi_em.config import load_config  # noqa: E402
from hi_em.embedding import QueryEncoder, make_encoder  # noqa: E402
from hi_em.eval_logging import (  # noqa: E402
    WandbRun, aggregate_summary, count_prefill_tokens, parse_judge_yes_no,
)
from hi_em.experiment import (  # noqa: E402
    DEFAULT_RESULTS_ROOT, SUMMARY_JSON_SCHEMA_VERSION, ExperimentMeta,
    create_experiment, experiment_dir, find_resumable_experiment,
    make_experiment_id, mark_experiment_complete, mark_round_complete,
    round_dir, sanity_check_summary, session_dir,
)
from hi_em.llm import OpenAIChatLLM  # noqa: E402
from hi_em.locomo_judge import score_one as locomo_score_one  # noqa: E402
from hi_em.locomo_loader import build_entries as build_locomo_entries  # noqa: E402
from hi_em.orchestrator import HiEM  # noqa: E402

# LongMemEval prompt template은 judge_longmemeval에 있음 — duplication 회피 위해 import.
JUDGE_SCRIPT = REPO_ROOT / "scripts" / "judge_longmemeval.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from judge_longmemeval import get_prompt  # noqa: E402


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def flatten_history(haystack_sessions: list[list[dict]]) -> list[dict]:
    return [t for session in haystack_sessions for t in session]


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# --- Method functions (4 baseline) — same logic as run_longmemeval.py ----

def run_sliding(history, question, llm, model, k_turns, **llm_kwargs):
    selected = history[-k_turns:] if k_turns > 0 else []
    msgs = [{"role": t["role"], "content": t["content"]} for t in selected]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}


def run_full(history, question, llm, model, **llm_kwargs):
    msgs = [{"role": t["role"], "content": t["content"]} for t in history]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}


class EncoderCache:
    """Per-(sample_id, kind) encoder embedding cache for RAG-family methods.

    LoCoMo poses ~200 questions per conversation, all sharing the same
    600-turn history (or the same per-session ``session_summaries`` /
    per-(session, speaker) ``observations``). Without caching, ``run_rag``
    re-encodes the full 600-turn history for every question — and the
    encoder is lock-serialized (``QueryEncoder.encode`` holds a global
    ``threading.Lock``), so the cost grows linearly with question count
    even with 8 worker threads.

    With this cache we encode each conversation's corpus exactly once
    per ``(sample_id, kind)`` pair and reuse the matrix for every
    subsequent query. The question embedding is still computed fresh on
    every call (it's per-Q by definition).

    Same locking pattern as :class:`HiEMConvCache`: per-key build lock
    inside an outer ``_dict_lock`` so concurrent workers for the same
    sample don't trigger duplicate encodes.
    """

    def __init__(self, encoder) -> None:
        self._encoder = encoder
        # key = (sample_id, kind) → embedding matrix (n_chunks, dim)
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._build_locks: dict[tuple[str, str], threading.Lock] = {}
        self._dict_lock = threading.Lock()

    def _get(self, sample_id: str, kind: str, texts: list[str]) -> np.ndarray:
        key = (sample_id, kind)
        with self._dict_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            lk = self._build_locks.setdefault(key, threading.Lock())
        with lk:
            with self._dict_lock:
                cached = self._cache.get(key)
                if cached is not None:
                    return cached
            embs = np.asarray(self._encoder.encode(texts))
            with self._dict_lock:
                self._cache[key] = embs
            return embs

    def history(self, sample_id: str, history: list[dict]) -> np.ndarray:
        return self._get(sample_id, "history", [t["content"] for t in history])

    def summaries(self, sample_id: str, summaries: list[dict]) -> np.ndarray:
        return self._get(sample_id, "summary", [s["text"] for s in summaries])

    def observations(self, sample_id: str, obs: list[dict]) -> np.ndarray:
        return self._get(sample_id, "observation", [o["text"] for o in obs])


def run_rag(history, question, llm, model, encoder, k,
            sample_id: str | None = None,
            enc_cache: "EncoderCache | None" = None,
            **llm_kwargs):
    if not history:
        msgs = [{"role": "user", "content": question}]
        return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}
    if enc_cache is not None and sample_id is not None:
        embs = enc_cache.history(sample_id, history)
    else:
        embs = np.asarray(encoder.encode([t["content"] for t in history]))
    q_emb = np.asarray(encoder.encode([question])[0])
    sims = embs @ q_emb
    top_idx = np.argsort(-sims)[: min(k, len(history))]
    chrono = sorted(int(i) for i in top_idx)
    msgs = [{"role": history[i]["role"], "content": history[i]["content"]} for i in chrono]
    msgs.append({"role": "user", "content": question})
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}


def run_rag_summary(extra_db, question, llm, model, encoder, k,
                    sample_id: str | None = None,
                    enc_cache: "EncoderCache | None" = None,
                    **llm_kwargs):
    """LoCoMo-only: retrieve over per-session summaries (data-provided).

    Mirrors the LoCoMo paper's summary-RAG baseline. Each session has one
    summary; we retrieve the top-K most relevant ones, prepend chronological
    date markers, and prefill them as a single ``user`` context blob followed
    by the question.
    """
    summaries = (extra_db or {}).get("session_summaries") or []
    if not summaries:
        msgs = [{"role": "user", "content": question}]
        return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}
    if enc_cache is not None and sample_id is not None:
        embs = enc_cache.summaries(sample_id, summaries)
    else:
        embs = np.asarray(encoder.encode([s["text"] for s in summaries]))
    q_emb = np.asarray(encoder.encode([question])[0])
    sims = embs @ q_emb
    top_idx = np.argsort(-sims)[: min(k, len(summaries))]
    chrono = sorted(int(i) for i in top_idx)
    blocks = []
    for i in chrono:
        s = summaries[i]
        blocks.append(f"[{s.get('date', '')}] Session {s['session_idx']} summary: {s['text']}")
    context = "\n\n".join(blocks)
    msgs = [
        {"role": "user", "content": context},
        {"role": "user", "content": question},
    ]
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {"k_retrieved": len(chrono)}


def run_rag_observation(extra_db, question, llm, model, encoder, k,
                        sample_id: str | None = None,
                        enc_cache: "EncoderCache | None" = None,
                        **llm_kwargs):
    """LoCoMo-only: retrieve over per-(session, speaker) observations.

    Observations are atomic facts extracted by the LoCoMo authors via
    GPT-3.5 — finer-grained than session summaries. We retrieve the top-K
    observation strings (no aggregation by speaker / session) and prefill
    them with their date + speaker prefix.
    """
    obs = (extra_db or {}).get("observations") or []
    if not obs:
        msgs = [{"role": "user", "content": question}]
        return llm.chat(msgs, model=model, **llm_kwargs), msgs, {}
    if enc_cache is not None and sample_id is not None:
        embs = enc_cache.observations(sample_id, obs)
    else:
        embs = np.asarray(encoder.encode([o["text"] for o in obs]))
    q_emb = np.asarray(encoder.encode([question])[0])
    sims = embs @ q_emb
    top_idx = np.argsort(-sims)[: min(k, len(obs))]
    chrono = sorted(int(i) for i in top_idx)
    blocks = []
    for i in chrono:
        o = obs[i]
        blocks.append(f"[{o.get('date', '')}] {o.get('speaker', '?')}: {o['text']}")
    context = "\n".join(blocks)
    msgs = [
        {"role": "user", "content": context},
        {"role": "user", "content": question},
    ]
    return llm.chat(msgs, model=model, **llm_kwargs), msgs, {"k_retrieved": len(chrono)}


class HiEMConvCache:
    """Per-conversation HiEM instance cache for LoCoMo-style benchmarks.

    LoCoMo asks ~200 questions against the **same** ~600-turn conversation
    history. The LongMemEval-shaped pipeline rebuilds the entire HiEM
    state (LTM jsonl + segmenter centroids + STM round-promote) for every
    question — a 200× preload cost. This cache builds the post-conversation
    state exactly once per ``sample_id`` and serves it to every test
    question via :meth:`HiEM.eval_query` (read-only, no state mutation).

    Thread-safe lazy build: the first worker that requests a sample takes
    its per-sample lock and builds; concurrent requests for the same sample
    block until the build completes, then share the cached instance. Other
    samples build in parallel.
    """

    def __init__(self, ltm_root: Path, encoder, llm, model: str,
                 llm_kwargs: dict, args) -> None:
        self._ltm_root = Path(ltm_root)
        self._encoder = encoder
        self._llm = llm
        self._model = model
        self._llm_kwargs = llm_kwargs
        self._args = args
        self._instances: dict[tuple[str, str], HiEM] = {}
        self._build_locks: dict[tuple[str, str], threading.Lock] = {}
        self._dict_lock = threading.Lock()

    def _build(self, sample_id: str, method: str,
               haystack_sessions: list[list[dict]]) -> HiEM:
        conv_dir = self._ltm_root / f"{sample_id}_{method}"
        if conv_dir.exists():
            shutil.rmtree(conv_dir)
        a = self._args
        common = dict(
            conv_id=sample_id, encoder=self._encoder, llm=self._llm,
            model=self._model, ltm_root=conv_dir,
            alpha=a.alpha, lmda=a.lmda, sigma0_sq=a.sigma0_sq,
            response_filter=strip_think_tags,
            **self._llm_kwargs,
        )
        if method == "hi-em":
            hi = HiEM(
                k_topics=a.k_topics,
                k_turns_per_topic=a.k_turns_per_topic,
                **common,
            )
            hi.preload_history(flatten_history(haystack_sessions))
        elif method in ("hi-em-full-v1", "hi-em-full-v2",
                        "hi-em-full-v3.1.1", "hi-em-full-v3.2.1"):
            if method == "hi-em-full-v3.1.1":
                v3_extra = {
                    "version": "v3.1.1",
                    "tau": a.tau,
                    "cos_threshold": a.cos_threshold,
                }
            elif method == "hi-em-full-v3.2.1":
                v3_extra = {
                    "version": "v3.2.1",
                    "tau": a.tau,
                    "cos_threshold": a.cos_threshold,
                    "beta": a.beta,
                }
            else:
                v3_extra = {}
            hi = HiEM(
                use_stm=True,
                round_size=a.round_size,
                stm_max_topics=a.stm_max_topics,
                stm_max_turns=a.stm_max_turns,
                promotion_threshold=a.promotion_threshold,
                importance_alpha=tuple(a.importance_alpha),
                lambda_r=a.lambda_r,
                lambda_freq=a.lambda_freq,
                min_floor=a.min_floor,
                round_async=False,
                round_clear_stm=(method == "hi-em-full-v2"),
                **v3_extra,
                **common,
            )
            if method == "hi-em-full-v2":
                hi.preload_history_sessioned(haystack_sessions, flush_stm_each=True)
            else:
                hi.preload_history(flatten_history(haystack_sessions))
        else:
            raise ValueError(method)
        return hi

    def get(self, sample_id: str, method: str,
            haystack_sessions: list[list[dict]]) -> HiEM:
        key = (sample_id, method)
        with self._dict_lock:
            cached = self._instances.get(key)
            if cached is not None:
                return cached
            lk = self._build_locks.setdefault(key, threading.Lock())
        with lk:
            with self._dict_lock:
                cached = self._instances.get(key)
                if cached is not None:
                    return cached
            hi = self._build(sample_id, method, haystack_sessions)
            with self._dict_lock:
                self._instances[key] = hi
            return hi


def run_hi_em(history, question, llm, model, encoder, ltm_root, conv_id,
              k_topics, k_turns_per_topic, **llm_kwargs):
    # Resume safety: a previous attempt at this round may have written a partial
    # LTM jsonl for this conv_id. preload_history is append-only, so calling it
    # again would double-write the history and corrupt prefill / topic state.
    # Wipe the per-question dir before reconstructing — the source of truth is
    # the question's haystack_sessions, which we replay deterministically.
    if ltm_root.exists():
        shutil.rmtree(ltm_root)
    hi = HiEM(
        conv_id=conv_id, encoder=encoder, llm=llm, model=model,
        ltm_root=ltm_root,
        k_topics=k_topics, k_turns_per_topic=k_turns_per_topic,
        response_filter=strip_think_tags,
        **llm_kwargs,
    )
    hi.preload_history(history)
    response, debug = hi.handle_turn(question, return_debug=True)
    revisit_hit = int(any(t["topic_id"] == debug["topic_id"]
                          for t in debug["prefill_turns"]))
    return response, debug["messages"], {"topic_revisit_hit": revisit_hit}


def run_hi_em_full(history, question, llm, model, encoder, ltm_root, conv_id,
                   *, alpha, lmda, sigma0_sq, round_size,
                   stm_max_topics, stm_max_turns, promotion_threshold,
                   importance_alpha, lambda_r, lambda_freq, min_floor,
                   version="v2", tau=50.0, cos_threshold=0.7, beta=0.5,
                   **llm_kwargs):
    """Phase 2-Full: STM-stateful HiEM (v2 Gaussian, v3.1 cosine MAP, or v3.2 cosine MAP + sub-linear count)."""
    if ltm_root.exists():
        shutil.rmtree(ltm_root)
    hi = HiEM(
        conv_id=conv_id, encoder=encoder, llm=llm, model=model,
        ltm_root=ltm_root,
        alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq,
        response_filter=strip_think_tags,
        use_stm=True,
        round_size=round_size,
        stm_max_topics=stm_max_topics,
        stm_max_turns=stm_max_turns,
        promotion_threshold=promotion_threshold,
        importance_alpha=tuple(importance_alpha),
        lambda_r=lambda_r,
        lambda_freq=lambda_freq,
        min_floor=min_floor,
        round_async=False,
        version=version,
        tau=tau,
        cos_threshold=cos_threshold,
        beta=beta,
        **llm_kwargs,
    )
    hi.preload_history(history)
    response, debug = hi.handle_turn(question, return_debug=True)
    revisit_hit = int(any(t["topic_id"] == debug["topic_id"]
                          for t in debug["prefill_turns"]))
    extras = {
        "topic_revisit_hit": revisit_hit,
        "stm_hit": int(bool(debug.get("stm_hit"))),
        "stm_topics": len(hi.stm.current_topics()) if hi.stm else 0,
        "stm_turns": hi.stm.total_turns() if hi.stm else 0,
    }
    return response, debug["messages"], extras


def run_hi_em_full_v2(haystack_sessions, question, llm, model, encoder,
                      ltm_root, conv_id, *, alpha, lmda, sigma0_sq, round_size,
                      stm_max_topics, stm_max_turns, promotion_threshold,
                      importance_alpha, lambda_r, lambda_freq, min_floor,
                      **llm_kwargs):
    """Phase 2-Full v2: STM cleared at **every round end** (clear → promote
    high-importance only) AND at **every haystack-session boundary**. LTM
    persists across sessions. Spec mapping: step 3 (라운드마다 STM = high-
    importance only) + step 4 (세션 끝날 때마다 STM 초기화)."""
    if ltm_root.exists():
        shutil.rmtree(ltm_root)
    hi = HiEM(
        conv_id=conv_id, encoder=encoder, llm=llm, model=model,
        ltm_root=ltm_root,
        alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq,
        response_filter=strip_think_tags,
        use_stm=True,
        round_size=round_size,
        stm_max_topics=stm_max_topics,
        stm_max_turns=stm_max_turns,
        promotion_threshold=promotion_threshold,
        importance_alpha=tuple(importance_alpha),
        lambda_r=lambda_r,
        lambda_freq=lambda_freq,
        min_floor=min_floor,
        round_async=False,
        round_clear_stm=True,
        **llm_kwargs,
    )
    hi.preload_history_sessioned(haystack_sessions, flush_stm_each=True)
    response, debug = hi.handle_turn(question, return_debug=True)
    revisit_hit = int(any(t["topic_id"] == debug["topic_id"]
                          for t in debug["prefill_turns"]))
    extras = {
        "topic_revisit_hit": revisit_hit,
        "stm_hit": int(bool(debug.get("stm_hit"))),
        "stm_topics": len(hi.stm.current_topics()) if hi.stm else 0,
        "stm_turns": hi.stm.total_turns() if hi.stm else 0,
        "n_sessions": len(haystack_sessions),
    }
    return response, debug["messages"], extras


# --- Round phase implementations ----------------------------------------

def phase_run(
    questions: list[dict], rdir: Path, args, encoder, llm, llm_kwargs,
    ltm_root: Path,
    hiem_cache: "HiEMConvCache | None" = None,
    enc_cache: "EncoderCache | None" = None,
) -> list[dict]:
    """Phase 1: hypothesis 생성. Returns per-question records.

    ``hiem_cache`` is non-None only when running in LoCoMo (or other
    multi-question-per-conversation) mode. In that case Hi-EM-family
    methods build the post-conversation memory state once per
    ``sample_id`` and answer each question via :meth:`HiEM.eval_query`
    (read-only — no state mutation, no preload re-run).

    ``enc_cache`` likewise enables per-conversation embedding reuse for
    ``rag`` / ``rag-summary`` / ``rag-observation``: each conversation's
    history (or summary / observation) chunks are encoded exactly once
    and the resulting matrix is reused for all of that conversation's
    questions. Without it, the encoder lock serializes ~200 redundant
    re-encodes of the same history per LoCoMo conversation.
    """
    hyp_path = rdir / "hypothesis.jsonl"
    # Idempotent: if file exists from interrupted previous attempt, restart phase fresh.
    if hyp_path.exists():
        hyp_path.unlink()

    write_lock = threading.Lock()
    counter = itertools.count(1)
    total = len(questions)
    records: list[dict] = []

    def _hi_eval(entry: dict, method: str) -> tuple[str, list[dict], dict]:
        """LoCoMo cached path: build conv state once, eval_query per Q."""
        hi = hiem_cache.get(entry["sample_id"], method, entry["haystack_sessions"])
        response, debug = hi.eval_query(entry["question"], return_debug=True)
        revisit_hit = int(any(t["topic_id"] == debug["topic_id"]
                              for t in debug["prefill_turns"]))
        extras: dict = {"topic_revisit_hit": revisit_hit}
        if "stm_hit" in debug:
            extras["stm_hit"] = int(bool(debug["stm_hit"]))
            extras["stm_topics"] = (
                len(hi.stm.current_topics()) if hi.stm else 0
            )
            extras["stm_turns"] = hi.stm.total_turns() if hi.stm else 0
        return response, debug["messages"], extras

    def process(entry: dict) -> dict:
        qid = entry["question_id"]
        question = entry["question"]
        haystack_sessions = entry["haystack_sessions"]
        history = flatten_history(haystack_sessions)
        t0 = time.perf_counter()
        try:
            if args.method == "sliding":
                hyp, msgs, extras = run_sliding(history, question, llm, args.model,
                                                k_turns=args.sliding_k, **llm_kwargs)
            elif args.method == "full":
                hyp, msgs, extras = run_full(history, question, llm, args.model, **llm_kwargs)
            elif args.method == "rag":
                hyp, msgs, extras = run_rag(
                    history, question, llm, args.model,
                    encoder=encoder, k=args.rag_k,
                    sample_id=entry.get("sample_id"), enc_cache=enc_cache,
                    **llm_kwargs)
            elif args.method == "rag-summary":
                hyp, msgs, extras = run_rag_summary(
                    entry.get("extra_databases"), question, llm, args.model,
                    encoder=encoder, k=args.rag_k,
                    sample_id=entry.get("sample_id"), enc_cache=enc_cache,
                    **llm_kwargs)
            elif args.method == "rag-observation":
                hyp, msgs, extras = run_rag_observation(
                    entry.get("extra_databases"), question, llm, args.model,
                    encoder=encoder, k=args.rag_k,
                    sample_id=entry.get("sample_id"), enc_cache=enc_cache,
                    **llm_kwargs)
            elif args.method in ("hi-em", "hi-em-full-v1", "hi-em-full-v2",
                                 "hi-em-full-v3.1.1", "hi-em-full-v3.2.1"):
                if hiem_cache is not None:
                    # LoCoMo path: build state once per sample_id,
                    # answer each question read-only via eval_query.
                    hyp, msgs, extras = _hi_eval(entry, args.method)
                else:
                    conv_id = qid.replace("/", "_")
                    if args.method == "hi-em":
                        hyp, msgs, extras = run_hi_em(
                            history, question, llm, args.model,
                            encoder=encoder,
                            ltm_root=ltm_root / conv_id,
                            conv_id=conv_id,
                            k_topics=args.k_topics,
                            k_turns_per_topic=args.k_turns_per_topic,
                            alpha=args.alpha, lmda=args.lmda, sigma0_sq=args.sigma0_sq,
                            **llm_kwargs,
                        )
                    elif args.method in ("hi-em-full-v1", "hi-em-full-v3.1.1", "hi-em-full-v3.2.1"):
                        if args.method == "hi-em-full-v3.1.1":
                            seg_version = "v3.1.1"
                        elif args.method == "hi-em-full-v3.2.1":
                            seg_version = "v3.2.1"
                        else:
                            seg_version = "v2"
                        hyp, msgs, extras = run_hi_em_full(
                            history, question, llm, args.model,
                            encoder=encoder,
                            ltm_root=ltm_root / conv_id,
                            conv_id=conv_id,
                            alpha=args.alpha, lmda=args.lmda, sigma0_sq=args.sigma0_sq,
                            round_size=args.round_size,
                            stm_max_topics=args.stm_max_topics,
                            stm_max_turns=args.stm_max_turns,
                            promotion_threshold=args.promotion_threshold,
                            importance_alpha=args.importance_alpha,
                            lambda_r=args.lambda_r,
                            lambda_freq=args.lambda_freq,
                            min_floor=args.min_floor,
                            version=seg_version,
                            tau=args.tau,
                            cos_threshold=args.cos_threshold,
                            beta=args.beta,
                            **llm_kwargs,
                        )
                    else:  # hi-em-full-v2
                        hyp, msgs, extras = run_hi_em_full_v2(
                            haystack_sessions, question, llm, args.model,
                            encoder=encoder,
                            ltm_root=ltm_root / conv_id,
                            conv_id=conv_id,
                            alpha=args.alpha, lmda=args.lmda, sigma0_sq=args.sigma0_sq,
                            round_size=args.round_size,
                            stm_max_topics=args.stm_max_topics,
                            stm_max_turns=args.stm_max_turns,
                            promotion_threshold=args.promotion_threshold,
                            importance_alpha=args.importance_alpha,
                            lambda_r=args.lambda_r,
                            lambda_freq=args.lambda_freq,
                            min_floor=args.min_floor,
                            **llm_kwargs,
                        )
            else:
                raise ValueError(args.method)
            hyp_clean = strip_think_tags(hyp)
            elapsed = time.perf_counter() - t0
            tokens = (count_prefill_tokens(msgs, args.model)
                      if not args.no_token_count else None)
            # Per-thread streaming metrics from the LLM call inside run_xxx
            # (ttft_sec / tpot_sec / output_tokens / gen_sec). Same thread
            # → thread-local last_call_metrics is the call we just made.
            # Tests / alt LLM classes may not implement the property — fall
            # back to empty dict.
            llm_metrics = getattr(llm, "last_call_metrics", None) or {}
            rec = {
                "question_id": qid,
                "hypothesis": hyp_clean,
                "question_type": entry["question_type"],
                "history_n_turns": len(history),
                "prefill_n_msgs": len(msgs),
                "latency_sec": elapsed,
                "error": None,
                **extras,
                **llm_metrics,
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

    def write_result(rec: dict) -> None:
        with write_lock:
            i = next(counter)
            tag = ("ERROR " + rec["error"]) if rec.get("error") else "ok"
            tok = f" tok={rec.get('prefill_tokens', '?')}" if "prefill_tokens" in rec else ""
            print(f"  run [{i}/{total}] {rec['question_id']} ({rec['question_type']}) "
                  f"hist={rec['history_n_turns']}{tok} / {rec['latency_sec']:.1f}s  {tag}")
            append_jsonl(hyp_path, {
                "question_id": rec["question_id"],
                "hypothesis": rec["hypothesis"],
                "method": args.method,
                "model": args.model,
            })
            records.append(rec)

    if args.workers <= 1:
        for entry in questions:
            write_result(process(entry))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process, e) for e in questions]
            for fut in as_completed(futures):
                write_result(fut.result())

    return records


def phase_judge_locomo(
    run_records: list[dict], questions: list[dict], rdir: Path, args,
) -> list[dict]:
    """LoCoMo F1 judge — pure local (no LLM call). Mirrors the official
    benchmarks/locomo/task_eval/evaluation.py metric per category. cat-5
    multi-choice answers are decoded via the entry's ``answer_key`` and
    abstention-checked.
    """
    judged_path = rdir / "judged.jsonl"
    if judged_path.exists():
        judged_path.unlink()

    qid2ref = {q["question_id"]: q for q in questions}
    judged_records: list[dict] = []
    for i, rec in enumerate(run_records, 1):
        qid = rec["question_id"]
        ref = qid2ref.get(qid, {})
        cat = int(ref.get("category", 0))
        scored = locomo_score_one(
            category=cat,
            prediction=rec.get("hypothesis", ""),
            answer=ref.get("answer", ""),
            answer_key=ref.get("answer_key"),
        )
        f1 = scored["f1"]
        out_rec = {
            **rec,
            "category": cat,
            "f1": f1,
            "accuracy": f1,  # so aggregate_summary picks it up
            "abstention": cat == 5,
            "abstention_correct": scored.get("abstention_correct"),
            "resolved_prediction": scored.get("resolved"),
        }
        mark = "✓" if f1 >= 0.5 else ("·" if f1 > 0 else "✗")
        print(f"  judge [{i}/{len(run_records)}] {qid} ({rec.get('question_type')}): "
              f"{mark} f1={f1:.2f}")
        append_jsonl(judged_path, out_rec)
        judged_records.append(out_rec)
    return judged_records


def phase_judge(
    run_records: list[dict], questions: list[dict], rdir: Path,
    args, llm, llm_kwargs,
) -> list[dict]:
    """Phase 2: judge each hypothesis using LongMemEval prompt template."""
    judged_path = rdir / "judged.jsonl"
    if judged_path.exists():
        judged_path.unlink()

    qid2ref = {q["question_id"]: q for q in questions}
    write_lock = threading.Lock()
    counter = itertools.count(1)
    total = len(run_records)
    judged_records: list[dict] = []

    def judge_one(rec: dict) -> dict:
        qid = rec["question_id"]
        ref = qid2ref.get(qid)
        if ref is None:
            return {**rec, "label": False, "skipped": True}
        is_abs = "_abs" in qid
        prompt = get_prompt(ref["question_type"], ref["question"], ref["answer"],
                            rec["hypothesis"], is_abs)
        chat_kwargs = {
            "model": args.judge_model,
            "temperature": 0.0,
            "max_tokens": args.judge_max_tokens,
        }
        if args.no_thinking:
            # vLLM honors chat_template_kwargs.enable_thinking; Crts honors
            # OpenRouter-style reasoning.enabled. Send both — each backend
            # ignores the other's key.
            chat_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning": {"enabled": False},
            }
        raw = llm.chat([{"role": "user", "content": prompt}], **chat_kwargs)
        label = parse_judge_yes_no(raw)
        return {
            **rec,
            "label": int(label),
            "accuracy": int(label),
            "abstention": is_abs,
            "judge_raw": raw,
        }

    def write_result(rec: dict) -> None:
        with write_lock:
            i = next(counter)
            mark = "✓" if rec.get("label") else "✗"
            print(f"  judge [{i}/{total}] {rec['question_id']} "
                  f"({rec['question_type']}): {mark}")
            append_jsonl(judged_path, rec)
            judged_records.append(rec)

    if args.workers <= 1:
        for rec in run_records:
            write_result(judge_one(rec))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(judge_one, r) for r in run_records]
            for fut in as_completed(futures):
                write_result(fut.result())

    return judged_records


# --- Round-level summary -------------------------------------------------

def compute_round_summary(judged: list[dict], runtime_sec: float) -> dict:
    agg = aggregate_summary(judged)
    return {
        "n_processed": len(judged),
        "primary_metric": agg.get("accuracy_overall", 0.0),
        "accuracy_overall": agg.get("accuracy_overall", 0.0),
        **{k: v for k, v in agg.items() if k.startswith("accuracy_by_qtype/")},
        "prefill_tokens_avg": agg.get("prefill_tokens_avg"),
        "prefill_tokens_p50": agg.get("prefill_tokens_p50"),
        "prefill_tokens_p95": agg.get("prefill_tokens_p95"),
        "latency_sec_avg": agg.get("latency_sec_avg"),
        "latency_sec_p50": agg.get("latency_sec_p50"),
        "latency_sec_p95": agg.get("latency_sec_p95"),
        "error_rate": agg.get("error_or_empty_rate", agg.get("error_rate", 0.0)),
        "topic_revisit_hit_rate": agg.get("topic_revisit_hit_rate"),
        "round_runtime_sec": runtime_sec,
    }


def stratify(questions: list[dict], limit: int | None) -> list[dict]:
    if not limit:
        return questions
    from collections import defaultdict
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q)
    k_per = max(1, limit // len(by_type))
    return [q for qs in by_type.values() for q in qs[:k_per]]


# --- Main ---------------------------------------------------------------

def main() -> None:
    load_dotenv()
    cfg = load_config()
    seg, mw, ev = cfg["segmenter"], cfg["memory_window"], cfg["evaluation"]
    imp_cfg = cfg["topic_importance"]
    stm_cfg = cfg["stm"]
    round_cfg = cfg["round"]

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   choices=["sliding", "full", "rag", "rag-summary", "rag-observation",
                            "hi-em", "hi-em-full-v1", "hi-em-full-v2",
                            # Bounded Cosine MAP (v3.1 = legacy alias for v3.1.1)
                            "hi-em-full-v3.1", "hi-em-full-v3.1.1",
                            # Bounded Cosine MAP + sub-linear sCRP count (v3.2 = legacy alias for v3.2.1)
                            "hi-em-full-v3.2", "hi-em-full-v3.2.1"])
    p.add_argument("--benchmark", choices=["longmemeval", "locomo"],
                   default=None,
                   help="Benchmark spec. Default inferred from --data path "
                        "(.../LongMemEval/... → longmemeval, .../locomo/... → locomo).")
    p.add_argument("--data", default="benchmarks/LongMemEval/data/longmemeval_oracle.json")
    p.add_argument("--exp-id", default=None,
                   help="Resume an existing experiment by id, or auto-generate.")
    p.add_argument("--session", default=None,
                   help="Session id under results/sessions/. session.json common_config "
                        "merges into config (CLI > session > hiem.json defaults).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--questions-per-round", type=int, default=50,
                   help="Round granularity (resume unit). Default 50.")
    p.add_argument("--model", default=os.environ.get("HIEM_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--judge-model",
                   default=os.environ.get("HIEM_JUDGE_MODEL")
                           or os.environ.get("HIEM_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--max-tokens", type=int,
                   default=int(os.environ.get("HIEM_RUN_MAX_TOKENS", "800")))
    p.add_argument("--judge-max-tokens", type=int,
                   default=int(os.environ.get("HIEM_JUDGE_MAX_TOKENS", "256")))
    p.add_argument("--temperature", type=float,
                   default=float(os.environ.get("HIEM_TEMPERATURE", "0.7")))
    p.add_argument("--no-thinking", action="store_true",
                   default=os.environ.get("HIEM_NO_THINKING", "").lower()
                           in {"1", "true", "yes", "on"})
    # ``.env`` may carry an inline-comment value; only honor real device names.
    _env_device = (os.environ.get("HIEM_DEVICE") or "").strip()
    p.add_argument(
        "--device",
        default=_env_device if _env_device in {"cuda", "mps", "cpu"} else None,
    )
    _env_emb_backend = (os.environ.get("HIEM_EMBEDDING_BACKEND") or "").strip().lower()
    p.add_argument(
        "--embedding-backend",
        choices=["local", "api"],
        default=_env_emb_backend if _env_emb_backend in {"local", "api"} else "local",
        help="local: sentence-transformers (bge); api: OpenAI-compatible /v1/embeddings (OpenRouter etc.)",
    )
    p.add_argument(
        "--embedding-model",
        default=os.environ.get("HIEM_EMBEDDING_MODEL"),
        help="Override embedding model id (e.g. openai/text-embedding-3-small).",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sliding-k", type=int, default=ev["sliding_k"])
    p.add_argument("--rag-k", type=int, default=ev["rag_k"])
    p.add_argument("--k-topics", type=int, default=mw["k_topics"])
    p.add_argument("--k-turns-per-topic", type=int, default=mw["k_turns_per_topic"])
    p.add_argument("--alpha", type=float, default=seg["alpha"])
    p.add_argument("--lmda", type=float, default=seg["lmda"])
    p.add_argument("--sigma0-sq", type=float, default=seg["sigma0_sq"])
    # hi-em-full-v1 HP (Phase 2-Full)
    p.add_argument("--round-size", type=int,
                   default=round_cfg["turns_per_round"] // 2)
    p.add_argument("--stm-max-topics", type=int, default=stm_cfg["max_topics"])
    p.add_argument("--stm-max-turns", type=int, default=stm_cfg["max_turns"])
    p.add_argument("--promotion-threshold", type=float,
                   default=stm_cfg["promotion_threshold"])
    p.add_argument("--importance-alpha", type=float, nargs=4,
                   metavar=("A1", "A2", "A3", "A4"),
                   default=imp_cfg["alpha"])
    p.add_argument("--lambda-r", type=float, default=imp_cfg["lambda_r"])
    p.add_argument("--lambda-freq", type=float, default=imp_cfg["lambda_freq"])
    p.add_argument("--min-floor", type=float, default=imp_cfg["min_floor"])
    # v3.x (Bounded Cosine variants) only
    p.add_argument("--tau", type=float, default=50.0,
                   help="Cosine temperature for hi-em-full-v3.x. "
                        "Score = log prior + tau * cos(s, mu_or_pred). "
                        "Default 50 (TopiOCQA-tuned).")
    p.add_argument("--cos-threshold", type=float, default=0.7,
                   help="New-cluster cosine baseline for hi-em-full-v3.x — a fresh "
                        "topic competes as if it had cos = cos_threshold. "
                        "Default 0.7 (TopiOCQA-tuned).")
    p.add_argument("--beta", type=float, default=0.5,
                   help="v3.2 only: sub-linear sCRP count exponent. "
                        "prior_k ∝ (C_k + 1)^beta + lambda·1[prev=k]. "
                        "beta=1.0 reduces to v3.1; beta<1 squashes the "
                        "rich-get-richer accumulation that drives mega-topic. "
                        "Default 0.5 (sqrt).")
    p.add_argument("--no-token-count", action="store_true")
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--wandb-project", default="hi-em-phase4")
    args = p.parse_args()

    # Backwards-compat aliases: old method names normalize to new ones.
    _METHOD_ALIASES = {
        "hi-em-full-v3.1": "hi-em-full-v3.1.1",
        "hi-em-full-v3.2": "hi-em-full-v3.2.1",
    }
    if args.method in _METHOD_ALIASES:
        args.method = _METHOD_ALIASES[args.method]

    results_root = Path(args.results_root)

    # Session common config (merged on top of hiem.json, CLI still wins).
    session_common: dict[str, Any] = {}
    if args.session:
        sjson = session_dir(args.session, root=results_root) / "session.json"
        if sjson.exists():
            session_common = load_json(sjson).get("common_config", {})

    # Compute or use exp_id.
    parts = [args.method]
    if args.session:
        parts.insert(0, args.session.split("_", 1)[-1])  # short tag from session
    exp_id = args.exp_id or make_experiment_id(*parts, suffix=Path(args.data).stem)

    # Build full config snapshot (immutable).
    config_snapshot = {
        "method": args.method,
        "data": args.data,
        "model": args.model,
        "judge_model": args.judge_model,
        "max_tokens": args.max_tokens,
        "judge_max_tokens": args.judge_max_tokens,
        "temperature": args.temperature,
        "no_thinking": args.no_thinking,
        "device": args.device or "auto",
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "workers": args.workers,
        "limit": args.limit,
        "stratify": args.stratify,
        "questions_per_round": args.questions_per_round,
        "segmenter": {"alpha": args.alpha, "lmda": args.lmda, "sigma0_sq": args.sigma0_sq,
                      "tau": args.tau, "cos_threshold": args.cos_threshold,
                      "beta": args.beta},
        "memory_window": {"k_topics": args.k_topics, "k_turns_per_topic": args.k_turns_per_topic},
        "evaluation": {"sliding_k": args.sliding_k, "rag_k": args.rag_k},
        "session_common": session_common,
    }
    seeds = {
        "data_seed": None,            # dataset is static
        "sampling_seed": None,        # LLM temperature → genuinely stochastic
        "init_seed": None,
        "env_seed": None,
    }
    meta = ExperimentMeta(
        experiment_id=exp_id, session_id=args.session,
        config=config_snapshot, seeds=seeds,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
    )
    exp_dir = create_experiment(meta, root=results_root)
    print(f"[exp] {exp_dir}")
    print(f"[env] {os.environ.get('OPENAI_BASE_URL', '(default)')}  model={args.model}")

    # Idempotent re-invocation: if the experiment already finished, return early
    # without redoing any rounds. find_resumable_experiment returns None for both
    # "fresh" and "completed", so we must disambiguate via completed.json.
    if (exp_dir / "completed.json").exists():
        print("[skip] experiment already completed; nothing to do")
        return

    # --- Resolve benchmark + load questions --------------------------------
    benchmark = args.benchmark
    if benchmark is None:
        dpath = str(args.data).lower()
        if "locomo" in dpath:
            benchmark = "locomo"
        else:
            benchmark = "longmemeval"
    print(f"[benchmark] {benchmark}")

    if benchmark == "locomo":
        all_questions = build_locomo_entries(args.data)
        if args.stratify:
            # 10 conv × 5 cat × per_cat = limit (limit interpreted as total target).
            from hi_em.locomo_loader import stratify_by_category
            per_cat = max(1, (args.limit or 50) // (10 * 5))
            all_questions = stratify_by_category(all_questions, per_cat,
                                                 also_per_sample=True)
        elif args.limit:
            all_questions = all_questions[: args.limit]
    else:
        all_questions = json.loads(Path(args.data).read_text())
        if args.stratify:
            all_questions = stratify(all_questions, args.limit)
        elif args.limit:
            all_questions = all_questions[: args.limit]
    n_total = len(all_questions)
    rounds_n = max(1, (n_total + args.questions_per_round - 1) // args.questions_per_round)
    print(f"[data] {n_total} questions → {rounds_n} rounds × {args.questions_per_round}")

    # STM top-K importance turn-count stats: hi-em-full-v1 only, one-shot.
    # File-existence gate honors "그뒤로 그 파일이 있으면 다시 안만들게" (CLAUDE.md).
    if args.method == "hi-em-full-v1":
        explicit = os.environ.get("HIEM_STM_TOPK_STATS_PATH")
        topk_stats_path = (
            Path(explicit) if explicit
            else REPO_ROOT / "outputs" / "stm_importance_topk_stats.jsonl"
        )
        if topk_stats_path.exists():
            print(f"[stm-topk-stats] {topk_stats_path} exists — skipping")
            os.environ.pop("HIEM_STM_TOPK_STATS_PATH", None)
        else:
            os.environ["HIEM_STM_TOPK_STATS_PATH"] = str(topk_stats_path)
            print(f"[stm-topk-stats] writing per-round top-3/4/5 stats to {topk_stats_path}")

    # Encoder + LLM (heavy, do once outside the round loop).
    needs_encoder = args.method in {
        "rag", "rag-summary", "rag-observation",
        "hi-em", "hi-em-full-v1", "hi-em-full-v2",
        "hi-em-full-v3.1.1", "hi-em-full-v3.2.1",
    }
    encoder = (
        make_encoder(
            device=args.device,
            backend=args.embedding_backend,
            model=args.embedding_model,
        )
        if needs_encoder
        else None
    )
    if encoder is not None:
        print(
            f"[encoder] backend={args.embedding_backend} "
            f"model={getattr(encoder, 'model_name', '?')} "
            f"device={getattr(encoder, 'device', '?')} dim={encoder.dim}"
        )
        if not args.no_token_count:
            count_prefill_tokens([{"role": "user", "content": "warmup"}], args.model)
    elif not args.no_token_count:
        count_prefill_tokens([{"role": "user", "content": "warmup"}], args.model)

    llm = OpenAIChatLLM()
    llm_kwargs: dict = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    if args.no_thinking:
        # vLLM honors chat_template_kwargs.enable_thinking; Crts honors
        # OpenRouter-style reasoning.enabled. Send both — each backend
        # ignores the other's key.
        llm_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning": {"enabled": False},
        }

    # Hi-EM ltm root: per-experiment, isolated from archive.
    ltm_root = exp_dir / "working_state" / "ltm"
    if args.method in {"hi-em", "hi-em-full-v1",
                        "hi-em-full-v3.1.1", "hi-em-full-v3.2.1"} and ltm_root.exists():
        # Resume-safe: per-question conv_id dirs are ephemeral; cleaning them is OK
        # because preload_history rebuilds from the input for the current round.
        pass

    # LoCoMo-style benchmarks share a conversation across many test questions
    # (1 sample_id × ~200 Q). Build Hi-EM state once per sample to avoid the
    # 200× preload cost; each Q answers read-only via :meth:`HiEM.eval_query`.
    hiem_cache: HiEMConvCache | None = None
    if benchmark == "locomo" and args.method in {
        "hi-em", "hi-em-full-v1", "hi-em-full-v2",
        "hi-em-full-v3.1.1", "hi-em-full-v3.2.1",
    }:
        hiem_cache = HiEMConvCache(
            ltm_root=ltm_root, encoder=encoder, llm=llm,
            model=args.model, llm_kwargs=llm_kwargs, args=args,
        )
        print("[hiem-cache] enabled (per-sample_id conv-level state cache)")

    # Same idea for RAG-family methods: cache per-conversation embedding
    # matrices so the encoder (which is lock-serialized inside QueryEncoder)
    # does each conversation's corpus exactly once instead of ~200 times.
    enc_cache: EncoderCache | None = None
    if benchmark == "locomo" and args.method in {
        "rag", "rag-summary", "rag-observation"
    } and encoder is not None:
        enc_cache = EncoderCache(encoder=encoder)
        print("[enc-cache] enabled (per-sample_id embedding cache)")

    # Resume.
    last = find_resumable_experiment(exp_id, root=results_root)
    start_round = (last + 1) if last is not None else 1
    if last:
        print(f"[resume] last completed round {last} → starting round {start_round}")
    if start_round > rounds_n:
        print("[skip] all rounds already complete; marking experiment done.")
        mark_experiment_complete(exp_id, total_rounds=rounds_n, root=results_root)
        return

    # Wandb (optional, no-op if not authed).
    # Sidecar holds the run id as plain text (see WandbRun.__init__);
    # never load_json — it's not JSON.
    sidecar = exp_dir / "wandb-run-id.txt"
    resume_id = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else None
    wb = WandbRun(
        project=args.wandb_project,
        name=exp_id,
        group=args.session or Path(args.data).stem,
        config=config_snapshot,
        sidecar_path=sidecar,
        resume_id=resume_id or None,
    )

    # Round loop.
    prev_summary: dict | None = None
    if start_round > 1:
        prev = round_dir(exp_id, start_round - 1, root=results_root) / "summary.json"
        if prev.exists():
            prev_summary = load_json(prev)

    for r in range(start_round, rounds_n + 1):
        rdir = round_dir(exp_id, r, root=results_root)
        rdir.mkdir(parents=True, exist_ok=True)
        i_lo = (r - 1) * args.questions_per_round
        i_hi = min(i_lo + args.questions_per_round, n_total)
        batch = all_questions[i_lo:i_hi]
        print(f"\n=== round {r}/{rounds_n}  questions[{i_lo}:{i_hi}]  ({len(batch)}) ===")

        t_round = time.perf_counter()
        run_records = phase_run(
            batch, rdir, args, encoder, llm, llm_kwargs, ltm_root,
            hiem_cache=hiem_cache, enc_cache=enc_cache,
        )
        if benchmark == "locomo":
            judged = phase_judge_locomo(run_records, batch, rdir, args)
        else:
            judged = phase_judge(run_records, batch, rdir, args, llm, llm_kwargs)

        runtime = time.perf_counter() - t_round
        summary = compute_round_summary(judged, runtime)
        warns = sanity_check_summary(summary, prev_summary)
        if warns:
            for w in warns:
                print(f"  [sanity] WARN: {w}")
        mark_round_complete(exp_id, r, summary, root=results_root)
        if wb.enabled:
            wb.log({"round": r, **{k: v for k, v in summary.items() if isinstance(v, (int, float))}}, step=r)
        prev_summary = summary
        print(f"  → round {r} done: primary={summary['primary_metric']:.3f}  "
              f"runtime={runtime:.1f}s")

    # --- Experiment-level final summary (across ALL rounds) ----------------
    # Collected on disk (results/.../summary.json) AND pushed to wandb.summary.
    # This is the row that compares against archive baselines (R-11) and is
    # what cross-experiment session reports aggregate.
    all_judged: list[dict] = []
    for r in range(1, rounds_n + 1):
        jp = round_dir(exp_id, r, root=results_root) / "judged.jsonl"
        all_judged.extend(load_jsonl(jp))
    final = aggregate_summary(all_judged)
    final["schema_version"] = SUMMARY_JSON_SCHEMA_VERSION
    final["primary_metric"] = final.get("accuracy_overall", 0.0)
    final["n_questions"] = len(all_judged)
    final["n_rounds"] = rounds_n
    save_json(exp_dir / "summary.json", final)

    if args.method == "hi-em-full-v1" and os.environ.get("HIEM_STM_TOPK_STATS_PATH"):
        from hi_em.stm_topk_stats import flush_aggregate
        written = flush_aggregate()
        if written:
            print(f"[stm-topk-stats] aggregated stats written to {written}")

    mark_experiment_complete(exp_id, total_rounds=rounds_n, root=results_root)

    if wb.enabled:
        wb.summary(**{k: v for k, v in final.items() if isinstance(v, (int, float))})
        wb.finish()

    # --- stdout table-row (matches user comparison-table format) -----------
    if benchmark == "locomo":
        qtype_cols = [
            "multi-hop", "temporal-reasoning", "open-domain",
            "single-hop", "adversarial",
        ]
    else:
        qtype_cols = [
            "knowledge-update", "multi-session", "single-session-assistant",
            "single-session-preference", "single-session-user", "temporal-reasoning",
        ]
    print(f"\n=== Final Summary (experiment_id={exp_id}) ===")
    print(f"  n_questions = {final['n_questions']}  ({rounds_n} rounds)")
    print(f"  Overall accuracy = {final.get('accuracy_overall', 0.0):.3f}")
    print(f"  By question_type:")
    for qt in qtype_cols:
        v = final.get(f"accuracy_by_qtype/{qt}")
        print(f"    {qt:30s} = {v:.2f}" if v is not None else f"    {qt:30s} = -")
    if "topic_revisit_hit_rate" in final:
        print(f"  topic_revisit_hit_rate = {final['topic_revisit_hit_rate']:.2f}")
    print(f"  prefill_tokens p50/p95 = "
          f"{final.get('prefill_tokens_p50', 0):.0f} / "
          f"{final.get('prefill_tokens_p95', 0):.0f}")
    print(f"  latency_sec  p50/p95 = "
          f"{final.get('latency_sec_p50', 0):.2f} / "
          f"{final.get('latency_sec_p95', 0):.2f}")
    print(f"  error_rate = {final.get('error_or_empty_rate', final.get('error_rate', 0.0)):.3f}")

    # --- Latency breakdown (streaming-mode TTFT / TPOT) -------------------
    if "ttft_sec_p50" in final or "tpot_sec_p50" in final:
        ttft_p50 = final.get("ttft_sec_p50", 0.0) * 1000
        ttft_p95 = final.get("ttft_sec_p95", 0.0) * 1000
        tpot_p50 = final.get("tpot_sec_p50", 0.0) * 1000
        tpot_p95 = final.get("tpot_sec_p95", 0.0) * 1000
        gen_p50 = final.get("gen_sec_p50", 0.0)
        gen_p95 = final.get("gen_sec_p95", 0.0)
        out_p50 = final.get("output_tokens_p50", 0.0)
        out_p95 = final.get("output_tokens_p95", 0.0)
        # Total wallclock for this experiment (sum of round runtimes is
        # close enough; prefer ``time.perf_counter`` accumulator if you
        # want exact). Here we read from disk: sum of per-round runtime.
        total_wall = 0.0
        for r in range(1, rounds_n + 1):
            sp = round_dir(exp_id, r, root=results_root) / "summary.json"
            if sp.exists():
                total_wall += float(load_json(sp).get("round_runtime_sec", 0.0))
        print()
        print(f"  ─── Latency (streaming) ───────────────────────────────")
        print(f"    TTFT (Time To First Token)  p50 / p95 = "
              f"{ttft_p50:7.1f} ms / {ttft_p95:7.1f} ms")
        print(f"    TPOT (Time Per Output Token) p50 / p95 = "
              f"{tpot_p50:7.2f} ms / {tpot_p95:7.2f} ms")
        print(f"    Output tokens                p50 / p95 = "
              f"{out_p50:7.0f}    / {out_p95:7.0f}   ")
        print(f"    Gen wallclock (sec)          p50 / p95 = "
              f"{gen_p50:7.2f}   / {gen_p95:7.2f}  ")
        print(f"    Total experiment runtime              = "
              f"{total_wall/60:6.1f} min ({total_wall:7.1f} s)")
        print(f"  ──────────────────────────────────────────────────────")
    print(f"\n[done] results: {exp_dir}")
    print(f"       summary: {exp_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
