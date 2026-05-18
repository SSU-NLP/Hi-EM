#!/usr/bin/env python3
"""LLM pseudo-label 기반 segmentation HP sweep (codex 2026-05-16 수정 프로토콜).

요지 (decision-log 2026-05-16 참조):
  * Qwen3.5-9B 가 매긴 turn별 topic cluster 는 **gold 가 아니라 pseudo-label**.
  * 1차 metric = evidence-cohesion (정답 evidence 턴이 같은 topic 에 모이나).
  * 보조 metric = pseudo-label 대비 boundary F1 / ARI / NMI.
  * 데이터 = LongMemEval **단일 idx** haystack (cross-idx concat 금지 — seam 회피).
  * segmenter v2 + v3.3.4, alpha×lmda grid sweep. v3.3.4 는 GRU 발동 여부 동시 기록.

LLM/임베딩 모두 .env 의 Crts 프록시 사용 (project convention).
embedding backend 는 make_encoder() 가 env (HIEM_EMBEDDING_BACKEND=api) 따라
실험과 동일한 bge-base-en-v1.5 를 씀.

Usage:
    uv run python scripts/llm_pseudolabel_sweep.py --idx 374 [--reuse-pseudo]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import make_encoder  # noqa: E402
from hi_em.llm import OpenAIChatLLM  # noqa: E402
from hi_em.sem_core import HiEMSegmenter  # noqa: E402
from hi_em.sem_core_v334_rnn_var import HiEMSegmenterV334  # noqa: E402

DATA = REPO_ROOT / "benchmarks" / "LongMemEval" / "data" / "longmemeval_oracle.json"
ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0]
LMDAS = [0.0, 1.0, 3.0, 10.0]


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_user_turns(idx: int) -> tuple[dict, list[dict]]:
    entry = json.loads(DATA.read_text())[idx]
    turns: list[dict] = []
    for si, sess in enumerate(entry["haystack_sessions"], start=1):
        for t in sess:
            if t.get("role") == "user":
                turns.append(
                    {
                        "sess": si,
                        "content": t.get("content", ""),
                        "has_answer": bool(t.get("has_answer", False)),
                    }
                )
    return entry, turns


# ----------------------------------------------------------------------
# Qwen pseudo-label (turn별 topic cluster id)
# ----------------------------------------------------------------------
def _extract_clusters(txt: str) -> list[int]:
    """thinking 모델의 <think>...</think> preamble 을 제거하고 마지막
    ``{...}`` JSON 블록에서 clusters 를 뽑는다 (preamble 안의 brace 에
    낚이지 않도록 닫는 brace 부터 역으로 매칭)."""
    import re

    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
    end = txt.rfind("}")
    if end == -1:
        raise ValueError(f"no JSON object in LLM output: {txt[:200]!r}")
    depth = 0
    for i in range(end, -1, -1):
        if txt[i] == "}":
            depth += 1
        elif txt[i] == "{":
            depth -= 1
            if depth == 0:
                return [int(c) for c in json.loads(txt[i : end + 1])["clusters"]]
    raise ValueError(f"unbalanced JSON in LLM output: {txt[:200]!r}")


_SYS_P = (
    # Crts 가 chat_template_kwargs.enable_thinking 을 무시하므로 Qwen3
    # soft switch '/no_think' 를 프롬프트에 직접 박아 thinking 차단
    # (Cloudflare 120s origin cap 회피의 핵심).
    "/no_think\n"
    "You segment a single multi-session conversation (only the user's turns "
    "are shown) into topics. Assign every NEW turn an integer topic cluster "
    "id. Some earlier turns are shown as already-labeled context in the form "
    "'(id=K) text' — DO NOT relabel them, but REUSE id K for a new turn that "
    "is about that same topic (a topic can recur, even non-adjacent). Only "
    "introduce the next unused integer for a genuinely new topic. Output ONLY "
    "a JSON object {\"clusters\": [id, ...]} with EXACTLY one id per NEW turn, "
    "in order. No prose."
)


def _legend(gold: list[int], texts: list[str], gist_chars: int = 90) -> str:
    """이미 배정된 global id 별 1줄 gist (그 id 의 첫 turn 앞부분).
    full prior text 대신 압축 → 프롬프트·추론량 bound (Cloudflare 120s 캡)."""
    seen: dict[int, str] = {}
    for g, t in zip(gold, texts):
        if g not in seen:
            seen[g] = t.replace("\n", " ")[:gist_chars]
    if not seen:
        return ""
    lines = "\n".join(f"  id={g}: {seen[g]}" for g in sorted(seen))
    return f"Existing topics so far:\n{lines}\n\n"


def _chunk_call(
    llm, model: str, thinking: bool, timeout: float, max_tokens: int,
    legend: str, new_texts: list[str],
) -> list[int]:
    body = (
        f"{legend}Label these {len(new_texts)} NEW turns "
        f"(reuse an existing id above if same topic, else next unused int; "
        f"output exactly {len(new_texts)} ids):\n"
        + "\n".join(f"[{i}] {t}" for i, t in enumerate(new_texts))
    )
    txt = llm.chat(
        [
            {"role": "system", "content": _SYS_P},
            {"role": "user", "content": body},
        ],
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    ids = _extract_clusters(txt)
    if len(ids) != len(new_texts):
        raise ValueError(
            f"chunk: Qwen returned {len(ids)} ids for {len(new_texts)} turns"
        )
    return ids


def qwen_pseudo_clusters(
    turns: list[dict], model: str, thinking: bool, timeout: float,
    max_tokens: int, chunk_size: int,
) -> list[int]:
    """작은 청크 sequential 라벨링. 이전 청크까지 배정된 topic 을 **압축
    legend**(id별 1줄 gist)로만 보여줌 → 같은 토픽이면 그 id 재사용 (청크
    간 cluster id 연속 보정). full prior text 를 안 실어 호출당 생성이
    Cloudflare 120s origin-read-timeout 안에 끝나도록 bound."""
    # streaming (기본) — Crts/qwen3.5-9b 는 non-stream 에 빈 content 반환.
    llm = OpenAIChatLLM()
    texts = [t["content"] for t in turns]
    gold: list[int] = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        legend = _legend(gold, texts[:start])
        t_c = time.perf_counter()
        ids = _chunk_call(
            llm, model, thinking, timeout, max_tokens, legend, chunk
        )
        gold.extend(int(c) for c in ids)
        print(
            f"[pseudo] chunk {start}-{start+len(chunk)-1} "
            f"({time.perf_counter()-t_c:.0f}s) → ids {ids}",
            flush=True,
        )
    if len(gold) != len(turns):
        raise ValueError(f"{len(gold)} ids for {len(turns)} turns")
    return gold


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def boundary_f1(pred_topics: list[int], gold_clusters: list[int]) -> float:
    """Transition-level boundary F1 (i=1..n-1). boundary_i = topic_i != topic_{i-1}."""
    gp = [gold_clusters[i] != gold_clusters[i - 1] for i in range(1, len(gold_clusters))]
    pp = [pred_topics[i] != pred_topics[i - 1] for i in range(1, len(pred_topics))]
    tp = sum(1 for g, p in zip(gp, pp) if g and p)
    fp = sum(1 for g, p in zip(gp, pp) if p and not g)
    fn = sum(1 for g, p in zip(gp, pp) if g and not p)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * P * R / (P + R) if (P + R) else 0.0


def run_segmenter(version: str, embs: np.ndarray, alpha: float, lmda: float):
    """Return per-turn topics + v3.3.4 GRU diagnostics."""
    dim = embs.shape[1]
    rnn_min = 2
    if version == "v3.3.4":
        seg = HiEMSegmenterV334(dim=dim, alpha=alpha, lmda=lmda)
    else:
        seg = HiEMSegmenter(dim=dim, alpha=alpha, lmda=lmda, sigma0_sq=0.01)

    topics: list[int] = []
    gru_used = 0
    gru_cos: list[float] = []
    cen_cos: list[float] = []
    for s in embs:
        if version == "v3.3.4":
            snap_pred = {k: tp.predict_next().copy() for k, tp in enumerate(seg.topics)}
            snap_n = {k: tp.n for k, tp in enumerate(seg.topics)}
            snap_mu = {k: tp.mu.copy() for k, tp in enumerate(seg.topics)}
            k, _ = seg.assign(s)
            if snap_n.get(k, 0) >= rnn_min:
                gru_used += 1
                gru_cos.append(float(np.dot(snap_pred[k], s)))
                mu = snap_mu[k]
                if np.linalg.norm(mu) > 0:
                    cen_cos.append(float(np.dot(mu, s)))
        else:
            k, _ = seg.assign(s)
        topics.append(int(k))

    return {
        "topics": topics,
        "gru_used": gru_used,
        "mean_cos_gru_next": float(np.mean(gru_cos)) if gru_cos else float("nan"),
        "mean_cos_centroid_next": float(np.mean(cen_cos)) if cen_cos else float("nan"),
    }


def evaluate(topics, gold_clusters, ev_idx):
    n = len(topics)
    ev_topics = sorted({topics[i] for i in ev_idx})
    m = {
        "raw_n_topics": len(set(topics)),
        "new_topic_rate": round(len(set(topics)) / n, 3),
        "max_topic_share": round(
            max(topics.count(t) for t in set(topics)) / n, 3
        ),
        "n_evidence": len(ev_idx),
        "evidence_topics": ev_topics,
        "mean_evidence_topics": len(ev_topics),
        "evidence_cohesion": int(len(ev_topics) == 1),  # 1차 metric
    }
    if gold_clusters is None:  # LLM pseudo-label 없음 → 보조 metric 생략
        m.update(boundary_f1=None, ari=None, nmi=None)
    else:
        m.update(
            boundary_f1=round(boundary_f1(topics, gold_clusters), 3),
            ari=round(adjusted_rand_score(gold_clusters, topics), 3),
            nmi=round(normalized_mutual_info_score(gold_clusters, topics), 3),
        )
    return m


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=374)
    ap.add_argument("--model", default=os.environ.get("HIEM_MODEL", "qwen/qwen3.5-9b"))
    ap.add_argument("--reuse-pseudo", action="store_true")
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Qwen pseudo-label 생략 (Crts qwen3.5-9b 가 이 프롬프트에서 "
        "reasoning 으로 max_tokens 소진 → content 빈응답, 어떤 flag 로도 "
        "thinking 못 끔). Codex 1차 metric evidence-cohesion + segmenter "
        "진단만 산출, bF1/ARI/NMI 생략.",
    )
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="Qwen thinking 모드 ON (enable_thinking=True). "
        "기본 OFF. <think> preamble 은 robust 파서가 제거.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="LLM 요청당 timeout(초). 무한 poll hang 방지 (default 240).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="LLM max_tokens. thinking ON 이면 추론+답 둘 다 들어가야 "
        "하므로 크게 (예 16000).",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="청크당 새로 라벨링할 turn 수 (default 12). 작을수록 호출당 "
        "출력 짧아 thinking 예산/ timeout 안전, 호출 수 증가.",
    )
    args = ap.parse_args()

    name = f"2026-05-16_llm_pseudolabel_sweep_lme{args.idx}"
    out_dir = REPO_ROOT / "outputs" / "experiments" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_path = out_dir / "qwen_pseudo_clusters.json"

    entry, turns = load_user_turns(args.idx)
    ev_idx = [i for i, t in enumerate(turns) if t["has_answer"]]
    print(
        f"idx={args.idx} qid={entry.get('question_id')} "
        f"qtype={entry.get('question_type')}  user턴={len(turns)} "
        f"evidence(has_answer)={len(ev_idx)} @ {ev_idx}"
    )

    if args.no_llm:
        gold = None
        print(
            "[pseudo] --no-llm: Qwen pseudo-label 생략. "
            "evidence-cohesion(1차) + segmenter 진단만 산출.",
            flush=True,
        )
    elif args.reuse_pseudo and pseudo_path.exists():
        gold = json.loads(pseudo_path.read_text())["clusters"]
        print(f"[pseudo] reuse {pseudo_path}")
    else:
        print(
            f"[pseudo] Qwen ({args.model}) labeling {len(turns)} turns "
            f"(thinking={args.thinking}, timeout={args.timeout}s, "
            f"max_tokens={args.max_tokens}, chunk={args.chunk_size}) ...",
            flush=True,
        )
        t0 = time.perf_counter()
        gold = qwen_pseudo_clusters(
            turns, args.model, args.thinking, args.timeout,
            args.max_tokens, args.chunk_size,
        )
        pseudo_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "idx": args.idx,
                    "clusters": gold,
                    "note": "LLM pseudo-label, NOT human gold (decision-log 2026-05-16)",
                },
                indent=2,
            )
        )
        print(f"[pseudo] done in {time.perf_counter()-t0:.1f}s → {pseudo_path}")
    if gold is not None:
        print(f"[pseudo] Qwen clusters: {gold}  (n_topics={len(set(gold))})")

    enc = make_encoder()
    embs = np.asarray(enc.encode([t["content"] for t in turns]))

    rows = []
    for version in ("v2", "v3.3.4"):
        for a in ALPHAS:
            for l in LMDAS:
                r = run_segmenter(version, embs, a, l)
                m = evaluate(r["topics"], gold, ev_idx)
                rows.append(
                    {
                        "version": version,
                        "alpha": a,
                        "lmda": l,
                        **m,
                        "gru_used": r["gru_used"],
                        "cos_gru": round(r["mean_cos_gru_next"], 3)
                        if not np.isnan(r["mean_cos_gru_next"])
                        else None,
                        "cos_cen": round(r["mean_cos_centroid_next"], 3)
                        if not np.isnan(r["mean_cos_centroid_next"])
                        else None,
                    }
                )

    write_report(out_dir, args, entry, turns, ev_idx, gold, rows)
    print(f"\nREPORT → {out_dir/'REPORT.md'}")


def write_report(out_dir, args, entry, turns, ev_idx, gold, rows):
    n = len(turns)
    has_llm = gold is not None
    gold_n = len(set(gold)) if has_llm else None
    # 1차 기준: evidence_cohesion ↑, 동률이면 mean_evidence_topics ↓,
    # raw_n_topics ↓ (덜 과분절), LLM 있으면 ARI ↑ 보조.
    best = sorted(
        rows,
        key=lambda r: (
            -r["evidence_cohesion"],
            r["mean_evidence_topics"],
            r["raw_n_topics"],
            -(r["ari"] if r["ari"] is not None else -1),
        ),
    )[0]
    fmt = lambda v: "-" if v is None else v

    def tbl(version):
        head = (
            "| α | λ | ev_cohesion | ev_topics | raw_topics | new_rate | "
            "max_share | bF1 | ARI | NMI | gru_used | cosGRU | cosCen |\n"
            "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
        )
        body = ""
        for r in rows:
            if r["version"] != version:
                continue
            star = " ⭐" if r is best else ""
            body += (
                f"| {r['alpha']:g} | {r['lmda']:g} | {r['evidence_cohesion']} | "
                f"{r['mean_evidence_topics']} | {r['raw_n_topics']} | "
                f"{r['new_topic_rate']} | {r['max_topic_share']} | "
                f"{fmt(r['boundary_f1'])} | {fmt(r['ari'])} | {fmt(r['nmi'])} | "
                f"{r['gru_used']} | {fmt(r['cos_gru'])} | {fmt(r['cos_cen'])} |{star}\n"
            )
        return head + body

    ev_list = ", ".join(
        f"[{i}]sess{turns[i]['sess']}" for i in ev_idx
    )
    md = f"""# REPORT — LLM pseudo-label segmentation sweep (LongMemEval idx {args.idx})

## 1. 실험 setup

- **목적**: α/λ 가 segmentation 의 evidence-topic cohesion 을 어디서 깨는지,
  Qwen pseudo-label 대비 boundary/cluster 정합과 함께 직접 확인.
  (codex 2026-05-16 수정 프로토콜 — LLM 라벨은 gold 아님, pseudo-label.)
- **데이터**: `benchmarks/LongMemEval/data/longmemeval_oracle.json` idx={args.idx}
  단일 haystack (cross-idx concat 없음 — seam 회피).
  qid={entry.get('question_id')} · qtype={entry.get('question_type')} ·
  user턴 {n}개 · {len(entry['haystack_sessions'])} sessions.
- **정답 evidence**: LongMemEval `has_answer=true` user턴 {len(ev_idx)}개 — {ev_list}.
  질문 1개당 evidence 가 같은 topic 에 모이면 retrieval atomicity 보존.
- **Qwen pseudo-label**: {'**생략 (--no-llm)** — Crts qwen3.5-9b 가 이 '
  'segmentation 프롬프트에서 reasoning 으로 max_tokens 소진 → content '
  '빈응답, enable_thinking/`/no_think` 어떤 것으로도 thinking 차단 불가 '
  '(probe 로 확정: finish_reason=length, reasoning_tokens 수천). 따라서 '
  'bF1/ARI/NMI 미산출.' if not has_llm else
  f'model=`{args.model}` (Crts, temp=0), turn별 topic cluster id. '
  f'**human gold 아님** (decision-log 2026-05-16). '
  f'Qwen clusters = `{gold}` → n_topics={gold_n}.'}
- **embedding**: `make_encoder()` env backend (실험과 동일).
- **segmenter**: v2 (`HiEMSegmenter`, σ₀²=0.01) · v3.3.4 (`HiEMSegmenterV334`).
- **HP grid**: α∈{ALPHAS} × λ∈{LMDAS} (각 segmenter 20 run, 총 40).
- **metric**:
  - 1차 `evidence_cohesion` = 1[모든 evidence 턴이 동일 topic]; `ev_topics`=evidence 가 흩어진 topic 수.
  - 보조 `bF1`/`ARI`/`NMI` = Qwen pseudo-label 대비.
  - 진단 `raw_topics`/`new_rate`/`max_share`; v3.3.4 `gru_used`(rnn_min_history=2 충족 턴 수)/`cosGRU`/`cosCen`.

## 2. 결과 — v2 (SEM core)

{tbl("v2")}

## 3. 결과 — v3.3.4

{tbl("v3.3.4")}

⭐ = 선택 기준(evidence_cohesion↑ → ev_topics↓ → raw_topics↓) 1위:
**{best['version']} α={best['alpha']:g} λ={best['lmda']:g}** —
evidence_cohesion={best['evidence_cohesion']}, ev_topics={best['mean_evidence_topics']},
raw_topics={best['raw_n_topics']}, bF1={fmt(best['boundary_f1'])}, ARI={fmt(best['ari'])}.

## 4. 해석

- evidence_cohesion=1 인 HP 가 있나? 없으면 어떤 α/λ 에서도 단일 질문의
  정답 근거가 한 topic 에 안 모인다는 뜻 → 과분절이 HP tuning 으로 안 풀림
  (codex 진단: 진짜 병목은 segmentation atomicity).
- v3.3.4 `gru_used`: 0 이면 그 HP 에서 GRU dynamics 死문 (α 큰 영역 예상).
  cosGRU < cosCen 이면 GRU 가 centroid 보다 예측 열세.
- bF1/ARI 최적점과 evidence_cohesion 최적점이 어긋나면 "Qwen 에 맞추기"가
  downstream 병목과 무관함을 보임.

## 5. 한계 / 검증 미해결

- Qwen pseudo-label 은 human gold 가 아니다. 1회 호출, temp=0 이라 재현되나
  Qwen 의 주관적 topic ontology 에 의존. boundary 는 cluster id 변화에서 파생.
- 단일 idx·단일 질문(evidence_cohesion 이 0/1 binary) → 표본 1. 경향 참고용.
- assistant 턴 제외(orchestrator 와 동일), STM/RoundProcessor 미적용 — 순수
  segmenter-level 진단. STM importance eviction 효과는 별도.
- 데이터 늘려 (다른 idx, LoCoMo conv0 단일대화) 일반화 필요.
"""
    (out_dir / "REPORT.md").write_text(md)


if __name__ == "__main__":
    main()
