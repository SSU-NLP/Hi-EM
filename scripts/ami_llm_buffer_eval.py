#!/usr/bin/env python3
"""실험2 — LLM 버퍼-폴링 분절. 버퍼 B초로 제한한 윈도우만 LLM에 주고 경계 검출.
버퍼 작을수록 문맥 부족 → 성능↓. Hi-OnTop(0버퍼) 대비 trade-off 곡선용."""
from __future__ import annotations
import json, os, re, sys, argparse
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO/".env")
TOPIC = REPO/"data"/"ami"/"topic"

WIN_PROMPT = """These are consecutive utterances from a meeting (a short window of the conversation).
List the utterance numbers where a NEW topic clearly begins within this window.
If no clear topic change, return [].
Return ONLY a JSON array, e.g. [12] or [].

Utterances:
{block}

JSON array:"""


def parse_bounds(txt, valid):
    for cand in reversed(re.findall(r"\[[\d,\s]*\]", txt or "")):
        try:
            arr = json.loads(cand)
            return sorted({int(x) for x in arr if int(x) in valid})
        except Exception:
            continue
    return []


def tol_f1(gold, pred, tol):
    if not pred or not gold: return 0.0
    pr = sum(1 for i in pred if any(abs(i-j) <= tol for j in gold))/len(pred)
    rc = sum(1 for j in gold if any(abs(i-j) <= tol for i in pred))/len(gold)
    return 2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/qwen/qwen3.5-27b")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--buffers", default="10,30,60,120")
    args = ap.parse_args()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    bufs = [int(x) for x in args.buffers.split(",")]
    mids = sorted(m["meeting"] for m in json.load(open(TOPIC/"manifest.json")))[:args.limit]

    def ask(block):
        try:
            r = client.chat.completions.create(model=args.model,
                messages=[{"role":"user","content":WIN_PROMPT.format(block=block)}],
                max_tokens=200, temperature=0.0, extra_body={"reasoning":{"enabled":False}})
            m = r.choices[0].message
            return m.content or getattr(m, "reasoning_content", None) or ""
        except Exception as ex:
            print(f"   err {ex}", flush=True); return ""

    results = {}
    import time as _t
    for B in bufs:
        ex0, f2 = [], []; npred = 0; ngold = 0; ncall = 0; _t0 = _t.perf_counter()
        for mi, mid in enumerate(mids):
            d = json.load(open(TOPIC/f"{mid}.json")); turns = d["turns"]; bt = d["bnd_top"]; n = len(turns)
            t0 = turns[0]["start"]
            gold = [i for i, b in enumerate(bt) if b == 1]
            pred = set()
            # B초 비중첩 윈도우로 분할, 각 윈도우만 LLM 에 줌
            w_start = t0; i = 0
            while i < n:
                w = [k for k in range(n) if w_start <= turns[k]["start"] < w_start + B]
                if not w:
                    w_start += B; continue
                valid = set(w[1:])           # 윈도우 첫 turn 은 경계 후보 아님
                if len(w) >= 2:
                    block = "\n".join(f"[{k}] ({turns[k]['speaker']}) {turns[k]['text']}" for k in w)
                    pred.update(parse_bounds(ask(block), valid)); ncall += 1
                i = w[-1] + 1; w_start += B
            pred = sorted(pred)
            yt = [1 if k in set(gold) else 0 for k in range(n)]; yp = [1 if k in set(pred) else 0 for k in range(n)]
            ex0.append(f1_score(yt, yp, zero_division=0)); f2.append(tol_f1(gold, pred, 2))
            npred += len(pred); ngold += len(gold)
            print(f"    [B={B}s {mi+1}/{len(mids)}] {mid}: ±2={tol_f1(gold,pred,2):.2f} "
                  f"({_t.perf_counter()-_t0:.0f}s, {ncall} calls)", flush=True)
        results[B] = (np.mean(ex0), np.mean(f2), npred, ngold, ncall)
        print(f"  buffer={B}s: exactF1={np.mean(ex0):.3f} ±2F1={np.mean(f2):.3f} pred={npred}/{ngold} ({ncall} LLM calls)", flush=True)

    print(f"\n=== 실험2 버퍼 곡선 ({args.model}, {len(mids)}미팅) ===")
    print(f"  {'buffer':>8} {'exactF1':>8} {'±2F1':>7}")
    for B in bufs:
        e, f, *_ = results[B]
        print(f"  {B:>6}s {e:>8.3f} {f:>7.3f}")
    print(f"  {'full(∞)':>8}     ~0.034   ~0.526   (전체 transcript)")
    print(f"  {'Hi-OnTop(0)':>8}     ~0.03    ~0.19    (0버퍼 embedding)")


if __name__ == "__main__":
    main()
