#!/usr/bin/env python3
"""AMI LLM 분절 — Crts API로 LLM에게 화제 경계 turn 을 직접 묻고 exact F1 측정.
embedding(Hi-OnTop) 대비 "LLM이 AMI 정밀 경계를 더 잘 찍나" 검증."""
from __future__ import annotations
import json, os, re, sys, time, argparse
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO/".env")
TOPIC = REPO/"data"/"ami"/"topic"

PROMPT = """You segment a meeting transcript into topic segments.
Below are numbered utterances from one meeting. A topic boundary is an utterance
that STARTS a new discussion topic (e.g. moving from introductions to project goals,
or from budget to design). Utterance 0 is never a boundary.

Return ONLY a JSON array of the utterance numbers that start a new topic, e.g. [8, 14, 54].
Typical meetings have only 5-12 topic boundaries. Do not over-segment.

Transcript:
{transcript}

JSON array of boundary utterance numbers:"""


def parse_bounds(txt, n):
    ms = re.findall(r"\[[\d,\s]*\]", txt or "")
    for cand in reversed(ms):          # 마지막 배열 = 최종 답 (reasoning fallback 대비)
        try:
            arr = json.loads(cand)
            if arr:
                return sorted({int(x) for x in arr if 0 < int(x) < n})
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
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])

    mids = sorted(m["meeting"] for m in json.load(open(TOPIC/"manifest.json")))[:args.limit]
    rows = []; F0=[]; F2=[]; npred=0; ngold=0
    for k, mid in enumerate(mids):
        d = json.load(open(TOPIC/f"{mid}.json")); turns = d["turns"]; bt = d["bnd_top"]; n = len(turns)
        gold = [i for i, b in enumerate(bt) if b == 1]
        transcript = "\n".join(f"[{i}] ({t['speaker']}) {t['text']}" for i, t in enumerate(turns))
        try:
            r = client.chat.completions.create(model=args.model,
                messages=[{"role": "user", "content": PROMPT.format(transcript=transcript)}],
                max_tokens=400, temperature=0.0,
                extra_body={"reasoning": {"enabled": False}})   # no-thinking
            msg = r.choices[0].message
            content = msg.content or getattr(msg, "reasoning_content", None) or ""
            pred = parse_bounds(content, n)
        except Exception as ex:
            print(f"  {mid}: ERROR {ex}", flush=True); pred = []
        f0 = f1_score([1 if i in set(gold) else 0 for i in range(n)],
                      [1 if i in set(pred) else 0 for i in range(n)], zero_division=0)
        f2 = tol_f1(gold, pred, 2)
        F0.append(f0); F2.append(f2); npred += len(pred); ngold += len(gold)
        rows.append((mid, len(gold), len(pred), f0, f2))
        print(f"  [{k+1}/{len(mids)}] {mid}: gold={len(gold)} pred={len(pred)} exF1={f0:.3f} ±2F1={f2:.3f}", flush=True)
    print(f"\n=== LLM 분절 ({args.model}), {len(mids)}미팅 ===")
    print(f"  exact F1 = {np.mean(F0):.3f} | ±2 F1 = {np.mean(F2):.3f} | pred {npred} / gold {ngold}")
    print(f"  (비교: embedding geometry-merge exact≈0.03, ±2≈0.19)")


if __name__ == "__main__":
    main()
