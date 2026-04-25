#!/usr/bin/env python3
"""Phase 4: judge LongMemEval hypotheses using our vLLM (Qwen) endpoint.

LongMemEval 공식 ``evaluate_qa.py``는 ``model_zoo``가 hardcoded이고
base_url이 fix되어 있어 우리 vLLM endpoint를 못 씀. 또한 외부 레포 수정 금지
(CLAUDE.md). 따라서 자체 judge를 작성한다.

**Prompt templates 6개는 LongMemEval 원본을 그대로 인용** (MIT License,
Copyright 2024 Di Wu — ``benchmarks/LongMemEval/LICENSE``).

Usage:
    uv run python scripts/judge_longmemeval.py outputs/phase-4-sanity-sliding.jsonl \\
        --ref benchmarks/LongMemEval/data/longmemeval_oracle.json \\
        --judge-model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.llm import OpenAIChatLLM  # noqa: E402


# --- Prompt templates (LongMemEval evaluate_qa.py, MIT License) ----------

_T_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {q}\n\nCorrect Answer: {a}\n\n"
    "Model Response: {r}\n\nIs the model response correct? Answer yes or no only."
)

_T_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/"
    "months, etc., and the model makes off-by-one errors (e.g., predicting 19 "
    "days when the answer is 18), the model's response is still correct. \n\n"
    "Question: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_T_KU = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {q}\n\nCorrect Answer: "
    "{a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or "
    "no only."
)

_T_PREF = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {q}\n\n"
    "Rubric: {a}\n\nModel Response: {r}\n\nIs the model response correct? Answer "
    "yes or no only."
)

_T_ABSTAIN = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {q}\n\nExplanation: {a}\n\nModel Response: {r}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes "
    "or no only."
)


def get_prompt(qtype: str, q: str, a: str, r: str, abstention: bool) -> str:
    if abstention:
        return _T_ABSTAIN.format(q=q, a=a, r=r)
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return _T_DEFAULT.format(q=q, a=a, r=r)
    if qtype == "temporal-reasoning":
        return _T_TEMPORAL.format(q=q, a=a, r=r)
    if qtype == "knowledge-update":
        return _T_KU.format(q=q, a=a, r=r)
    if qtype == "single-session-preference":
        return _T_PREF.format(q=q, a=a, r=r)
    raise ValueError(f"unknown qtype: {qtype}")


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_yes_no(raw: str) -> bool:
    """Return True iff judge's first non-think word is 'yes' (case-insensitive)."""
    cleaned = _THINK_RE.sub("", raw).strip().lower()
    # Take first whitespace-delimited token
    first = cleaned.split()[0] if cleaned.split() else ""
    return first.startswith("yes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hyp_file", help="Hypothesis jsonl from run_longmemeval.py")
    parser.add_argument("--ref", required=True, help="LongMemEval reference json")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Judge response is just yes/no — keep tight.")
    parser.add_argument("--output", default=None,
                        help="Default: <hyp_file>.judged.jsonl")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("[fatal] OPENAI_API_KEY missing"); sys.exit(1)
    print(f"[env] base_url={os.environ.get('OPENAI_BASE_URL', '(default)')}")
    print(f"[args] judge_model={args.judge_model}  hyp={args.hyp_file}  ref={args.ref}")

    refs = json.loads(Path(args.ref).read_text())
    qid2ref = {e["question_id"]: e for e in refs}

    hyps: list[dict] = []
    for line in Path(args.hyp_file).read_text().splitlines():
        line = line.strip()
        if line:
            hyps.append(json.loads(line))

    out_path = Path(args.output or args.hyp_file + ".judged.jsonl")

    llm = OpenAIChatLLM()
    qtype2hits: dict[str, list[int]] = defaultdict(list)
    abs_hits: list[int] = []
    rows: list[dict] = []

    for i, h in enumerate(hyps, 1):
        qid = h["question_id"]
        if qid not in qid2ref:
            print(f"  [{i}] skip {qid} (not in ref)")
            continue
        ref = qid2ref[qid]
        qtype = ref["question_type"]
        is_abs = "_abs" in qid
        prompt = get_prompt(qtype, ref["question"], ref["answer"], h["hypothesis"], is_abs)
        raw = llm.chat(
            [{"role": "user", "content": prompt}],
            model=args.judge_model,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        label = parse_yes_no(raw)
        bucket = "abstention" if is_abs else qtype
        qtype2hits[bucket].append(int(label))
        rows.append({**h, "question_type": qtype, "abstention": is_abs,
                      "judge_raw": raw, "label": label})
        print(f"  [{i}/{len(hyps)}] {qid} ({bucket}): {'✓' if label else '✗'}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n=== Accuracy ===")
    total = sum(sum(v) for v in qtype2hits.values())
    n = sum(len(v) for v in qtype2hits.values())
    print(f"  overall: {total/n:.3f} ({total}/{n})")
    for k in sorted(qtype2hits):
        v = qtype2hits[k]
        print(f"  {k:30s}: {np.mean(v):.3f} ({sum(v)}/{len(v)})")
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
