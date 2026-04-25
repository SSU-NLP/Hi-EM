#!/usr/bin/env python3
"""Phase 4 평가 4 method × (run + judge) 한 번에 실행.

Sanity (30개 subset):
    uv run python scripts/run_phase4_all.py --limit 30

전체 (500):
    uv run python scripts/run_phase4_all.py

특정 method만:
    uv run python scripts/run_phase4_all.py --methods hi-em rag --limit 30

다른 데이터:
    uv run python scripts/run_phase4_all.py --data benchmarks/LongMemEval/data/longmemeval_s_cleaned.json --prefix phase-4-s
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
        print(f"[fatal] {label} failed", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="Default: 전체 (subset sanity 시 30 권장)")
    parser.add_argument("--data",
                        default="benchmarks/LongMemEval/data/longmemeval_oracle.json")
    parser.add_argument("--methods", nargs="+",
                        default=["sliding", "full", "rag", "hi-em"],
                        choices=["sliding", "full", "rag", "hi-em"])
    parser.add_argument("--prefix", default=None,
                        help="Default: phase-4-sanity (limit 있을 때) 또는 phase-4-full")
    parser.add_argument("--skip-judge", action="store_true",
                        help="run만 하고 judge는 건너뛰기 (수동 inspection 후 judge)")
    args = parser.parse_args()

    if args.prefix is None:
        args.prefix = "phase-4-sanity" if args.limit else "phase-4-full"

    for m in args.methods:
        out = f"outputs/{args.prefix}-{m}.jsonl"
        run_cmd = [
            "uv", "run", "python", "scripts/run_longmemeval.py",
            "--method", m, "--data", args.data, "--output", out,
        ]
        if args.limit:
            run_cmd += ["--limit", str(args.limit)]
        run(run_cmd, f"RUN  {m}")

        if not args.skip_judge:
            judge_cmd = [
                "uv", "run", "python", "scripts/judge_longmemeval.py", out,
                "--ref", args.data,
            ]
            run(judge_cmd, f"JUDGE {m}")

    print("\n=== ALL DONE ===")
    print(f"  hypothesis: outputs/{args.prefix}-*.jsonl")
    if not args.skip_judge:
        print(f"  judged:     outputs/{args.prefix}-*.jsonl.judged.jsonl")


if __name__ == "__main__":
    main()
