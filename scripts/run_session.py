#!/usr/bin/env python3
"""Phase 4-Re session wrapper: run multiple methods with shared config.

새 인프라 (``scripts/run_experiment.py``) 위에서 4 method 비교 (혹은 임의의
method/HP 조합 sweep) 한 줄 실행. 옛 ``run_phase4_all.py`` 의 대체.

각 method 는 별도 experiment 로 실행되며 (lost 측정 회피, resume 가능), 모두
같은 ``session.json`` 의 ``common_config`` 를 공유한다 (SKILL §2.5).

Examples:
    # Sanity 30 × 4 method (5분~10분)
    uv run python scripts/run_session.py \\
        --session-id 20260427_phase4-re_sanity \\
        --no-thinking --stratify --limit 30 \\
        --questions-per-round 10 --workers 8

    # Full 500 × 4 method (1~2시간)
    uv run python scripts/run_session.py \\
        --session-id 20260427_phase4-re_full \\
        --no-thinking --questions-per-round 50 --workers 8

    # 부분 method 만
    uv run python scripts/run_session.py \\
        --session-id ... --methods rag hi-em ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.experiment import DEFAULT_RESULTS_ROOT, Session, save_session  # noqa: E402

QTYPES = [
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session-id", default=None,
                   help="Default: session_<UTC ts>")
    p.add_argument("--methods", nargs="+",
                   default=["sliding", "full", "rag", "hi-em"],
                   choices=["sliding", "full", "rag", "hi-em"])
    p.add_argument("--data",
                   default="benchmarks/LongMemEval/data/longmemeval_oracle.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--questions-per-round", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--skip-existing", action="store_true",
                   help="이미 completed 된 experiment 는 건너뛰기 (resume 와 별개).")
    args = p.parse_args()

    sid = args.session_id or (
        "session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    )
    results_root = Path(args.results_root)

    common_config: dict = {
        "data": args.data,
        "no_thinking": args.no_thinking,
        "questions_per_round": args.questions_per_round,
        "workers": args.workers,
        "stratify": args.stratify,
        "limit": args.limit,
    }
    session = Session(
        session_id=sid,
        purpose=f"Method comparison on {Path(args.data).stem}",
        common_config=common_config,
        tags=["phase-4-re", "comparison"],
    )

    # --- Run each method as a separate experiment ----------------------
    exp_ids: list[str] = []
    for m in args.methods:
        eid = f"{sid}_{m}"
        exp_dir = results_root / "experiments" / eid

        if args.skip_existing and (exp_dir / "completed.json").exists():
            print(f"\n=== SKIP {m} ({eid}) — already completed ===")
            session.add_experiment(eid, overrides={"method": m})
            exp_ids.append(eid)
            continue

        cmd = [
            "uv", "run", "python", "scripts/run_experiment.py",
            "--method", m,
            "--data", args.data,
            "--questions-per-round", str(args.questions_per_round),
            "--workers", str(args.workers),
            "--exp-id", eid,
            "--session", sid,
            "--results-root", str(results_root),
        ]
        if args.no_thinking:
            cmd.append("--no-thinking")
        if args.stratify:
            cmd.append("--stratify")
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        print(f"\n=== RUN {m} ({eid}) ===")
        print(" ".join(cmd))
        ret = subprocess.run(cmd, cwd=REPO_ROOT)
        if ret.returncode != 0:
            print(f"[fatal] {m} failed (exit {ret.returncode}). "
                  "Re-run this command to resume from the last completed round.")
            sys.exit(1)

        session.add_experiment(eid, overrides={"method": m})
        exp_ids.append(eid)

    # --- Save session.json ---------------------------------------------
    save_session(session, root=results_root)

    # --- Comparison table -----------------------------------------------
    rows: list[tuple[str, dict]] = []
    for m, eid in zip(args.methods, exp_ids):
        sp = results_root / "experiments" / eid / "summary.json"
        if not sp.exists():
            rows.append((m, {}))
            continue
        rows.append((m, json.loads(sp.read_text())))

    # stdout
    print(f"\n=== Session Summary: {sid} ===")
    header = f"{'Method':<10} {'Overall':>8} " + " ".join(
        f"{qt[:5]:>7}" for qt in QTYPES
    )
    print(header)
    print("-" * len(header))
    for m, s in rows:
        if not s:
            print(f"{m:<10} (no summary)")
            continue
        ov = s.get("accuracy_overall", 0.0)
        cells = [f"{ov:.3f}"]
        for qt in QTYPES:
            v = s.get(f"accuracy_by_qtype/{qt}")
            cells.append(f"{v:.2f}" if v is not None else "  -  ")
        print(f"{m:<10} {cells[0]:>8} " + " ".join(c.rjust(7) for c in cells[1:]))

    # comparison.md (markdown table)
    md_lines = [
        f"# Session {sid}",
        "",
        f"- data: `{args.data}`",
        f"- no_thinking: {args.no_thinking}",
        f"- questions_per_round: {args.questions_per_round}",
        f"- limit: {args.limit}",
        f"- methods: {', '.join(args.methods)}",
        "",
        "## Comparison",
        "",
        "| Method | Overall | " + " | ".join(QTYPES) + " |",
        "| --- | --- | " + " | ".join("---" for _ in QTYPES) + " |",
    ]
    for m, s in rows:
        if not s:
            continue
        ov = s.get("accuracy_overall", 0.0)
        cells = [m, f"{ov:.3f}"]
        for qt in QTYPES:
            v = s.get(f"accuracy_by_qtype/{qt}")
            cells.append(f"{v:.2f}" if v is not None else "—")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_lines.append("")
    md_lines.append("## Per-experiment artifacts")
    for m, eid in zip(args.methods, exp_ids):
        md_lines.append(f"- `{m}`: `results/experiments/{eid}/`")

    cmp_path = results_root / "sessions" / sid / "comparison.md"
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    cmp_path.write_text("\n".join(md_lines))
    print(f"\n[done] session: {results_root / 'sessions' / sid}")
    print(f"       comparison: {cmp_path}")


if __name__ == "__main__":
    main()
