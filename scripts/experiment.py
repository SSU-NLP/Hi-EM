#!/usr/bin/env python3
"""Unified experiment harness — one entry point for any LoCoMo / LongMemEval experiment (sweep, ablation, compare).

Replaces the per-experiment shell scripts. Each experiment:
    - lives under ``outputs/experiments/<name>/``
    - has one subdir per method config (resumable: skipped if exit_code.txt = 0)
    - captures STM top-K stats for hi-em-full-* methods
    - emits ``REPORT.md`` with every available metric (acc + qtype + T1/T2/T3
      + T1max/T2max/T1var + n_topics_avg + gen_p50 + wall_s) + RAG/sliding/full
      baselines if present.

Usage:
    uv run python scripts/experiment.py \\
        --name 2026-05-09_my_ablation \\
        --benchmark locomo \\
        --data benchmarks/locomo/data/locomo10.json \\
        --limit 100 --stratify \\
        --workers 100 --no-thinking \\
        --method "hi-em-full-v1:v1_default" \\
        --method "hi-em-full-v3.3.2:v332_best@cos_threshold=0.9,alpha=100,beta=0.25,pe_threshold=0.5,rnn_train_steps=3" \\
        --method rag --method rag-summary --method sliding --method full

Method spec grammar:
    method[:label][@k1=v1,k2=v2,...]
        method  : run_experiment.py --method value (required)
        label   : output subdir name (default = method, '.' → 'p', '-' → '_')
        k=v     : forwarded as --k v to run_experiment.py (HP overrides)

Resume:
    Re-running the same --name skips any method whose <run_dir>/exit_code.txt
    contains "0". Delete that file (or the whole subdir) to force re-run.

Aggregation only:
    --aggregate-only skips running and just regenerates REPORT.md from existing
    subdir state. Useful after manual edits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
experiments_root = REPO / "outputs" / "experiments"

HI_EM_PREFIXES = ("hi-em-full-v",)


# ---------------------------------------------------------------- spec parse


def parse_method_spec(spec: str) -> tuple[str, str, dict[str, str]]:
    """`method[:label][@k=v,k=v]` → (method, label, overrides)."""
    overrides: dict[str, str] = {}
    head = spec
    if "@" in spec:
        head, kvs = spec.split("@", 1)
        for tok in kvs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise ValueError(f"bad HP override token {tok!r} (need k=v)")
            k, v = tok.split("=", 1)
            overrides[k.strip()] = v.strip()
    if ":" in head:
        method, label = head.split(":", 1)
    else:
        method, label = head, head
    method = method.strip()
    label = label.strip()
    if not method:
        raise ValueError(f"empty method in spec {spec!r}")
    safe_label = label.replace(".", "p").replace("-", "_")
    return method, safe_label, overrides


def is_hi_em_method(method: str) -> bool:
    return method.startswith(HI_EM_PREFIXES)


# ---------------------------------------------------------------- runner


def build_run_cmd(
    *,
    method: str,
    run_dir: Path,
    overrides: dict[str, str],
    benchmark: str,
    data: str,
    exp_id: str,
    common_flags: list[str],
) -> list[str]:
    cmd = [
        "uv", "run", "python", "scripts/run_experiment.py",
        "--method", method,
        "--benchmark", benchmark,
        "--data", data,
        "--exp-id", exp_id,
        "--results-root", str(run_dir / "results"),
    ] + common_flags
    for k, v in overrides.items():
        cli_flag = "--" + k.replace("_", "-")
        cmd += [cli_flag, str(v)]
    return cmd


def run_method(
    *,
    experiment_dir: Path,
    method: str,
    label: str,
    overrides: dict[str, str],
    benchmark: str,
    data: str,
    common_flags: list[str],
    experiment_name: str,
) -> int:
    run_dir = experiment_dir / label
    exit_path = run_dir / "exit_code.txt"
    if exit_path.exists() and exit_path.read_text().strip() == "0":
        print(f"[skip] {label} — already done")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    topk_path = run_dir / "stm_topk.json"
    rounds_jsonl = topk_path.with_suffix(".rounds.jsonl")
    for p in (topk_path, rounds_jsonl):
        if p.exists():
            p.unlink()

    exp_id = f"{experiment_name}_{label}"
    cmd = build_run_cmd(
        method=method,
        run_dir=run_dir,
        overrides=overrides,
        benchmark=benchmark,
        data=data,
        exp_id=exp_id,
        common_flags=common_flags,
    )

    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    # wandb default = online (project convention 2026-05-08~). To opt out for a
    # specific run, prefix the harness invocation with WANDB_MODE=disabled.
    if is_hi_em_method(method):
        env["HIEM_STM_TOPK_STATS_PATH"] = str(topk_path)

    with log_path.open("w") as logf:
        logf.write(f"=== START {datetime.now().isoformat()} ===\n")
        logf.write(f"method={method}\nlabel={label}\noverrides={overrides}\n")
        logf.write(f"exp_id={exp_id}\nrun_dir={run_dir}\n")
        logf.write("cmd: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        rc = proc.returncode
        logf.write(f"\n[exit] {exp_id} rc={rc}\n")
        logf.write(f"=== END {datetime.now().isoformat()} ===\n")
    exit_path.write_text(str(rc))

    # On success, run the post-hoc evidence→topic analyzer. No-op for
    # non-hi-em methods (the analyzer silently skips when working_state
    # is absent). Failure is non-fatal — REPORT will just miss the
    # topics/q columns for this run.
    if rc == 0:
        try:
            subprocess.run(
                ["uv", "run", "python", "scripts/analyze_evidence_topics.py",
                 "--run", str(run_dir)],
                cwd=REPO, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[warn] analyze_evidence_topics failed for {label}: {e}")
    return rc


# ---------------------------------------------------------------- aggregator


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.strip())
    except Exception:
        return None


def wall_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    starts: list[datetime] = []
    ends: list[datetime] = []
    for line in log_path.open():
        m = re.match(r"=== START (\S+) ===", line)
        if m and (t := _parse_iso(m.group(1))):
            starts.append(t)
        m = re.match(r"=== END (\S+) ===", line)
        if m and (t := _parse_iso(m.group(1))):
            ends.append(t)
    if starts and ends:
        return (ends[-1] - starts[0]).total_seconds()
    return None


def load_summary(run_dir: Path) -> dict:
    sp = list((run_dir / "results" / "experiments").glob("*/summary.json"))
    return json.loads(sp[0].read_text()) if sp else {}


def load_config(run_dir: Path) -> dict:
    ep = list((run_dir / "results" / "experiments").glob("*/experiment.json"))
    if not ep:
        return {}
    try:
        return json.loads(ep[0].read_text()).get("config", {})
    except Exception:
        return {}


def _fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def format_notes(method: str, config: dict, overrides: dict[str, str]) -> str:
    """One-line HP summary per method. Only HPs the method actually consumes."""
    parts: list[str] = []
    seg = config.get("segmenter", {})
    mw = config.get("memory_window", {})
    ev = config.get("evaluation", {})

    if method.startswith("hi-em-full-v"):
        seg_keys = ["alpha", "lmda", "cos_threshold", "beta"]
        if method >= "hi-em-full-v3.3":
            seg_keys.append("rnn_train_steps")
        seg_str = ", ".join(f"{k.replace('cos_threshold','cos').replace('alpha','α').replace('lmda','λ').replace('beta','β')}={_fmt_num(seg[k])}"
                            for k in seg_keys if k in seg)
        mem_str = f"k_top={mw.get('k_topics','?')}, k_turn={mw.get('k_turns_per_topic','?')}"
        parts.append(f"seg: {seg_str} · mem: {mem_str}")
    elif method.startswith("rag"):
        parts.append(f"rag_k={ev.get('rag_k','?')}")
    elif method == "sliding":
        parts.append(f"sliding_k={ev.get('sliding_k','?')}")
    elif method == "full":
        parts.append("(full context)")

    if overrides:
        ov_str = ", ".join(f"{k}={v}" for k, v in overrides.items())
        parts.append(f"override: {ov_str}")
    return " · ".join(parts) if parts else "—"


def load_topk(run_dir: Path) -> dict:
    tk = run_dir / "stm_topk.json"
    return json.loads(tk.read_text()) if tk.exists() else {}


def load_evidence_topic_summary(run_dir: Path) -> dict:
    p = run_dir / "evidence_topic_summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def n_topics_avg(run_dir: Path) -> float:
    rj = run_dir / "stm_topk.rounds.jsonl"
    if not rj.exists():
        return 0.0
    n: list[int] = []
    for line in rj.open():
        if line.strip():
            try:
                n.append(json.loads(line).get("n_stm_topics", 0))
            except Exception:
                pass
    return sum(n) / len(n) if n else 0.0


def fmt(v, dp: int = 3) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{dp}f}"
    except Exception:
        return str(v)


def fmt_wall(sec) -> str:
    """Format wall-clock seconds, dropping leading zero units."""
    if sec is None or sec == "":
        return "-"
    try:
        s = int(round(float(sec)))
    except Exception:
        return str(sec)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def parse_overrides_from_log(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    m = re.search(r"^overrides=({.*})\s*$", text, re.MULTILINE)
    if not m:
        return {}
    try:
        return eval(m.group(1), {"__builtins__": {}}, {})
    except Exception:
        return {}


def write_report(experiment_dir: Path, experiment_name: str) -> None:
    out = experiment_dir / "REPORT.md"
    rows: list[tuple[Path, dict]] = []
    for d in sorted(experiment_dir.iterdir()):
        if not d.is_dir():
            continue
        ec = d / "exit_code.txt"
        if not ec.exists():
            continue
        rows.append((d, parse_overrides_from_log(d / "run.log")))

    header = (
        "| method | notes | accuracy_overall | "
        "multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | "
        "H@k | R@k | R-multi-hop@k | P@k | "
        "dormant_ev_rate | top_ev_promoted | "
        "n_dormant_top | dormant_top_n_ev | dormant_top_share | "
        "mh_topics/q | sh_topics/q | temp_topics/q | od_topics/q | "
        "T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | "
        "gen_p50(s) | wall |"
    )
    sep = "|" + "---|" * 30

    # n_questions is identical across runs in one sweep (same data + limit/stratify).
    # Pull it out of the per-row table and report once in the header.
    n_set: set[int] = set()
    for d, _ in rows:
        nq = load_summary(d).get("n_questions")
        if isinstance(nq, int):
            n_set.add(nq)
    n_str = str(next(iter(n_set))) if len(n_set) == 1 else (
        "/".join(str(n) for n in sorted(n_set)) if n_set else "?"
    )

    lines: list[str] = []
    lines.append(f"# Sweep: {experiment_name}\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Path: `{experiment_dir.relative_to(REPO)}/`\n")
    lines.append(f"n_questions: **{n_str}** (uniform across runs)\n")
    lines.append("Columns: H@k/R@k/P@k — retrieval hit/recall/precision against "
                 "LoCoMo answer-evidence turn ids; R-multi-hop@k — strict "
                 "all-evidence-included rate on multi-hop questions only; "
                 "{mh,sh,temp,od}_topics/q — mean number of distinct topics that "
                 "evidence turns of a question landed in, restricted to questions "
                 "with n_evidence≥2 (smaller = segmenter co-locates evidence; "
                 "computed by scripts/analyze_evidence_topics.py); "
                 "T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; "
                 "T1var — variance of top-1 across rounds; "
                 "STM_n_topics — mean STM topic count per round; "
                 "gen_p50 — per-question generation latency p50; "
                 "wall — total run wall-clock (h/m/s); "
                 "notes — HPs the method actually consumes (incl. overrides).\n")
    lines.append(header)
    lines.append(sep)

    for d, ov in rows:
        # method name from log header
        log = d / "run.log"
        method = "-"
        if log.exists():
            for line in log.open():
                if line.startswith("method="):
                    method = line.split("=", 1)[1].strip()
                    break
        s = load_summary(d)
        tk = load_topk(d)
        cfg = load_config(d)
        t1 = tk.get("top1_per_round", {})
        t2 = tk.get("top2_per_round", {})
        t3 = tk.get("top3_per_round", {})
        acc = s.get("accuracy_overall", "-")
        mh = s.get("accuracy_by_qtype/multi-hop", "-")
        sh = s.get("accuracy_by_qtype/single-hop", "-")
        tr = s.get("accuracy_by_qtype/temporal-reasoning", "-")
        adv = s.get("accuracy_by_qtype/adversarial", "-")
        od = s.get("accuracy_by_qtype/open-domain", "-")
        gen_p50 = s.get("gen_sec_p50", "-")
        wall = wall_seconds(log)
        # Retrieval metrics (added 2026-05-09; older runs may lack them).
        hk = s.get("H@k", "-")
        rk = s.get("R@k", "-")
        rmhk = s.get("R-multi-hop@k", "-")
        pk = s.get("P@k", "-")
        # Dormant evidence audit (added 2026-05-10; hi-em only, older runs lack).
        # 2026-05-11 추가: dormant 안 집중도 (top_share) — "한 dormant topic 에
        # evidence 가 몰려있는가" 직접 측정.
        dor_rate = s.get("dormant_ev_rate", "-")
        top_prom = s.get("top_ev_topic_promoted", "-")
        n_dor_top = s.get("n_dormant_topics_with_ev", "-")
        dor_top_n = s.get("dormant_top_topic_n_ev", "-")
        dor_top_share = s.get("dormant_top_topic_share", "-")
        # Evidence→topic concentration (added 2026-05-13; hi-em only).
        ets = load_evidence_topic_summary(d)
        mh_t = ets.get("mh_topics_per_q", "-")
        sh_t = ets.get("sh_topics_per_q", "-")
        temp_t = ets.get("temp_topics_per_q", "-")
        od_t = ets.get("od_topics_per_q", "-")
        notes = format_notes(method, cfg, ov)  # overrides folded into notes
        lines.append(
            f"| {method} ({d.name}) | {notes} | "
            f"{fmt(acc)} | {fmt(mh)} | {fmt(sh)} | {fmt(tr)} | {fmt(adv)} | {fmt(od)} | "
            f"{fmt(hk)} | {fmt(rk)} | {fmt(rmhk)} | {fmt(pk,4)} | "
            f"{fmt(dor_rate,3)} | {fmt(top_prom,3)} | "
            f"{fmt(n_dor_top,2)} | {fmt(dor_top_n,2)} | {fmt(dor_top_share,3)} | "
            f"{fmt(mh_t,2)} | {fmt(sh_t,2)} | {fmt(temp_t,2)} | {fmt(od_t,2)} | "
            f"{fmt(t1.get('mean'),1)} | {fmt(t2.get('mean'),1)} | {fmt(t3.get('mean'),1)} | "
            f"{t1.get('max','-')} | {t2.get('max','-')} | {fmt(t1.get('variance'),1)} | "
            f"{fmt(n_topics_avg(d),2)} | {fmt(gen_p50,2)} | {fmt_wall(wall)} |"
        )

    # best
    best: tuple[str, float] | None = None
    for d, _ov in rows:
        s = load_summary(d)
        acc = s.get("accuracy_overall")
        if acc is None:
            continue
        if best is None or float(acc) > best[1]:
            best = (d.name, float(acc))
    lines.append("")
    if best:
        lines.append(f"**best (acc)**: `{best[0]}` — {best[1]:.3f}")
    out.write_text("\n".join(lines) + "\n")
    print(f"report → {out}")


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", required=True,
                   help="Sweep name. Output goes to outputs/experiments/<name>/.")
    p.add_argument("--benchmark", default="locomo", choices=["locomo", "longmemeval"])
    p.add_argument("--data", required=True)
    p.add_argument("--method", action="append", required=False, default=[],
                   help="method[:label][@k=v,k=v]. Repeatable.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--workers", type=int, default=100)
    p.add_argument("--no-thinking", action="store_true", default=True,
                   help="Forwarded to run_experiment.py. Default ON.")
    p.add_argument("--questions-per-round", type=int, default=200)
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip running; just regenerate REPORT.md from current state.")
    p.add_argument("--extra", action="append", default=[],
                   help="Additional flag forwarded to run_experiment.py "
                        "(e.g. --extra='--tau 50'). Repeatable; quoted.")
    args = p.parse_args()

    experiment_dir = experiments_root / args.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        write_report(experiment_dir, args.name)
        return

    if not args.method:
        sys.exit("error: at least one --method spec required (or use --aggregate-only)")

    # Common flags forwarded to every run.
    common: list[str] = ["--workers", str(args.workers),
                         "--questions-per-round", str(args.questions_per_round),
                         "--no-token-count"]
    if args.no_thinking:
        common.append("--no-thinking")
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    if args.stratify:
        common.append("--stratify")
    for ex in args.extra:
        common += shlex.split(ex)

    print(f"[experiment] {args.name} → {experiment_dir}")
    print(f"[experiment] {len(args.method)} method specs")
    for spec in args.method:
        method, label, overrides = parse_method_spec(spec)
        print(f"\n[run] {method} :: {label} :: {overrides}")
        rc = run_method(
            experiment_dir=experiment_dir,
            method=method,
            label=label,
            overrides=overrides,
            benchmark=args.benchmark,
            data=args.data,
            common_flags=common,
            experiment_name=args.name,
        )
        if rc != 0:
            print(f"[warn] {label} exited rc={rc} — continuing")

    write_report(experiment_dir, args.name)


if __name__ == "__main__":
    main()
