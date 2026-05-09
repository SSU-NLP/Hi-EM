#!/usr/bin/env python3
"""Aggregate α×λ×cos sweep + RAG variants into a single markdown table.

Outputs:
    outputs/sweeps/2026-05-05_locomo_alpha_lambda_cos/summary_table.md
    outputs/sweeps/2026-05-05_locomo_alpha_lambda_cos/summary_table.csv

Run any time during/after the sweep. Skips configs without summary.json.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, median, pvariance

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "results" / "experiments"
SWEEP = REPO / "outputs" / "sweeps/2026-05-05_locomo_alpha_lambda_cos"
LOG = SWEEP / "run.log"
RAG_LOG = SWEEP / "run_rag.log"
V321_LOG = SWEEP / "run_v321.log"
TOP20_FILE = SWEEP / "top20_for_v321.txt"
V321_BETAS = ("0.25", "0.5", "1.0")


def parse_runtimes(log_path: Path) -> dict:
    """Extract (alpha, lmda[, cos]) → runtime_sec from sweep log."""
    out: dict = {}
    if not log_path.exists():
        return out
    text = log_path.read_text()
    blocks = re.split(r"\n=== Config:", text)
    for b in blocks[1:]:
        head = b.splitlines()[0]
        m_full = re.search(r"method=hi-em-full-v1\b.*alpha=(\d+) lmda=(\d+) sigma", head)
        # v3.2.1 must be checked BEFORE v3.1 (string prefix overlap)
        m_v321 = re.search(r"method=hi-em-full-v3\.2\.1\b.*alpha=(\d+) lmda=(\d+) cos=([\d.]+) beta=([\d.]+)", head)
        m_v31 = re.search(r"method=hi-em-full-v3\.1\.?\d*\b.*alpha=(\d+) lmda=(\d+) cos=([\d.]+)", head)
        m_rag = re.search(r"method=(rag(?:-\w+)?)\b", head)
        rt_match = re.search(r"runtime=([\d.]+)s", b)
        if not rt_match:
            continue
        rt = float(rt_match.group(1))
        if m_full:
            out[("full", m_full.group(1), m_full.group(2))] = rt
        elif m_v321:
            out[("v3.2.1", m_v321.group(1), m_v321.group(2), m_v321.group(3), m_v321.group(4))] = rt
        elif m_v31:
            out[("v3.1.1", m_v31.group(1), m_v31.group(2), m_v31.group(3))] = rt
        elif m_rag:
            out[("rag", m_rag.group(1))] = rt
    return out


def stm_stats_from_jsonl(p: Path) -> tuple:
    """(t1μ, t2μ, t3μ, t1max, t2max, t1var, n_rounds)."""
    if not p.exists():
        return (0.0, 0.0, 0.0, 0, 0, 0.0, 0)
    t1, t2, t3 = [], [], []
    for line in p.open():
        r = json.loads(line)
        ss = r["sorted_sizes"]
        if len(ss) >= 1:
            t1.append(ss[0])
        if len(ss) >= 2:
            t2.append(ss[1])
        if len(ss) >= 3:
            t3.append(ss[2])
    if not t1:
        return (0.0, 0.0, 0.0, 0, 0, 0.0, 0)
    return (
        mean(t1),
        mean(t2) if t2 else 0.0,
        mean(t3) if t3 else 0.0,
        max(t1),
        max(t2) if t2 else 0,
        pvariance(t1) if len(t1) > 1 else 0.0,
        len(t1),
    )


def build_row(method: str, alpha, lmda, cos_thr, exp_id: str, runtimes: dict,
              beta=None) -> dict | None:
    exp_dir = ROOT / exp_id
    sj = exp_dir / "summary.json"
    if not sj.exists():
        return None
    s = json.loads(sj.read_text())

    if method == "full":
        topk = SWEEP / f"stm_topk_hi-em-full-v1_a{alpha}_l{lmda}.rounds.jsonl"
        rt = runtimes.get(("full", str(alpha), str(lmda)))
    elif method == "v3.1.1":
        topk = SWEEP / f"stm_topk_hi-em-full-v3_1_a{alpha}_l{lmda}_c{cos_thr}.rounds.jsonl"
        rt = runtimes.get(("v3.1.1", str(alpha), str(lmda), str(cos_thr)))
    elif method == "v3.2.1":
        topk = SWEEP / f"stm_topk_v3_2_1_a{alpha}_l{lmda}_c{cos_thr}_b{beta}.rounds.jsonl"
        rt = runtimes.get(("v3.2.1", str(alpha), str(lmda), str(cos_thr), str(beta)))
    else:  # rag*
        topk = None
        rt = runtimes.get(("rag", method))

    t1m = t2m = t3m = 0.0
    t1mx = t2mx = 0
    t1var = 0.0
    if topk is not None:
        t1m, t2m, t3m, t1mx, t2mx, t1var, _ = stm_stats_from_jsonl(topk)

    gap = t1m - t2m
    ratio = (t1m / t2m) if t2m else float("inf")

    return {
        "method": method,
        "α": alpha if alpha is not None else "—",
        "λ": lmda if lmda is not None else "—",
        "cos": cos_thr if cos_thr is not None else "—",
        "β": beta if beta is not None else "—",
        "acc": s["accuracy_overall"],
        "err": s["error_rate"],
        "rt_s": rt,
        "gen_p50": s.get("gen_sec_p50"),
        "lat_p50": s.get("latency_sec_p50"),
        "t1μ": t1m,
        "t2μ": t2m,
        "t3μ": t3m,
        "gap": gap,
        "ratio": ratio,
        "t1max": t1mx,
        "t2max": t2mx,
        "t1var": t1var,
    }


def collect_rows(runtimes: dict) -> list[dict]:
    rows: list[dict] = []
    # Block 1: full × 4×4
    for alpha in (1, 10, 100, 1000):
        for lmda in (0, 1, 10, 100):
            exp_id = f"20260505_locomo_aL_a{alpha}_l{lmda}_hi-em-full-v1"
            r = build_row("full", alpha, lmda, None, exp_id, runtimes)
            if r:
                rows.append(r)
    # Block 2: v3.1 × 4×4×4
    for alpha in (1, 10, 100, 1000):
        for lmda in (0, 1, 10, 100):
            for cos_thr in ("0.3", "0.5", "0.7", "0.9"):
                exp_id = f"20260505_locomo_aL_a{alpha}_l{lmda}_c{cos_thr}_hi-em-full-v3_1_1"
                r = build_row("v3.1.1", alpha, lmda, cos_thr, exp_id, runtimes)
                if r:
                    rows.append(r)
    # Block 3: v3.2.1 top-20 × β
    if TOP20_FILE.exists():
        for line in TOP20_FILE.open():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cfg = line.split("#", 1)[0].strip()
            parts = cfg.split()
            if len(parts) < 3:
                continue
            alpha_s, lmda_s, cos_s = parts[0], parts[1], parts[2]
            for beta in V321_BETAS:
                exp_id = f"20260505_locomo_aL_top20_a{alpha_s}_l{lmda_s}_c{cos_s}_b{beta}_hi-em-full-v3_2_1"
                r = build_row("v3.2.1", int(alpha_s), int(lmda_s), cos_s, exp_id, runtimes, beta=beta)
                if r:
                    rows.append(r)

    # RAG variants
    for m in ("rag", "rag-summary", "rag-observation"):
        exp_id = f"20260505_locomo_aL_{m.replace('-', '_')}"
        r = build_row(m, None, None, None, exp_id, runtimes)
        if r:
            rows.append(r)
    return rows


def annotate(rows: list[dict]) -> None:
    """Add 비고 column based on outliers + structural flags."""
    # Per-block runtime medians + best/worst per method
    blocks = ("full", "v3.1.1", "v3.2.1")
    rt_med = {}
    best = {}
    worst = {}
    for m in blocks:
        rts = [r["rt_s"] for r in rows if r["method"] == m and r["rt_s"]]
        accs = [r["acc"] for r in rows if r["method"] == m]
        rt_med[m] = median(rts) if rts else 0.0
        best[m] = max(accs) if accs else None
        worst[m] = min(accs) if accs else None

    for r in rows:
        notes = []
        # β=1.0 in v3.2.1 reduces sub-linear (C+1)^β to v3.1.1 raw sCRP exactly;
        # acc difference vs source v3.1.1 is therefore LLM temperature noise only.
        if r["method"] == "v3.2.1" and str(r.get("β", "")) == "1.0":
            notes.append("v3.1.1과 알고리즘 동일 — sanity-50 LLM noise 측정용 control")
        if r["method"] in rt_med and rt_med[r["method"]] and r["rt_s"] and r["rt_s"] > 2 * rt_med[r["method"]]:
            notes.append(f"느림 ({r['rt_s']/60:.1f}min, mega-topic 추정)")
        if r["method"] in {"full", "v3.1.1", "v3.2.1"}:
            if r["t2μ"] == 0:
                notes.append("STM single-topic")
            elif r["ratio"] != float("inf") and r["ratio"] > 50:
                notes.append("topic skew 극단")
        if r["err"] and r["err"] > 0:
            notes.append(f"err_rate={r['err']:.2f}")
        if r["method"] in best and best[r["method"]] is not None and abs(r["acc"] - best[r["method"]]) < 1e-9:
            notes.append("⭐ best (block)")
        if r["method"] in worst and worst[r["method"]] is not None and abs(r["acc"] - worst[r["method"]]) < 1e-9:
            notes.append("worst (block)")
        r["비고"] = "; ".join(notes) or "-"


def fmt_cell(key: str, v) -> str:
    if v is None:
        return "?"
    if v == float("inf"):
        return "∞"
    if key in {"acc", "err"}:
        return f"{v:.4f}"
    if key in {"gen_p50", "lat_p50"}:
        return f"{v:.2f}"
    if key == "rt_s":
        return f"{v:.1f}"
    if key in {"t1μ", "t2μ", "t3μ", "gap"}:
        return f"{v:.1f}" if isinstance(v, float) else str(v)
    if key == "ratio":
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    if key == "t1var":
        return f"{v:.0f}" if isinstance(v, float) else str(v)
    return str(v)


def write_outputs(rows: list[dict]) -> None:
    SWEEP.mkdir(parents=True, exist_ok=True)
    md_path = SWEEP / "summary_table.md"
    csv_path = SWEEP / "summary_table.csv"

    cols = ["method", "α", "λ", "cos", "β", "acc", "rt_s", "gen_p50", "lat_p50",
            "t1μ", "t2μ", "t3μ", "gap", "ratio", "t1max", "t2max", "t1var", "비고"]

    # Markdown
    md_lines = [
        f"# LoCoMo sanity-50 α×λ×cos sweep + RAG (Crts qwen/qwen3.5-9b)",
        "",
        f"Generated: {Path(__file__).stem} from {SWEEP.name}",
        f"Configs: {len(rows)} rows",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        cells = [fmt_cell(c, r.get(c)) for c in cols]
        md_lines.append("| " + " | ".join(cells) + " |")

    # Block summaries
    md_lines += ["", "## Block summaries", ""]
    for block in ("full", "v3.1.1", "rag", "rag-summary", "rag-observation"):
        sub = [r for r in rows if r["method"] == block]
        if not sub:
            continue
        accs = [r["acc"] for r in sub]
        md_lines.append(f"- **{block}** (n={len(sub)}): "
                        f"mean={sum(accs)/len(accs):.4f}, "
                        f"median={median(accs):.4f}, "
                        f"min={min(accs):.4f}, max={max(accs):.4f}")
    md_path.write_text("\n".join(md_lines) + "\n")

    # CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print(f"rows: {len(rows)}")


def main() -> int:
    runtimes = {
        **parse_runtimes(LOG),
        **parse_runtimes(RAG_LOG),
        **parse_runtimes(V321_LOG),
    }
    rows = collect_rows(runtimes)
    annotate(rows)
    write_outputs(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
