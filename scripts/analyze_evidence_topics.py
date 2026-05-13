"""Evidence-to-topic distribution analyzer (post-hoc, hi-em runs only).

Given a Hi-EM run directory (the per-run dir under
``outputs/experiments/<sweep>/<label>/``), join each LoCoMo question's
``evidence`` dia_ids against the run's per-conversation
``working_state/ltm/<conv>_<method>/<conv>.jsonl`` (which records the
``dia_id → topic_id`` assignment Hi-EM produced) and emit:

* ``<run_dir>/per_question_evidence_topics.csv`` — one row per question
  with the raw breakdown (evidence dia_ids, the topic each landed in,
  the resulting count-per-topic).
* ``<run_dir>/evidence_topic_summary.json`` — qtype-aggregated metrics
  for direct use in the sweep REPORT (mean ``n_topics_used`` on
  ``n_evidence >= 2`` questions, per qtype).

The summary is what gets surfaced in the sweep REPORT.md (columns
``mh_topics/q``, ``sh_topics/q``, ``temp_topics/q``, ``od_topics/q``).
Small values mean the segmenter co-locates evidence into fewer topics;
large values mean evidence is fragmented across many topics. This is a
direct segmentation-quality signal, separate from any importance / RAG
retrieval policy effect.

Non-hi-em runs (``rag*``, ``sliding``, ``full``) have no
``working_state/ltm`` and are silently skipped (exit 0) so the same hook
can run unconditionally after every method.

Usage:
    uv run python scripts/analyze_evidence_topics.py --run <run_dir>
        [--data benchmarks/locomo/data/locomo10.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "benchmarks" / "locomo" / "data" / "locomo10.json"

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
QTYPE_SHORT = {
    "multi-hop": "mh",
    "single-hop": "sh",
    "temporal": "temp",
    "open-domain": "od",
    "adversarial": "adv",
}


def find_ltm_dir(run_dir: Path) -> Path | None:
    """Return ``results/experiments/<exp_id>/working_state/ltm`` if it exists."""
    cands = list(run_dir.glob("results/experiments/*/working_state/ltm"))
    return cands[0] if cands else None


def build_dia_topic_map(ltm_dir: Path) -> dict[str, dict[str, int]]:
    """conv_id → {dia_id: topic_id}."""
    out: dict[str, dict[str, int]] = {}
    for d in ltm_dir.iterdir():
        if not d.is_dir():
            continue
        conv = d.name.split("_")[0]
        jl = d / f"{conv}.jsonl"
        if not jl.exists():
            continue
        m: dict[str, int] = {}
        with jl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dia = r.get("dia_id")
                tid = r.get("topic_id")
                if dia is not None and tid is not None:
                    m[dia] = int(tid)
        out[conv] = m
    return out


def flatten_evidence(evs) -> list[str]:
    if not evs:
        return []
    if isinstance(evs, str):
        return [evs]
    flat: list[str] = []
    for e in evs:
        if isinstance(e, list):
            flat.extend(e)
        else:
            flat.append(e)
    return flat


def analyze(run_dir: Path, data_path: Path) -> int:
    ltm_dir = find_ltm_dir(run_dir)
    if ltm_dir is None:
        # Non-hi-em run (rag/sliding/full) — silently no-op.
        return 0

    conv_map = build_dia_topic_map(ltm_dir)
    if not conv_map:
        print(f"[analyze-evidence] warning: empty ltm at {ltm_dir}", file=sys.stderr)
        return 0

    data = json.loads(data_path.read_text())

    csv_path = run_dir / "per_question_evidence_topics.csv"
    json_path = run_dir / "evidence_topic_summary.json"

    rows: list[dict] = []
    for s in data:
        conv = s.get("sample_id")
        topic_of = conv_map.get(conv, {})
        for qi, q in enumerate(s.get("qa", [])):
            flat = flatten_evidence(q.get("evidence"))
            ev_topics = [(e, topic_of.get(e)) for e in flat]
            found_topics = [t for _, t in ev_topics if t is not None]
            counts = Counter(found_topics)
            breakdown = ", ".join(
                f"t{t}:{c}" for t, c in sorted(counts.items(), key=lambda x: -x[1])
            )
            rows.append({
                "qid": f"{conv}_q{qi}",
                "cat": CAT.get(q.get("category"), str(q.get("category"))),
                "n_ev": len(flat),
                "n_ev_found": len(found_topics),
                "n_topics_used": len(counts),
                "evidence_dia_ids": "|".join(flat),
                "evidence_topics": "|".join(
                    str(t) if t is not None else "?" for _, t in ev_topics
                ),
                "topic_breakdown": breakdown,
            })

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "qid", "cat", "n_ev", "n_ev_found", "n_topics_used",
            "evidence_dia_ids", "evidence_topics", "topic_breakdown",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary: dict[str, float | int | None] = {}
    for full_cat, short in QTYPE_SHORT.items():
        sub = [r for r in rows if r["cat"] == full_cat and r["n_ev"] >= 2 and r["n_ev_found"] > 0]
        summary[f"{short}_n_q"] = len(sub)
        if sub:
            summary[f"{short}_topics_per_q"] = sum(r["n_topics_used"] for r in sub) / len(sub)
            summary[f"{short}_all1_pct"] = 100 * sum(1 for r in sub if r["n_topics_used"] == 1) / len(sub)
            summary[f"{short}_mean_n_ev"] = sum(r["n_ev"] for r in sub) / len(sub)
        else:
            summary[f"{short}_topics_per_q"] = None
            summary[f"{short}_all1_pct"] = None
            summary[f"{short}_mean_n_ev"] = None

    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[analyze-evidence] wrote {csv_path.name} + {json_path.name} for {run_dir.name}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path,
                   help="Per-run directory (e.g. outputs/experiments/<sweep>/<label>/)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="LoCoMo dataset JSON (default: benchmarks/locomo/data/locomo10.json)")
    args = p.parse_args()
    run_dir = args.run.resolve()
    if not run_dir.is_dir():
        print(f"[analyze-evidence] run dir not found: {run_dir}", file=sys.stderr)
        return 2
    if not args.data.exists():
        print(f"[analyze-evidence] data file not found: {args.data}", file=sys.stderr)
        return 2
    return analyze(run_dir, args.data.resolve())


if __name__ == "__main__":
    sys.exit(main())
