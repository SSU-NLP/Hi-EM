"""Early-stop watchdog for v3342_sweep_full.

Polls every 3s for new exit_code.txt files under the sweep dir. When a
*_r1 run completes, fetches its acc from summary.json and compares to a
dynamic threshold. If acc < threshold, writes a fake exit_code.txt=0 to
the r2/r3 directories so experiment.py's same-name skip rule short-
circuits them.

Threshold starts at the current best 3-seed mean accuracy (0.266 from
sigma=0.04 c=4) and updates upward whenever a newly-completed cell with
all three seeds beats it.

Baselines (rag_obs_*, v332_*) are ignored — they do not have a cell
structure compatible with this skip rule.

Run as:
  nohup uv run python outputs/experiments/2026-05-12_v3342_sweep_full/early_stop_watchdog.py \\
    > outputs/experiments/2026-05-12_v3342_sweep_full/watchdog.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SWEEP = Path("/home/namchailin/Hi-EM/outputs/experiments/2026-05-13_v3342_sigma_gpt41mini")
SWEEP_NAME = "2026-05-13_v3342_sigma_gpt41mini"
INITIAL_THRESHOLD = 0.266
POLL_SEC = 3
# Match v3342-prefixed runs (cm/sxxx/etc. label) but ignore baselines.
RUN_RE = re.compile(r"^(v3342_[a-z0-9]+)_r(\d+)$")


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_acc(run: str) -> float | None:
    sum_path = SWEEP / run / "results" / "experiments" / f"{SWEEP_NAME}_{run}" / "summary.json"
    if not sum_path.exists():
        return None
    try:
        with sum_path.open() as f:
            return float(json.load(f)["accuracy_overall"])
    except Exception as e:
        log(f"WARN: failed to read {sum_path}: {e}")
        return None


def cell_3seed_mean(cell: str) -> float | None:
    accs = []
    for r in (1, 2, 3):
        a = get_acc(f"{cell}_r{r}")
        if a is None:
            return None
        accs.append(a)
    return sum(accs) / 3


def find_completed() -> set[str]:
    """Runs with exit_code.txt == '0' (skipped or successfully finished)."""
    out: set[str] = set()
    for f in SWEEP.glob("*/exit_code.txt"):
        try:
            if f.read_text().strip() == "0":
                out.add(f.parent.name)
        except Exception:
            pass
    return out


def make_fake_skip(run: str) -> None:
    d = SWEEP / run
    # Don't touch a run that has already started (results/ exists or is running)
    if (d / "results").exists():
        log(f"  {run}: results/ already exists (in progress or done) — not skipping")
        return
    d.mkdir(exist_ok=True)
    ec = d / "exit_code.txt"
    if ec.exists():
        log(f"  {run}: exit_code.txt already exists, leaving alone")
        return
    ec.write_text("0\n")
    log_file = d / "run.log"
    log_file.write_text(
        f"=== SKIPPED by early-stop watchdog {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n"
        "Reason: same-cell r1 acc below dynamic threshold\n"
    )
    log(f"  [skip-written] {run}")


def sweep_alive() -> bool:
    try:
        subprocess.check_call(
            ["pgrep", "-f", f"scripts/experiment.py.*{SWEEP_NAME}"],
            stdout=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    threshold = INITIAL_THRESHOLD
    log(f"start: threshold={threshold:.3f}, poll={POLL_SEC}s, sweep_dir={SWEEP}")
    # CATCH-UP: treat already-completed runs as fresh events so we apply
    # the early-stop rule and bump threshold from past results.
    seen: set[str] = set()
    existing = find_completed()
    log(f"existing completed at start: {len(existing)} → entering catch-up pass")
    while True:
        if not sweep_alive():
            log("experiment.py no longer running — watchdog exiting")
            return 0
        cur = find_completed()
        new = sorted(cur - seen)
        for run in new:
            m = RUN_RE.match(run)
            if not m:
                log(f"  [non-v3342] {run} (ignored)")
                continue
            cell, r = m.group(1), int(m.group(2))
            acc = get_acc(run)
            log(f"  r-done {run} acc={acc}")
            if r == 1 and acc is not None and acc < threshold:
                log(f"  [EARLY-STOP] {run} acc={acc:.3f} < {threshold:.3f} → skipping {cell}_r2, {cell}_r3")
                for k in (2, 3):
                    make_fake_skip(f"{cell}_r{k}")
            mean = cell_3seed_mean(cell)
            if mean is not None and mean > threshold:
                log(f"  [threshold] cell {cell} 3-seed mean={mean:.3f} > {threshold:.3f} → bump")
                threshold = mean
        seen = cur
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
