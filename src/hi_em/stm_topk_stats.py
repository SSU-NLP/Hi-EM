"""STM top-K importance turn-count statistics aggregator + per-round logger.

Activated by setting ``HIEM_STM_TOPK_STATS_PATH`` to a JSON output path. During
a hi-em-full-v1 run, ``record_round`` is called once per round (after evict) and:

* Appends one JSONL line per round to ``<path>.rounds.jsonl`` with the full
  importance-sorted STM topic-size list (so per-round top1/top2/top3 gaps are
  recoverable post-hoc).
* Accumulates module-level lists for end-of-run aggregation.

At the end of the run, ``flush_aggregate`` reduces those lists into mean / max
/ min / variance and writes a single JSON to ``<path>``. The aggregate now
includes **per-rank** stats (``top1_per_round``, ``top2_per_round``,
``top3_per_round`` — distribution of the rank-1, rank-2, rank-3 topic size
**across rounds**) alongside the original combined ``top3``/``top4``/``top5``
(distribution of all top-K topic sizes pooled).

``run_experiment.py`` sets the env var only when the target file does not yet
exist; once it exists, recording is a no-op.

Thread safety: ``record_round`` and ``flush_aggregate`` share a single lock
(RoundProcessor may run rounds on a background thread). The first call within
a process truncates any stale ``<path>.rounds.jsonl`` so the sweep wrapper
does not need to clean both paths.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from statistics import mean, pvariance
from typing import Any

_LOCK = threading.Lock()
_KS = (3, 4, 5)
_TURN_COUNTS: dict[int, list[int]] = {k: [] for k in _KS}
_TOP_PER_ROUND: dict[int, list[int]] = {1: [], 2: [], 3: []}
_ROUNDS_RECORDED = 0
_JSONL_INITIALIZED = False


def _enabled_path() -> str | None:
    return os.environ.get("HIEM_STM_TOPK_STATS_PATH") or None


def _per_round_path(p: Path) -> Path:
    if p.suffix == ".json":
        return p.with_suffix(".rounds.jsonl")
    return p.with_suffix(p.suffix + ".rounds.jsonl")


def record_round(
    *,
    conv_id: str,
    round_idx: int,
    importance: dict[int, float],
    stm_turn_counts: dict[int, int],
) -> None:
    """Record one round's STM top-K topic sizes (per-round + aggregate)."""
    path = _enabled_path()
    if not path:
        return
    in_stm = {tid: imp for tid, imp in importance.items() if tid in stm_turn_counts}
    sorted_topics = sorted(in_stm.items(), key=lambda kv: -kv[1])
    sorted_sizes = [stm_turn_counts[tid] for tid, _ in sorted_topics]
    sorted_topic_ids = [tid for tid, _ in sorted_topics]

    with _LOCK:
        global _ROUNDS_RECORDED, _JSONL_INITIALIZED
        for k in _KS:
            _TURN_COUNTS[k].extend(sorted_sizes[:k])
        for rank in (1, 2, 3):
            if len(sorted_sizes) >= rank:
                _TOP_PER_ROUND[rank].append(sorted_sizes[rank - 1])
        _ROUNDS_RECORDED += 1

        jl = _per_round_path(Path(path))
        if not _JSONL_INITIALIZED:
            jl.parent.mkdir(parents=True, exist_ok=True)
            if jl.exists():
                jl.unlink()
            _JSONL_INITIALIZED = True

        rec = {
            "conv_id": conv_id,
            "round_idx": round_idx,
            "n_stm_topics": len(in_stm),
            "sorted_sizes": sorted_sizes,
            "sorted_topic_ids": sorted_topic_ids,
            "top1_size": sorted_sizes[0] if len(sorted_sizes) >= 1 else None,
            "top2_size": sorted_sizes[1] if len(sorted_sizes) >= 2 else None,
            "top3_size": sorted_sizes[2] if len(sorted_sizes) >= 3 else None,
        }
        with jl.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": 0.0, "max": 0, "min": 0, "variance": 0.0}
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "max": int(max(values)),
        "min": int(min(values)),
        "variance": float(pvariance(values)) if len(values) > 1 else 0.0,
    }


def flush_aggregate() -> str | None:
    """Write aggregated stats to the configured path. Returns path or None."""
    path = _enabled_path()
    if not path:
        return None
    with _LOCK:
        if _ROUNDS_RECORDED == 0:
            return None
        out = {
            "rounds_recorded": _ROUNDS_RECORDED,
            "top1_per_round": _stats(_TOP_PER_ROUND[1]),
            "top2_per_round": _stats(_TOP_PER_ROUND[2]),
            "top3_per_round": _stats(_TOP_PER_ROUND[3]),
            "top3": _stats(_TURN_COUNTS[3]),
            "top4": _stats(_TURN_COUNTS[4]),
            "top5": _stats(_TURN_COUNTS[5]),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(p)
        return str(p)
