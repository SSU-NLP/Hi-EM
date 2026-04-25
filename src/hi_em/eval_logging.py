"""W&B logging helpers for Phase 4 evaluation.

Design:
    - ``wandb`` is an optional runtime dep. If ``WANDB_API_KEY`` is unset the
      ``WandbRun`` becomes a no-op so existing scripts keep working without
      configuration.
    - Tokenizer is loaded lazily and cached per-process. We use the **actual
      Qwen chat template** for ``prefill_tokens`` so the "same accuracy, far
      fewer tokens" efficiency claim is defensible (~15% heuristic error
      would not be).
    - ``aggregate_summary`` derives all summary scalars from a list of
      per-question records, so the same code path serves run-time and
      post-judge re-aggregation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_judge_yes_no(raw: str) -> bool:
    """Parse a yes/no judgment from a (possibly reasoning-style) LLM response.

    Robust to ``<think>...</think>`` blocks that may or may not close before
    ``max_tokens`` runs out:

        1. Strip closed think blocks.
        2. If a think block was opened but not closed, retain only the tail
           (best-effort: any final yes/no should be near the end).
        3. Search the **last** standalone yes/no token (the answer typically
           comes after reasoning).

    Returns False (not yes) when no yes/no can be extracted — safer default
    than guessing.
    """
    cleaned = _THINK_RE.sub("", raw)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned[-200:]
    cleaned = cleaned.strip().lower()
    tokens = [w.strip(".,!?:;\"'") for w in cleaned.split()]
    for w in reversed(tokens):
        if w in ("yes", "no"):
            return w == "yes"
    return False

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


_TOKENIZER_CACHE: dict[str, Any] = {}


def _tokenizer_for(model: str):
    if model not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(model)
    return _TOKENIZER_CACHE[model]


def count_prefill_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """Exact prompt token count via the model's chat template."""
    tok = _tokenizer_for(model)
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tok.encode(text))


class WandbRun:
    """Optional W&B run wrapper.

    No-op when ``wandb`` is missing or ``WANDB_API_KEY`` is not set.
    Sidecar file ``<output>.wandb-run-id`` carries the run id from the
    ``run_longmemeval.py`` invocation to the subsequent
    ``judge_longmemeval.py`` invocation so they update the same run.
    """

    def __init__(
        self,
        project: str,
        name: str,
        group: str,
        config: dict[str, Any],
        sidecar_path: Path | None = None,
        resume_id: str | None = None,
    ) -> None:
        self.enabled = bool(wandb is not None and os.environ.get("WANDB_API_KEY"))
        self._run = None
        if not self.enabled:
            return
        kwargs: dict[str, Any] = {
            "project": project, "name": name, "group": group, "config": config,
        }
        if resume_id:
            kwargs["id"] = resume_id
            kwargs["resume"] = "allow"
        self._run = wandb.init(**kwargs)
        if sidecar_path and not resume_id and self._run is not None:
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(self._run.id)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.log(data, step=step)

    def summary(self, **kvs: Any) -> None:
        if not self.enabled or self._run is None:
            return
        for k, v in kvs.items():
            self._run.summary[k] = v

    def log_table(self, name: str, columns: list[str], rows: list[list[Any]]) -> None:
        if not self.enabled or self._run is None or wandb is None:
            return
        self._run.log({name: wandb.Table(columns=columns, data=rows)})

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.finish()


def _percentiles(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "avg": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def aggregate_summary(per_q: list[dict[str, Any]]) -> dict[str, float]:
    """Build summary scalars from a list of per-question records.

    Recognized fields per record (all optional, missing → skipped):
        accuracy (0/1) · prefill_tokens · latency_sec · is_empty ·
        topic_revisit_hit · question_type
    """
    if not per_q:
        return {}
    out: dict[str, float] = {"n_questions": len(per_q)}

    accs = [r["accuracy"] for r in per_q if "accuracy" in r]
    if accs:
        out["accuracy_overall"] = float(np.mean(accs))

    for key in ("prefill_tokens", "latency_sec"):
        vals = [r[key] for r in per_q if key in r]
        if vals:
            for sk, sv in _percentiles(vals).items():
                out[f"{key}_{sk}"] = sv

    empties = [int(r.get("is_empty", False)) for r in per_q]
    out["error_or_empty_rate"] = float(np.mean(empties))

    revisits = [r["topic_revisit_hit"] for r in per_q if "topic_revisit_hit" in r]
    if revisits:
        out["topic_revisit_hit_rate"] = float(np.mean(revisits))

    by_qt: dict[str, list[float]] = {}
    for r in per_q:
        qt = r.get("question_type")
        if qt and "accuracy" in r:
            by_qt.setdefault(qt, []).append(r["accuracy"])
    for qt, vs in by_qt.items():
        out[f"accuracy_by_qtype/{qt}"] = float(np.mean(vs))

    return out
