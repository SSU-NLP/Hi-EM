"""Atomic file operations for experiment infrastructure.

Pattern: write to ``{path}.tmp`` first, then ``os.replace`` (POSIX atomic rename
on the same filesystem). Surrogate-safe encoding so LLM-emitted strings with
lone surrogates don't blow up the whole round.

See: ``research-experiment-infrastructure/SKILL.md`` §3, §9.1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def save_json(path: Path | str, data: Any) -> None:
    """Atomic JSON save with surrogate-safe utf-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(data, default=str, ensure_ascii=False, indent=2)
    tmp.write_bytes(payload.encode("utf-8", errors="replace"))
    os.replace(tmp, path)


def load_json(path: Path | str) -> Any:
    """Read JSON; raises FileNotFoundError if absent (caller decides default)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path | str, row: dict[str, Any]) -> None:
    """Append one JSON object as a line (newline-terminated).

    Not atomic — caller must guard with a lock if multi-threaded. Within a
    single thread, append is the standard jsonl write pattern (fail-recovery
    relies on round-level checkpoint, not per-row durability).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=str, ensure_ascii=False)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read all rows from a jsonl file. Empty list if file absent."""
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
