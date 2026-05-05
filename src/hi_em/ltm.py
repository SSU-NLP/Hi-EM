"""File-backed long-term memory.

Per-conversation layout (see ``context/01-hi-em-design.md §9.1``)::

    <root>/
    ├── <conv_id>.jsonl        # turn 기록 (append-only)
    └── <conv_id>.state.json   # topic 상태 latest snapshot (overwrite)

Schemas are caller-defined dicts; ``LTM`` is a thin I/O layer that does no
validation. Embedding values must be JSON-serializable (e.g. ``ndarray.tolist()``
applied by caller).

Thread safety: an instance-level :class:`threading.RLock` guards every
public method so concurrent ``RoundProcessor`` (background thread, reads)
and ``HiEM.handle_turn`` (main thread, append + state update) don't
interleave a partial line write with a read. The lock is held for the
entire I/O operation, including the file open/write/close.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class LTM:
    """Read/write API over per-conversation JSONL + state.json files."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # In-memory mirror of <conv_id>.jsonl. Populated lazily on first access
        # and kept in sync by ``append_turn``. Eliminates O(n²) re-parses when
        # ``RoundProcessor.process`` calls ``load_turns`` per promoted topic
        # every round during ``preload_history``.
        self._turns_mem: dict[str, list[dict[str, Any]]] = {}

    def _turns_path(self, conv_id: str) -> Path:
        return self.root / f"{conv_id}.jsonl"

    def _state_path(self, conv_id: str) -> Path:
        return self.root / f"{conv_id}.state.json"

    def append_turn(self, conv_id: str, turn: dict[str, Any]) -> None:
        """Append a single turn record to ``<conv_id>.jsonl`` and the in-memory mirror."""
        with self._lock:
            with self._turns_path(conv_id).open("a") as f:
                f.write(json.dumps(turn) + "\n")
            mem = self._turns_mem.get(conv_id)
            if mem is None:
                # Lazy populate: include any pre-existing turns on disk plus this one.
                self._turns_mem[conv_id] = self._read_turns_from_disk(conv_id)
            else:
                mem.append(turn)

    def update_state(self, conv_id: str, state: dict[str, Any]) -> None:
        """Overwrite ``<conv_id>.state.json`` with the latest topic snapshot."""
        with self._lock:
            self._state_path(conv_id).write_text(json.dumps(state, indent=2))

    def _read_turns_from_disk(self, conv_id: str) -> list[dict[str, Any]]:
        path = self._turns_path(conv_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def load_turns(
        self, conv_id: str, topic_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Return all turns (optionally filtered to one ``topic_id``)."""
        with self._lock:
            mem = self._turns_mem.get(conv_id)
            if mem is None:
                mem = self._read_turns_from_disk(conv_id)
                self._turns_mem[conv_id] = mem
            if topic_id is None:
                return list(mem)
            return [t for t in mem if t["topic_id"] == topic_id]

    def load_state(self, conv_id: str) -> dict[str, Any] | None:
        """Return the latest topic-state snapshot, or ``None`` if absent."""
        with self._lock:
            path = self._state_path(conv_id)
            if not path.exists():
                return None
            return json.loads(path.read_text())

    def list_conversations(self) -> list[str]:
        """Return all known conv_ids (presence of either jsonl or state.json)."""
        with self._lock:
            ids = set()
            for p in self.root.iterdir():
                if p.suffix == ".jsonl":
                    ids.add(p.stem)
                elif p.name.endswith(".state.json"):
                    ids.add(p.name[: -len(".state.json")])
            return sorted(ids)
