"""Experiment lifecycle — directory layout, round cycle, resume, sanity check.

Implements ``research-experiment-infrastructure/SKILL.md`` §1, §2, §4, §5, §9.8.

Domain mapping for Hi-EM Phase 4:
    Experiment       = (method, HP, dataset, model) 평가 1회
    Round            = N questions batch (default 50)
    Phase            = (1) run → hypothesis (2) judge → accuracy
    working_state    = none for now (stateless eval). Phase 2-Full STM 도입 시
                       LTM/STM 디렉토리가 채워짐.
    Snapshot         = round 단위 jsonl checkpoint (no binary state yet)

Key invariant:
    ``checkpoint.json`` is the **only** source of truth for "round complete".
    summary.json may be present but be stale; checkpoint.json being absent
    means the round must be re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hi_em.atomic_io import load_json, save_json

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def utc_timestamp() -> str:
    """Format ``YYYYMMDDTHHMMSS`` (UTC). Used in experiment IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def make_experiment_id(*parts: str, suffix: str = "", timestamp: str | None = None) -> str:
    """Build a slug-safe experiment id.

    Example::

        make_experiment_id("persistence", "oracle", "method=hi-em")
        # → "20260427T120000_persistence_oracle_method=hi-em"

    Per SKILL §2: model is NOT included by default — put it in ``session.json``
    when the whole sweep uses one model. Add it as a part only when comparing
    across models.
    """
    ts = timestamp or utc_timestamp()
    safe_parts = [p.replace("/", "__").replace(" ", "_") for p in parts if p]
    components = [ts, *safe_parts]
    if suffix:
        components.append(suffix)
    return "_".join(components)


# --- Schema definitions (versioned for migration) -----------------------

EXPERIMENT_JSON_SCHEMA_VERSION = 1
SUMMARY_JSON_SCHEMA_VERSION = 1


@dataclass
class ExperimentMeta:
    """Immutable experiment metadata. Written once, never mutated."""

    experiment_id: str
    session_id: str | None
    config: dict[str, Any]              # full config snapshot (CLI + .env + hiem.json)
    seeds: dict[str, int | None]        # data_seed, sampling_seed, env_seed, init_seed
    created_at: str
    git_sha: str | None = None
    schema_version: int = EXPERIMENT_JSON_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "config": self.config,
            "seeds": self.seeds,
        }


# --- Directory layout helpers --------------------------------------------

def experiment_dir(exp_id: str, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    return Path(root) / "experiments" / exp_id


def round_dir(exp_id: str, round_num: int, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    return experiment_dir(exp_id, root) / "rounds" / f"round_{round_num:03d}"


def session_dir(session_id: str, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    return Path(root) / "sessions" / session_id


# --- Lifecycle: create, mark round complete, find resumable -------------

def create_experiment(meta: ExperimentMeta, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    """Initialize an experiment directory + write immutable experiment.json."""
    exp_dir = experiment_dir(meta.experiment_id, root)
    if exp_dir.exists():
        # Already exists → treat as resume. Don't overwrite experiment.json.
        existing = load_json(exp_dir / "experiment.json")
        if existing.get("experiment_id") != meta.experiment_id:
            raise ValueError(
                f"Directory {exp_dir} exists with different experiment_id "
                f"{existing.get('experiment_id')!r}"
            )
        return exp_dir
    exp_dir.mkdir(parents=True, exist_ok=False)
    (exp_dir / "rounds").mkdir()
    (exp_dir / "checkpoints").mkdir()
    save_json(exp_dir / "experiment.json", meta.to_dict())
    return exp_dir


def mark_round_complete(
    exp_id: str, round_num: int, summary: dict[str, Any],
    root: Path | str = DEFAULT_RESULTS_ROOT,
) -> None:
    """Write summary.json then checkpoint.json (atomic, in order).

    SKILL §9.7 mandates this exact ordering: summary first (informational),
    checkpoint last (commit signal). A crash between the two leaves the round
    "uncompleted" → resume re-runs it (idempotent assumption required).
    """
    rd = round_dir(exp_id, round_num, root)
    rd.mkdir(parents=True, exist_ok=True)

    # 1. summary (with sanity-check version stamp)
    summary_with_version = {"schema_version": SUMMARY_JSON_SCHEMA_VERSION, **summary}
    save_json(rd / "summary.json", summary_with_version)

    # 2. checkpoint — the only source of truth for "round done"
    save_json(rd / "checkpoint.json", {
        "round": round_num,
        "complete": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # 3. experiment-level checkpoint pointer
    save_json(experiment_dir(exp_id, root) / "checkpoints" / "latest.json", {
        "last_completed_round": round_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def mark_experiment_complete(
    exp_id: str, total_rounds: int, root: Path | str = DEFAULT_RESULTS_ROOT,
) -> None:
    """Write completed.json — present iff the entire experiment succeeded."""
    save_json(experiment_dir(exp_id, root) / "completed.json", {
        "total_rounds": total_rounds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def find_resumable_experiment(exp_id: str, root: Path | str = DEFAULT_RESULTS_ROOT) -> int | None:
    """Return the highest completed round_num, or ``None`` for fresh start.

    SKILL §5: trust ``checkpoint.json`` only. summary.json without checkpoint
    means the round was interrupted between writes; redo it.
    """
    exp_dir = experiment_dir(exp_id, root)
    if (exp_dir / "completed.json").exists():
        return None  # fully done — caller should not resume
    rounds = exp_dir / "rounds"
    if not rounds.exists():
        return None

    completed: list[int] = []
    for d in rounds.iterdir():
        if not d.is_dir() or not d.name.startswith("round_"):
            continue
        ckpt = d / "checkpoint.json"
        if not ckpt.exists():
            continue
        try:
            data = load_json(ckpt)
            if data.get("complete"):
                completed.append(int(d.name.split("_", 1)[1]))
        except (ValueError, KeyError, OSError):
            continue
    return max(completed) if completed else None


# --- Sanity check on round summary (SKILL §9.8) -------------------------

def sanity_check_summary(
    summary: dict[str, Any], prev: dict[str, Any] | None = None,
) -> list[str]:
    """Return warning strings; empty list = all OK.

    Caller decides whether to log, alert, or abort. Not blocking.
    """
    warns: list[str] = []
    n = summary.get("n_processed", 0)
    if n == 0:
        warns.append("zero items processed")
        return warns

    primary = summary.get("primary_metric")
    if primary == 0:
        warns.append("0% primary metric — pipeline / parsing corruption?")

    if prev is not None and "primary_metric" in summary and "primary_metric" in prev:
        delta = abs(summary["primary_metric"] - prev["primary_metric"])
        if delta > 0.3:
            warns.append(
                f"primary metric jumped "
                f"{prev['primary_metric']:.2f} → {summary['primary_metric']:.2f}"
            )

    err_rate = summary.get("error_rate", 0)
    if err_rate > 0.1:
        warns.append(f"error_rate={err_rate:.2f} — investigate")

    return warns


# --- Session helpers (SKILL §2.5) ---------------------------------------

@dataclass
class Session:
    """Multi-experiment grouping. Common config + member exp_ids + tags."""

    session_id: str
    purpose: str
    common_config: dict[str, Any]
    experiments: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "purpose": self.purpose,
            "common_config": self.common_config,
            "experiments": self.experiments,
            "tags": self.tags,
        }

    def add_experiment(self, exp_id: str, overrides: dict[str, Any]) -> None:
        self.experiments.append({"exp_id": exp_id, "overrides": overrides})


def save_session(session: Session, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    sd = session_dir(session.session_id, root)
    sd.mkdir(parents=True, exist_ok=True)
    save_json(sd / "session.json", session.to_dict())
    return sd
